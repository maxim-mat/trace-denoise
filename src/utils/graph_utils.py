from typing import Iterable, Tuple
from pm4py.objects.petri_net.utils import reachability_graph
import networkx as nx
import torch
# from node2vec import Node2Vec
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.utils import from_networkx
import torch_geometric
import pm4py
from utils.Config import Config
import ast


def get_node2vec_features(graph: nx.Graph, cfg: Config) -> Iterable:
    """

    :param cfg:
    :param graph:
    :return:
    """
    n2v = Node2Vec(graph, dimensions=cfg.n2v_dimensions, walk_length=cfg.n2v_walk_length, num_walks=cfg.n2v_num_walks,
                   p=cfg.n2v_p, q=cfg.n2v_q, workers=cfg.num_workers, seed=cfg.seed)
    n2v_model = n2v.fit(window=cfg.n2v_window, min_count=cfg.n2v_min_count, batch_words=cfg.n2v_batch_words)
    return [n2v_model.wv[n] for n in graph.nodes]


def get_onehot_features(graph: nx.Graph) -> Iterable:
    node_names = np.array([n[0] for n in graph.nodes(data=True)]).reshape(-1, 1)
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    return one_hot_encoder.fit_transform(node_names)


def get_index_features(graph: nx.Graph) -> Iterable:
    return [[x] for x in reversed(np.arange(graph.number_of_nodes()))]


def get_feature_collections(graph: nx.Graph, cfg: Config = None) -> Iterable[Iterable]:
    # n2v_features = get_node2vec_features(graph, cfg)
    index_features = get_index_features(graph)
    return [index_features]


def add_features_to_graph(graph: nx.Graph, feature_collections: Iterable[Iterable]) -> None:
    """
    modifies nodes in place with features under the attribute 'x'
    :param feature_collections:
    :param graph: graph to modify
    :return:
    """
    for node, *feature_collection in zip(graph.nodes(data=True), *feature_collections):
        # flatten each feature vector to single array
        node[1]['x'] = np.array([x for feature in feature_collection for x in feature])


def prepare_process_model_for_gnn(process_model: pm4py.PetriNet, init_marking: pm4py.Marking,
                                  final_marking: pm4py.Marking, cfg: Config) -> torch_geometric.data.Data:
    model_nx = pm4py.convert_petri_net_to_networkx(process_model, init_marking, final_marking)
    feature_collections = get_feature_collections(model_nx, cfg)
    add_features_to_graph(model_nx, feature_collections)
    data = from_networkx(model_nx)
    return data


def prepare_process_model_for_gnn_ordered(process_model: pm4py.PetriNet, init_marking: pm4py.Marking,
                                          final_marking: pm4py.Marking, cfg: Config) -> torch_geometric.data.Data:
    model_nx = pm4py.convert_petri_net_to_networkx(process_model, init_marking, final_marking)
    activity_nodes = {t.name for t in process_model.transitions if t.label is not None}
    activity_ids = {v: int(k) for k, v in cfg.activity_names.items()}
    node_index = max(activity_ids.items()) + 1
    for node in model_nx.nodes(data=True):
        if node[0] in activity_nodes:
            node[1]['x'] = [activity_ids[node[0]]]
    for node in model_nx.nodes(data=True):
        if node[0] not in activity_nodes:
            node[1]['x'] = [node_index]
            node_index += 1
    return from_networkx(model_nx)


def prepare_process_model_for_hetero_gnn(process_model: pm4py.PetriNet, init_marking: pm4py.Marking,
                                         final_marking: pm4py.Marking) -> Tuple[torch_geometric.data.HeteroData, Tuple]:
    model_nx = pm4py.convert_petri_net_to_networkx(process_model, init_marking, final_marking)
    activity_nodes = {t.name for t in process_model.transitions if t.label is not None}
    hetero_data = HeteroData()

    node_index = 0
    node_to_index_map = {}
    # this part is supposed to give activity nodes ids from 0 to number of activities and the rest to other nodes
    for node in model_nx.nodes(data=True):
        if node[0] in activity_nodes:
            node[1]['x'] = [node_index]
            node_to_index_map[node[0]] = node_index
            node_index += 1
    for node in model_nx.nodes(data=True):
        if node[0] not in activity_nodes:
            node[1]['x'] = [node_index]
            node_to_index_map[node[0]] = node_index
            node_index += 1

    sorted_nodes = sorted(model_nx.nodes(data=True),
                          key=lambda node: node[1]['x'])  # this should ensure that activity nodes are always first
    transition_features = torch.tensor(
        [node[1]['x'] for node in sorted_nodes if node[1]['attr']['type'] == 'transition'])
    place_features = torch.tensor([node[1]['x'] for node in sorted_nodes if node[1]['attr']['type'] == 'place'])
    hetero_data['transition'].x = transition_features.float()
    hetero_data['place'].x = place_features.float()
    edges_t_to_p = [(node_to_index_map[u], node_to_index_map[v]) for u, v in model_nx.edges() if
                    model_nx.nodes[u]['attr']['type'] == 'transition']
    edges_p_to_t = [(node_to_index_map[u], node_to_index_map[v]) for u, v in model_nx.edges() if
                    model_nx.nodes[u]['attr']['type'] == 'place']
    hetero_data['transition', 'transition_to_place', 'place'].edge_index = torch.tensor(edges_t_to_p).t()
    hetero_data['place', 'place_to_transition', 'transition'].edge_index = torch.tensor(edges_p_to_t).t()

    metadata = (['transition', 'place'],
                [('transition', 'transition_to_place', 'place'), ('place', 'place_to_transition', 'transition')])
    return hetero_data, metadata


def get_process_model_reachability_graph_transition_matrix(process_model: pm4py.PetriNet, init_marking: pm4py.Marking):
    rg = reachability_graph.construct_reachability_graph(process_model, init_marking)

    rg_nx = nx.DiGraph()

    for state in rg.states:
        rg_nx.add_node(state.name)

    transition_names = {tuple(s.strip(" '") for s in transition.name.strip("()").split(","))[1] for transition in
                        rg.transitions}
    transition_name_index = {name: idx for idx, name in enumerate(sorted(transition_names))}

    for transition in rg.transitions:
        transition_name = tuple(s.strip(" '") for s in transition.name.strip("()").split(","))
        rg_nx.add_edge(
            transition.from_state.name,
            transition.to_state.name,
            label=transition_name
        )

    nodes = sorted(rg_nx.nodes())
    num_transitions = len(transition_names)
    num_nodes = len(nodes)
    transition_matrix = np.zeros((num_transitions, num_nodes, num_nodes), dtype=int)

    for edge in rg_nx.edges(data=True):
        from_node = nodes.index(edge[0])
        to_node = nodes.index(edge[1])
        transition_name = edge[2]['label'][1]
        if transition_name in transition_name_index:
            transition_idx = transition_name_index[transition_name]
            transition_matrix[transition_idx, from_node, to_node] = 1
        else:
            raise RuntimeError(f"somehow, transition: {transition_name} was encountered but not indexed")

    return rg_nx, transition_matrix


def get_process_model_petri_net_transition_matrix(process_model: pm4py.PetriNet, init_marking: pm4py.Marking,
                                                  final_marking: pm4py.Marking):
    pn_nx = pm4py.convert_petri_net_to_networkx(process_model, init_marking, final_marking)
    transition_matrix = nx.adjacency_matrix(pn_nx).todense()

    return pn_nx, transition_matrix
