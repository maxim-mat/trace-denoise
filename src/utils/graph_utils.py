from typing import Iterable

import networkx as nx
import torch
from node2vec import Node2Vec
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from torch_geometric.utils import from_networkx
import torch_geometric
import pm4py
from src.utils.Config import Config


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
    return torch.arange(graph.number_of_nodes(), dtype=torch.long)


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
