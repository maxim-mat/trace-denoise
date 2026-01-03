"""
Petri‑net discovery and optional conditional‑probability building.
"""

from typing import Any, List, Optional

import pandas as pd
import pm4py

from sktr_update.config import logger
from sktr_update.constants import CASE_COLUMN, LABEL_COLUMN

# From your existing RunningHorizon module:
from sktr_update.utils import build_conditioned_prob_dict
from sktr_update.classes import Place, Transition, Arc, PetriNet, Marking


def discover_petri_net(
    train_df: pd.DataFrame,
    non_sync_penalty: float
) -> Any:
    """
    Discover an inductive Petri net from training traces.
    """
    logger.debug("Preparing log for discovery.")
    prep = prepare_df_cols_for_discovery(train_df)
    net, init_m, fin_m = pm4py.discover_petri_net_inductive(prep)
    model = from_discovered_model_to_PetriNet(
        net,
        non_sync_move_penalty=non_sync_penalty,
        pm4py_init_marking=init_m,
        pm4py_final_marking=fin_m
    )
    logger.info("Petri net discovered.")
    return model


def build_probability_dict(
    train_df: pd.DataFrame,
    use_conditional_probs: bool,
    lambdas: Optional[List[float]]
) -> Optional[dict]:
    """
    Build a conditioned‐probability dictionary if requested.
    """
    if use_conditional_probs and lambdas:
        logger.debug("Building conditional probability dictionary.")
        prob_dict = build_conditioned_prob_dict(train_df, max_hist_len=len(lambdas))
        logger.info("Probability dictionary built.")
        return prob_dict
    logger.debug("Skipping probability dictionary.")
    return None


def from_discovered_model_to_PetriNet(discovered_model, non_sync_move_penalty=1, name='discovered_net', 
                                      cost_function=None, conditioned_prob_compute=False, 
                                      quiet_moves_weight=1e-8, sync_moves_weight=1e-6, return_mapping=False,
                                      pm4py_init_marking=None, pm4py_final_marking=None):
    """
    Convert a discovered pm4py model to a PetriNet object.
    Args:
        discovered_model: The pm4py discovered model to convert.
        non_sync_move_penalty: Penalty for non-synchronous moves. Default is 1.
        name: Name of the new PetriNet. Default is 'discovered_net'.
        cost_function: Cost function for the PetriNet. Default is None.
        conditioned_prob_compute: Flag for conditioned probability computation. Default is False.
        quiet_moves_weight: Weight for quiet moves. Default is 0.0000001.
        return_mapping: If True, returns the place_mapping in addition to the PetriNet object. Default is False.
    Returns:
        A PetriNet object with additional pm4py-related fields.
        If return_mapping is True, returns a tuple (PetriNet, place_mapping).
    """
    def create_arc(source, target):
        arc = Arc(source, target)
        petri_new_arcs.append(arc)
        return arc

    # Sort and create new places
    sorted_places = sort_places(discovered_model.places)
    places = [Place(p.name) for p in sorted_places]
    
    # Create place mapping
    place_mapping = {old_p: idx for idx, old_p in enumerate(sorted_places)}
    reverse_place_mapping = {idx: old_p for idx, old_p in enumerate(sorted_places)}
    
    # Create transitions
    transitions = [Transition(t.name, t.label, set(), set(), 'model', 
                              weight=quiet_moves_weight if t.label is None else non_sync_move_penalty) 
                   for t in discovered_model.transitions]
    # Create mappings for efficient lookup
    place_dict = {p.name: p for p in places}
    trans_dict = {t.name: t for t in transitions}
    # Create arcs
    petri_new_arcs = []
    for t in discovered_model.transitions:
        new_t = trans_dict[t.name]
        for arc in t.in_arcs:
            new_arc = create_arc(place_dict[arc.source.name], new_t)
            new_t.in_arcs.add(new_arc)
        for arc in t.out_arcs:
            new_arc = create_arc(new_t, place_dict[arc.target.name])
            new_t.out_arcs.add(new_arc)
    # Set transition names to labels if available
    for t in transitions:
        if t.label is not None:
            t.name = t.label
            
    # Create and setup new PetriNet
    new_PetriNet = PetriNet(name)
    new_PetriNet.add_places(places)
    new_PetriNet.add_transitions(transitions)
    new_PetriNet.init_mark = Marking((1,) + (0,) * (len(places) - 1))
    new_PetriNet.final_mark = Marking((0,) * (len(places) - 1) + (1,))
    new_PetriNet.arcs = petri_new_arcs
    new_PetriNet.cost_function = cost_function
    new_PetriNet.conditioned_prob_compute = conditioned_prob_compute
    new_PetriNet.epsilon = sync_moves_weight
    
    # Add pm4py-related fields
    new_PetriNet.pm4py_net = discovered_model
    new_PetriNet.pm4py_initial_marking = pm4py_init_marking
    new_PetriNet.pm4py_final_marking = pm4py_final_marking
    new_PetriNet.place_mapping = place_mapping
    new_PetriNet.reverse_place_mapping = reverse_place_mapping

    if return_mapping:
        return new_PetriNet, place_mapping
    else:
        return new_PetriNet


def prepare_df_cols_for_discovery(df):
    df_copy = df.copy()
    df_copy.loc[:, 'order'] = df_copy.groupby('case:concept:name').cumcount()
    
    if 'time:timestamp' in df_copy.columns:
        df_copy['time:timestamp'] = pd.to_datetime(df_copy['time:timestamp'])
    else:
        df_copy.loc[:, 'time:timestamp'] = pd.to_datetime(df_copy['order'])
    
    return df_copy


def sort_places(places):
    init_mark = [place for place in places if place.name == 'source']
    final_mark = [place for place in places if place.name == 'sink']
    inner_places = [place for place in places if place.name not in {'source', 'sink'}]
    inner_places_sorted = sorted(inner_places, key=lambda x: float(x.name[2:]))
    places_sorted = init_mark + inner_places_sorted + final_mark
    
    return places_sorted