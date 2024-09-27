from collections import deque
import io
import copy
import numpy as np
import scipy as sp
from scipy.optimize import linprog
from heapq import heapify, heappush, heappop
import pm4py
import random
import pandas as pd
from statistics import mean
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
import warnings
from pm4py.objects.conversion.log import converter as log_converter
import pulp
from pulp import *
import pickle
import torch
import sympy
from random import sample


class Place:
    def __init__(self, name, in_arcs=None, out_arcs=None, properties={}):
        self.name = name
        self.in_arcs = set() if in_arcs is None else in_arcs
        self.out_arcs = set() if out_arcs is None else out_arcs
        self.properties = properties

    def __repr__(self):
        return self.name


class Transition:
    def __init__(self, name, label, in_arcs=None, out_arcs=None, move_type=None, prob=None, weight=None,
                 location_idx=None,
                 cost_function=None, properties={}):
        self.name = name
        self.label = label
        self.in_arcs = set() if in_arcs is None else in_arcs
        self.out_arcs = set() if out_arcs is None else out_arcs
        self.move_type = move_type
        self.prob = prob
        self.cost_function = cost_function
        self.weight = self.__initialize_weight(weight)
        self.properties = properties
        self.location_idx = location_idx

    def __repr__(self):
        return self.name

    def __initialize_weight(self, weight):
        if weight is not None:
            return weight

        if self.prob == 0:
            return np.inf

        if self.cost_function is None:
            return 0 if self.move_type == 'sync' else 1

        return self.cost_function(self.prob)


class Arc:
    def __init__(self, source, target, weight=1, properties={}):
        self.source = source
        self.target = target
        self.weight = weight
        self.properties = properties

    def __repr__(self):
        return self.source.name + ' -> ' + self.target.name


class Marking:
    def __init__(self, places=None):
        self.places = tuple([0]) if places is None else places

    def __repr__(self):
        return str(self.places)

    def __eq__(self, other):
        return self.places == other.places

    def __hash__(self):
        return hash(self.places)


class Node:
    def __init__(self, marking):
        self.marking = marking
        self.neighbors = set()

    def __repr__(self):
        return str(self.marking)

    def add_neighbor(self, node, transition):
        self.neighbors.add((node, transition))


class Edge:
    def __init__(self, name, source_marking, target_marking, move_type):
        self.name = name
        self.source_marking = source_marking
        self.target_marking = target_marking
        self.move_type = move_type

    def __repr__(self):
        return f'{self.source_marking} -> {self.name} -> {self.target_marking}'


class Graph:
    def __init__(self, nodes=None, edges=None, starting_node=None, ending_node=None):
        self.nodes = set() if nodes is None else nodes
        self.edges = set() if edges is None else edges
        self.starting_node = starting_node
        self.ending_node = ending_node

    def __repr__(self):
        return f'Nodes:{self.nodes}, \n edges:{self.edges}'

    def __get_markings(self):
        return set([node.marking for node in self.nodes])

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, edge):
        self.edges.add(edge)


class search_node:

    def __init__(self, ancestor, transition_to_ancestor, marking, dist_from_origin, solution_vec=None,
                 heuristic_distance=np.inf,
                 have_exact_known_solution=False, have_estimated_solution=False, n_explained_events=0,
                 disappointing=None):

        self.ancestor = ancestor
        self.transition_to_ancestor = transition_to_ancestor
        self.marking = marking
        self.dist_from_origin = dist_from_origin
        self.heuristic_distance = heuristic_distance
        self.total_distance = dist_from_origin + heuristic_distance
        self.solution_vec = solution_vec
        self.have_exact_known_solution = have_exact_known_solution
        self.have_estimated_solution = have_estimated_solution
        self.n_explained_events = n_explained_events
        self.disappointing = disappointing

    def __lt__(self, other):

        if self.disappointing is None:
            if self.have_exact_known_solution != other.have_exact_known_solution:
                return self.have_exact_known_solution > other.have_exact_known_solution

            if self.solution_vec is not None and other.solution_vec is not None:
                if self.total_distance == other.total_distance:
                    return sum(self.solution_vec) < sum(other.solution_vec)

            return self.total_distance < other.total_distance

        else:
            return self.total_distance < other.total_distance

    def __repr__(self):
        return f'Node: {self.marking}, dist:{self.dist_from_origin}'


class search_node_new:
    def __init__(self, marking, dist=np.inf, ancestor=None, transition_to_ancestor=None, path_prefix=None,
                 trace_activities_multiset=None, heuristic_distance=None, total_model_moves=0):
        self.dist = dist
        self.ancestor = ancestor
        self.transition_to_ancestor = transition_to_ancestor
        self.marking = marking
        self.path_prefix = path_prefix if path_prefix is not None else ''
        self.trace_activities_multiset = trace_activities_multiset
        # Initialize heuristic_distance with a default of 0 if not provided
        self.heuristic_distance = heuristic_distance if heuristic_distance is not None else 0
        self.total_model_moves = total_model_moves

    def __lt__(self, other):
        # First compare based on the sum of dist and heuristic_distance
        if (self.dist + self.heuristic_distance) == (other.dist + other.heuristic_distance):
            # If they are equal, compare based on total_model_moves (larger first)
            return self.total_model_moves > other.total_model_moves
        return (self.dist + self.heuristic_distance) < (other.dist + other.heuristic_distance)

    def __repr__(self):
        return f'Node: {self.marking}, dist: {self.dist}, heuristic: {self.heuristic_distance}'


class PetriNet:
    def __init__(self, name, places=None, transitions=None, arcs=None, trace_len=None, properties={}):
        self.name = name
        self.transitions = list() if transitions is None else transitions
        self.places = list() if places is None else places
        self.arcs = list() if arcs is None else arcs
        self.properties = properties
        self.init_mark = None
        self.final_mark = None
        self.reachability_graph = None
        self.places_indices = {} if places is None else {self.places[i].name: i for i in range(len(self.places))}
        self.transitions_indices = {} if transitions is None else {self.transitions[i].name: i for i in
                                                                   range(len(self.transitions))}
        self.dis_associated_indices = None
        self.trace_transitions = None
        self.transitions_weights = list() if transitions is None else np.array([t.weight for t in transitions])
        self.trace_len = trace_len
        self.cost_function = None

    def update_transitions_weights(self):
        transitions_weights = np.array([t.weight for t in self.transitions])

        return transitions_weights

    #     def compute_dis_associated_indices_for_heuristic(self):
    #         transition_labels_set = {transition.label for transition in self.transitions if transition.move_type == 'trace'}
    #         label_indices_dict = {}

    #         for label in transition_labels_set:
    #             dis_associated_indices = [idx for idx, transition in enumerate(self.transitions) if transition.label != label or
    #                                      transition.label == label and transition.move_type =='model']

    #             label_indices_dict[label] = dis_associated_indices

    #         print(f'number of possible indices for each transition label within the trace: {[len(self.transitions)-len(label_indices_dict[label]) for label in label_indices_dict.keys()]}')
    #         return label_indices_dict

    def compute_conditioned_weight(self, path_prefix, transition, prob_dict, lamda=0.5):

        if prob_dict is None:
            return transition.weight

            # print(f'prob dict = {prob_dict}')
        if transition.label is None:
            #             print(f'Transition label=None thus returning weight of 0')
            return 0

        #         print(f'original trans weight={transition.weight}')
        transition_weight = transition.weight
        transition_label = transition.label
        full_path = path_prefix + transition_label
        #         print(f'The full path including the transition label={full_path}')

        if path_prefix == '':
            return transition_weight

        if full_path in prob_dict:
            #             print(f'full path={full_path} is in the prob dict and conditioned weight= {0.5*(1-prob_dict[full_path]) + 0.5*transition_weight}')
            #             return (1-lamda)*(1-prob_dict[full_path]) + lamda*transition_weight
            return (1 - lamda) * ((1 - prob_dict[full_path]) * transition_weight) + lamda * transition_weight

        #         print(f'full path={full_path} is not in the prob dict.. sorry.. ')
        longest_prefix = self.find_longest_prefix(full_path, prob_dict)

        if longest_prefix:
            #             print(f'longest prefix={longest_prefix} is in dict! The conditioned weight= {0.5*(1-prob_dict[longest_prefix]) + 0.5*transition_weight}')
            #             return (1-lamda)*(1-prob_dict[longest_prefix]) + lamda*transition_weight
            return (1 - lamda) * ((1 - prob_dict[longest_prefix]) * transition_weight) + lamda * transition_weight
        #         print(f'no prefix exists for {full_path}..conditioned weight= {0.5 + 0.5*transition_weight}')
        #         return (1-lamda) + lamda*transition_weight
        return transition_weight

    def find_longest_prefix(self, full_path, prob_dict):
        longest_prefix = None
        for i in range(len(full_path) - 1):
            if full_path[i:] in prob_dict:
                return full_path[i:]
        #             print(f'prefix={full_path[i:]} is not in the dict')

        return longest_prefix

    def compute_disassosiate_indices_for_heuristic(self):
        disassociate_indices_dict = defaultdict(list)
        for i in range(self.trace_len):
            disassociate_indices = [idx for idx, transition in enumerate(self.transitions) if
                                    transition.location_idx != i]
            disassociate_indices_dict[i] = disassociate_indices
        return disassociate_indices_dict

    def add_places(self, places):
        if isinstance(places, list):
            self.places += places

        else:
            self.places.append(places)

        self.__update_indices_p_dict(places)

    def _find_available_transitions(self, mark_tuple):
        '''Input: tuple
           Output: list'''

        available_transitions = []
        for transition in self.transitions:
            if self.__check_transition_prerequesits(transition, mark_tuple):
                available_transitions.append(transition)

        return available_transitions

    def _fire_transition(self, mark, transition):
        '''Input: Mark object, Transition object
        Output: Marking object'''

        subtract_mark = [0] * len(mark.places)
        for arc in transition.in_arcs:
            place_idx = self.places_indices[arc.source.name]
            subtract_mark[place_idx] -= arc.weight

        add_mark = [0] * len(mark.places)
        for arc in transition.out_arcs:
            place_idx = self.places_indices[arc.target.name]
            add_mark[place_idx] += arc.weight

        new_mark = tuple([sum(x) for x in zip(list(mark.places), subtract_mark, add_mark)])
        for elem in new_mark:
            if elem < 0:
                print(
                    f'the mark was: {mark} and I subtract the following values: {subtract_mark} and adding these: {add_mark} \
                which results in this: {new_mark} and all this sh!t was by using this transition: {transition.name}')
        new_mark_obj = Marking(new_mark)

        return new_mark_obj

    def add_transitions(self, transitions):
        if isinstance(transitions, list):
            self.transitions += transitions

        else:
            self.transitions.append(transitions)

        self.__update_indices_t_dict(transitions)
        #         self.dis_associated_indices = self.compute_disassosiate_indices_for_heuristic()
        self.transitions_weights = self.update_transitions_weights()

    def compute_heuristic(self, incidence_mat, curr_marking):
        curr_marking_vec = self.convert_marking_to_np_vector(curr_marking)
        final_marking_vec = self.convert_marking_to_np_vector(self.final_mark)
        obj = self.transitions_weights
        rhs_eq = final_marking_vec - curr_marking_vec
        opt = linprog(c=obj, A_eq=incidence_mat, b_eq=rhs_eq, method="revised simplex")

        if opt['success'] == True:
            #             print(f'scipy heuristic is a success! heuristic distance={opt.fun}')

            return opt.fun, opt.x

        print('problems in heuristic.. no solution!!!!')
        return float('inf'), None

    #    def astar(self):
    #         # PROBLEM!!!! - NO heapify implemented and no update to nodes which were found by a shorter path!!!
    #         distance_min_heap = []
    #         heapify(distance_min_heap)
    #         init_node = search_node(None, None, self.init_mark, 0, 0)
    #         heappush(distance_min_heap, init_node)
    #         incidence_mat = self.compute_incidence_matrix()

    #         while distance_min_heap:
    #             min_dist_node = heappop(distance_min_heap)
    #             print(f'min node has total distance of {min_dist_node.total_distance}')
    #             if min_dist_node.marking == self.final_mark:
    #                 print('final marking has been reached!!')
    #                 print(f'n_nodes left in heap: {len(distance_min_heap)}')
    #                 break

    #             for transition in self.__find_available_transitions(min_dist_node.marking.places):
    #                 new_mark = self.__fire_transition(min_dist_node.marking, transition)
    #                 heuristic_distance = self.compute_heuristic(incidence_mat, new_mark)
    #                 new_node = search_node(min_dist_node, transition, new_mark,
    #                                        min_dist_node.dist_from_origin + transition.weight, heuristic_distance)
    #                 heappush(distance_min_heap, new_node)

    #         curr_node = min_dist_node
    #         path = []

    #         while curr_node.ancestor:
    #             path.append(curr_node.transition_to_ancestor.name)
    #             curr_node = curr_node.ancestor

    #         return path[::-1], min_dist_node.dist_from_origin

    def initialize_min_dist_node(self, k, incidence_mat, consump_mat):
        heuristic_distance, sol_vec = self.compute_heuristic_extended(k, self.transitions_weights,
                                                                      np.array(self.init_mark.places),
                                                                      np.array(self.final_mark.places),
                                                                      incidence_mat, consump_mat)

        init_node = search_node(ancestor=None, transition_to_ancestor=None, marking=self.init_mark, dist_from_origin=0,
                                solution_vec=sol_vec,
                                heuristic_distance=heuristic_distance, have_exact_known_solution=True)
        #         print(f'The initial starting node is initializd with the following values given k_values={k}: heuristic_distance={init_node.heuristic_distance}, total distance: {init_node.total_distance}')
        return init_node

    def astar_extended(self, k=None, s=0):
        if k is None:
            k = set()
        #         print(f'starting the search algorithm.. the k set is: {k}')

        visited_markings = set()
        visited_markings_distance_dict = {}
        distance_min_heap = []
        heapify(distance_min_heap)
        # these two should be new attributes in the petri net instead of copying them each time
        incidence_mat = self.compute_incidence_matrix()  # Need to check the condition (t,p) not in F
        #         print(f' sum of incidence mat values = {np.sum(incidence_mat)}')
        #         print(f'number of non zero elems in incidence mat: {np.count_nonzero(incidence_mat)}')
        consump_mat = self.compute_consumption_matrix()
        #         print(f' consump mat sum: {np.sum(consump_mat)}')
        init_node = self.initialize_min_dist_node(k, incidence_mat, consump_mat)
        heappush(distance_min_heap, init_node)
        visited_markings_distance_dict[self.init_mark] = init_node
        final_min_dist = np.inf
        node = None

        while distance_min_heap:
            need_heapify = False
            min_dist_node = heappop(distance_min_heap)

            #             print(f'The min dist node has distance so far={round(min_dist_node.dist_from_origin, 12)}, estimated_distance={min_dist_node.heuristic_distance}, have_exact_known_solution={min_dist_node.have_exact_known_solution}, have_estimated_solution={min_dist_node.have_estimated_solution}')
            #             if min_dist_node.have_exact_known_solution:
            #                 print([(self.transitions[idx].name, round(num,12)) for idx, num in enumerate(min_dist_node.solution_vec) if num>0])
            #             else:
            #                 print('min dist node does not have an exact solution')
            #             print(f'The heap has the following (except the min one which was popped) nodes within (node_total_distance, have_exact_sol, have_estimated_sol) \n:{[(round(node.total_distance,2), node.have_exact_known_solution, node.have_estimated_solution) for node in distance_min_heap]}')

            if min_dist_node.marking == self.final_mark:
                break

            if min_dist_node.have_estimated_solution:
                max_events_explained = max(min_dist_node.n_explained_events, s)
                if max_events_explained not in k:
                    k.add(max_events_explained)
                    return self.astar_extended(k, s=0)

                #                 heuristic_distance_test, sol_vec_test = self.compute_heuristic(incidence_mat, min_dist_node.marking)
                #                 temp_node = min_dist_node
                #                 temp_path = []

                #                 while temp_node.ancestor:
                #                     temp_path.append(temp_node.transition_to_ancestor.name)
                #                     temp_node = temp_node.ancestor

                heuristic_distance, sol_vec = self.compute_heuristic_extended(None, self.transitions_weights,
                                                                              np.array(min_dist_node.marking.places),
                                                                              np.array(self.final_mark.places),
                                                                              incidence_mat, consump_mat,
                                                                              init_node_comp=False,
                                                                              n_explained_events=min_dist_node.n_explained_events)
                #                 print(f'path to the node:{temp_path[::-1]}')
                #                 print(f'returning from mid-run heuristic values heuristic_distance={heuristic_distance} compared to previous estimation of {min_dist_node.heuristic_distance} and sol_vec={[(self.transitions[idx].name, round(num,3), round(self.transitions[idx].weight,3)) for idx, num in enumerate(sol_vec) if num>0]}')
                #                 if round(heuristic_distance_test) != round(heuristic_distance):
                #                     raise ValueError(f'heuristic_distance_test={heuristic_distance_test}, heuristic_distance={heuristic_distance}')

                min_dist_node.disappointing = True

                if sol_vec is not None:
                    #                     print('sol vec is not none')
                    min_dist_node.have_exact_known_solution = True
                    min_dist_node.have_estimated_solution = False
                    min_dist_node.solution_vec = sol_vec

                else:
                    #                     print('sol vec is none!!!!!!!')
                    min_dist_node.heuristic_distance = np.inf
                    min_dist_node.total_distance = min_dist_node.dist_from_origin + min_dist_node.heuristic_distance
                    min_dist_node.solution_vec = None
                    heappush(distance_min_heap, min_dist_node)
                    continue

                if heuristic_distance > min_dist_node.heuristic_distance:
                    min_dist_node.heuristic_distance = heuristic_distance
                    min_dist_node.total_distance = min_dist_node.dist_from_origin + min_dist_node.heuristic_distance
                    heappush(distance_min_heap, min_dist_node)
                    continue

                ## new ##
            #                 else:
            #                     min_dist_node.heuristic_distance = heuristic_distance
            #                     min_dist_node.total_distance = min_dist_node.dist_from_origin + min_dist_node.heuristic_distance
            ##########

            s = max(s, min_dist_node.n_explained_events)
            visited_markings.add(min_dist_node.marking)

            for transition in self.__find_available_transitions(min_dist_node.marking.places):
                #                 print(f'currently exploring tranisition= {transition.name}, weight={transition.weight}')
                new_mark = self.__fire_transition(min_dist_node.marking, transition)
                need_push_node = False
                transition_idx = self.transitions_indices[transition.name]

                if new_mark not in visited_markings:
                    dist_to_node = min_dist_node.dist_from_origin + transition.weight
                    sol_vec = np.array(min_dist_node.solution_vec)

                    if new_mark in visited_markings_distance_dict:
                        if ((dist_to_node > visited_markings_distance_dict[new_mark].dist_from_origin) or (
                                dist_to_node == visited_markings_distance_dict[new_mark].dist_from_origin and sol_vec[
                            transition_idx] < 0.999)):
                            #                             if dist_to_node > visited_markings_distance_dict[new_mark].dist_from_origin:
                            #                                 print(f'distance to node through transition {transition} is {dist_to_node} but current distance is {visited_markings_distance_dict[new_mark].dist_from_origin} and thus no update')
                            #                             if dist_to_node == visited_markings_distance_dict[new_mark].dist_from_origin and sol_vec[transition_idx] < 0.999:
                            #                                 print(f'same distance from both paths {dist_to_node} = {visited_markings_distance_dict[new_mark].dist_from_origin} but no heuristic since value of sol vel in idx is {sol_vec[transition_idx]}')
                            continue

                    if new_mark not in visited_markings_distance_dict:
                        need_push_node = True
                        node = search_node(min_dist_node, transition, new_mark, dist_to_node)
                        visited_markings_distance_dict[new_mark] = node


                    else:
                        #                         print(f'the marking already exists but using transition {transition.name} with the value of {visited_markings_distance_dict[new_mark].dist_from_origin} results in a shorted way to the marking thus updating the markings distance to: {dist_to_node} and the new dady transition is {transition.name}')
                        #                         print('Taking existing node and updating its value')
                        need_heapify = True
                        node = visited_markings_distance_dict[new_mark]
                        node.dist_from_origin = dist_to_node
                        node.ancestor = min_dist_node
                        node.transition_to_ancestor = transition
                    #                         print(f'number of markings within dict={len(visited_markings_distance_dict)} while number of unique nodes is:{len(set([id(value) for value in visited_markings_distance_dict.values()]))}')
                    node.heuristic_distance = max(0, min_dist_node.heuristic_distance - transition.weight)
                    node.total_distance = node.heuristic_distance + node.dist_from_origin

                    #                     need_push_node = True
                    #                     node = search_node(min_dist_node, transition, new_mark, dist_to_node) # This line sohuld be checked!!!!!
                    #                     visited_markings_distance_dict[new_mark] = node
                    #                     node.heuristic_distance = max(0, min_dist_node.heuristic_distance - transition.weight)
                    #                     node.total_distance = node.heuristic_distance + node.dist_from_origin

                    if min_dist_node.solution_vec[transition_idx] >= 0.999:
                        #                         print(f'since transition {transition.name}, idx={transition_idx} have value bigger than 1 within the heuristic we reuse existing sol vec')
                        new_sol_vec = np.array(min_dist_node.solution_vec, copy=True)
                        if min_dist_node.solution_vec[transition_idx] >= 1:
                            new_sol_vec[transition_idx] -= 1
                        else:
                            new_sol_vec[transition_idx] = 0
                        node.solution_vec = new_sol_vec
                        #                         print(f'the new sol vec is: {[(self.transitions[idx].name, round(num,2)) for idx, num in enumerate(new_sol_vec) if num>0]}')
                        node.have_exact_known_solution = True
                        node.have_estimated_solution = False

                    else:
                        #                         print(f'transition {transition.name} with idx={transition_idx} does not appear within the solution vec thus no sol vec and estimated_sol=True')
                        #                         print(f'the value of tranisiton={transition.name} with idx={transition_idx} inside the sol vector is={min_dist_node.solution_vec[transition_idx]}')
                        #                         print(f'Here is the solution vector where the transition does not appear in: \n {[(self.transitions[idx].name, round(num,12), idx) for idx, num in enumerate(min_dist_node.solution_vec) if num>0]}')
                        node.have_exact_known_solution = False
                        node.have_estimated_solution = True
                        node.solution_vec = None

                    node.disappointing = min_dist_node.disappointing

                    if transition.move_type in {'sync', 'trace'}:
                        node.n_explained_events = min_dist_node.n_explained_events + 1

                    else:
                        node.n_explained_events = min_dist_node.n_explained_events

                    #                     print(f'the value of need_push={need_push_node}')
                    if need_push_node:
                        #                         print('pushing node inside heap!!')
                        heappush(distance_min_heap, node)
            #                     print(f'after updating the node the new values are: distance_from_origin={node.dist_from_origin}, heuristic_value={node.heuristic_distance}, total_distance={node.total_distance}')
            if need_heapify:
                heapify(distance_min_heap)

        #             print(f'node daddy id before pop:{id(node.ancestor)}')
        #             daddy_id_before = id(node.ancestor)

        curr_node = min_dist_node
        path = []

        while curr_node.ancestor:
            path.append(curr_node.transition_to_ancestor.name)
            curr_node = curr_node.ancestor

        print(f'Optimal alignment cost: {min_dist_node.dist_from_origin}')  # , \n Optimal alignment: \n {path[::-1]}')
        return path[::-1], min_dist_node.dist_from_origin

    def compute_heuristic_extended(self, k_set, c, m_i, m_f, incidence_mat, consump_mat, init_node_comp=True,
                                   n_explained_events=None):
        #         print('Entering extended heuristic...')
        if self.dis_associated_indices is None:
            self.dis_associated_indices = self.compute_disassosiate_indices_for_heuristic()
        #             print([len(self.dis_associated_indices[key]) for key in self.dis_associated_indices.keys()])

        if k_set is None:
            k_set = set()

        model = LpProblem("Heuristic-Estimator", LpMinimize)
        if init_node_comp:
            n_ys = len(k_set) + 1
            n_xs = n_ys + 1

            X_mat = np.array([str(i) + '_' + str(j) for i in range(n_xs) for j in range(1, incidence_mat.shape[1] + 1)])
            Y_mat = np.array(
                [str(i) + '_' + str(j) for i in range(1, n_ys + 1) for j in range(1, incidence_mat.shape[1] + 1)])

            X_variables = LpVariable.matrix("X", X_mat, lowBound=0)  # constraint 3:  cat='Integer'
            Y_variables = LpVariable.matrix("Y", Y_mat, lowBound=0)  # constrainrt 4: cat='Binary'

            allocation_x = np.array(X_variables).reshape(n_xs, incidence_mat.shape[1])
            allocation_y = np.array(Y_variables).reshape(n_ys, incidence_mat.shape[1])

            # Objective Function
            obj_func = lpSum(c @ allocation_x.T) + lpSum(c @ allocation_y.T)
            model += obj_func

            # Constraint 1
            total_allocation = allocation_x.T.sum(axis=1) + allocation_y.T.sum(axis=1)
            for i in range(incidence_mat.shape[0]):
                model += lpSum([m_i[i], incidence_mat[i, :] @ total_allocation]) == m_f[i]

            # Constraint 2
            for j in range(1, n_xs):
                curr_alloc_x = allocation_x[:j, :]
                curr_alloc_y = allocation_y[:j - 1, :] if j != 1 else np.zeros(allocation_y.shape[1]).reshape(-1, 1)
                curr_total_allocation = curr_alloc_x.T.sum(axis=1) + curr_alloc_y.T.sum(axis=1)
                curr_alloc_y_j = allocation_y[j - 1, :]

                for i in range(incidence_mat.shape[0]):
                    model += lpSum(
                        [m_i[i], incidence_mat[i, :] @ curr_total_allocation, consump_mat[i, :] @ curr_alloc_y_j]) >= 0

                    # Constraint 5
            for row_idx, trace_location_idx in enumerate(sorted(list(k_set))):
                model += lpSum(
                    allocation_y[row_idx + 1][i] for i in self.dis_associated_indices[trace_location_idx]) == 0
            model += lpSum(allocation_y[0][i] for i in self.dis_associated_indices[0]) == 0

            # Constraint 6
            ones_vec = np.ones(incidence_mat.shape[1])
            for row_idx in range(len(allocation_y)):
                model += lpSum(ones_vec @ allocation_y[row_idx].T) == 1



        else:
            n_xs = 1
            X_mat = np.array([str(i) + '_' + str(j) for i in range(n_xs) for j in range(1, incidence_mat.shape[1] + 1)])
            X_variables = LpVariable.matrix("X", X_mat, lowBound=0)
            allocation_x = np.array(X_variables).reshape(n_xs, incidence_mat.shape[1])

            obj_func = lpSum(c @ allocation_x.T)
            model += obj_func

            total_allocation = allocation_x.T.sum(axis=1)
            for i in range(incidence_mat.shape[0]):
                model += lpSum([m_i[i], incidence_mat[i, :] @ total_allocation]) == m_f[i]

                # Solving the optimization problem and returning the results
        model.solve(PULP_CBC_CMD());
        if init_node_comp:
            x_res = np.array([val.value() for val in X_variables]).reshape(n_xs, incidence_mat.shape[1]).T
            y_res = np.array([val.value() for val in Y_variables]).reshape(n_ys, incidence_mat.shape[1]).T
            sol_vec = x_res.sum(axis=1) + y_res.sum(axis=1)

        else:
            x_res = np.array([val.value() for val in X_variables]).reshape(n_xs, incidence_mat.shape[1]).T
            sol_vec = x_res.sum(axis=1)

        heuristic_distance = model.objective.value()

        #         print(f'LP status: {LpStatus[model.status]}')

        if LpStatus[model.status] != 'Optimal':
            print(f'returning inf distance and None vector')
            return np.inf, None

        return heuristic_distance, sol_vec

    def compute_incidence_matrix(self):
        length = len(self.places)
        width = len(self.transitions)
        inc_mat = np.zeros((length, width))

        for arc in self.arcs:
            if isinstance(arc.source, Place):
                i, j = self.places_indices[arc.source.name], self.transitions_indices[arc.target.name]
                inc_mat[i, j] -= 1

            else:
                j, i = self.transitions_indices[arc.source.name], self.places_indices[arc.target.name]
                inc_mat[i, j] += 1

        return inc_mat

    def compute_consumption_matrix(self):
        length = len(self.places)
        width = len(self.transitions)
        consump_mat = np.zeros((length, width))

        for arc in self.arcs:
            if isinstance(arc.source, Place):
                i, j = self.places_indices[arc.source.name], self.transitions_indices[arc.target.name]
                consump_mat[i, j] = -1

        return consump_mat

    def generate_location_dict(self, iterable):
        return {item.name: idx for idx, item in enumerate(iterable)}

    def generate_nonsync_arcs_for_sync_product(self, model_places, model_transitions, trace_places, trace_transitions,
                                               trace_model):

        model_p_dict = self.generate_location_dict(model_places)
        model_t_dict = self.generate_location_dict(model_transitions)

        trace_p_dict = self.generate_location_dict(trace_places)
        trace_t_dict = self.generate_location_dict(trace_transitions)

        model_arcs = [Arc(model_places[model_p_dict[a.source.name]], model_transitions[model_t_dict[a.target.name]])
                      if type(a.source) == Place else Arc(model_transitions[model_t_dict[a.source.name]],
                                                          model_places[model_p_dict[a.target.name]]) for a in self.arcs]

        trace_arcs = [Arc(trace_places[trace_p_dict[a.source.name]], trace_transitions[trace_t_dict[a.target.name]])
                      if type(a.source) == Place else Arc(trace_transitions[trace_t_dict[a.source.name]],
                                                          trace_places[trace_p_dict[a.target.name]]) for a in
                      trace_model.arcs]

        return model_arcs, trace_arcs

    def convert_marking_to_np_vector(self, marking):
        return np.array(marking.places)

    def assign_non_sync_arcs_to_transitions(self, model_places, trace_places, model_transitions, trace_transitions,
                                            model_arcs, trace_arcs):

        model_p_dict = self.generate_location_dict(model_places)
        model_t_dict = self.generate_location_dict(model_transitions)
        trace_p_dict = self.generate_location_dict(trace_places)
        trace_t_dict = self.generate_location_dict(trace_transitions)

        for arc in model_arcs:
            if type(arc.source) == Place:
                p_idx, t_idx = model_p_dict[arc.source.name], model_t_dict[arc.target.name]
                model_places[p_idx].out_arcs.add(arc)
                model_transitions[t_idx].in_arcs.add(arc)

            else:
                t_idx, p_idx = model_t_dict[arc.source.name], model_p_dict[arc.target.name]
                model_transitions[t_idx].out_arcs.add(arc)
                model_places[p_idx].in_arcs.add(arc)

        for arc in trace_arcs:
            if type(arc.source) == Place:
                p_idx, t_idx = trace_p_dict[arc.source.name], trace_t_dict[arc.target.name]
                trace_places[p_idx].out_arcs.add(arc)
                trace_transitions[t_idx].in_arcs.add(arc)

            else:
                t_idx, p_idx = trace_t_dict[arc.source.name], trace_p_dict[arc.target.name]
                trace_transitions[t_idx].out_arcs.add(arc)
                trace_places[p_idx].in_arcs.add(arc)

    def construct_reachability_graph(self):
        curr_mark = self.init_mark
        curr_node = Node(curr_mark.places)
        self.reachability_graph = Graph()
        if self.final_mark is not None:
            self.reachability_graph.ending_node = Node(self.final_mark.places)
        self.reachability_graph.add_node(curr_node)
        self.reachability_graph.starting_node = curr_node
        available_transitions = self.__find_available_transitions(curr_mark.places)
        nodes_to_explore = deque()
        visited_marks = set()

        for transition in available_transitions:
            nodes_to_explore.append((curr_mark, transition, curr_node))

        visited_marks.add(curr_mark.places)

        while nodes_to_explore:
            prev_node_triplet = nodes_to_explore.popleft()
            prev_mark, prev_transition, prev_node = prev_node_triplet[0], prev_node_triplet[1], prev_node_triplet[2]
            assert self.__check_transition_prerequesits(prev_transition, prev_mark.places) == True
            curr_mark = self.__fire_transition(prev_mark, prev_transition)
            curr_node = Node(curr_mark.places)
            prev_node.add_neighbor(curr_node, prev_transition)
            self.reachability_graph.add_edge(
                Edge(prev_transition.name, prev_mark, curr_mark, prev_transition.move_type))

            if curr_mark.places in visited_marks:
                continue

            else:
                for transition in self.__find_available_transitions(curr_mark.places):
                    nodes_to_explore.append((curr_mark, transition, curr_node))

                visited_marks.add(curr_mark.places)
                self.reachability_graph.add_node(curr_node)

    def construct_synchronous_product(self, trace_model):
        '''This func assigns all trace transitions move_type=trace and all model transitions move_type=model
        additionaly, all sync transitions will be assigned move_type=sync '''

        model_places = [Place(p.name) for p in self.places]
        model_transitions = [Transition(t.name, t.label, move_type='model', prob=t.prob,
                                        weight=t.weight) for t in self.transitions]

        trace_places = [Place(p.name) for p in trace_model.places]
        trace_transitions = [Transition(t.name, t.label, move_type='trace', prob=t.prob,
                                        weight=t.weight, location_idx=t.location_idx) for t in trace_model.transitions]

        model_arcs, trace_arcs = self.generate_nonsync_arcs_for_sync_product(model_places, model_transitions,
                                                                             trace_places, trace_transitions,
                                                                             trace_model)

        self.assign_non_sync_arcs_to_transitions(model_places, trace_places, model_transitions, trace_transitions,
                                                 model_arcs, trace_arcs)

        new_sync_transitions, new_sync_arcs = self.__generate_all_sync_transitions(model_transitions, trace_transitions)

        sync_model_all_places = model_places + trace_places
        sync_model_all_transitions = model_transitions + trace_transitions + new_sync_transitions
        sync_model_all_arcs = model_arcs + trace_arcs + new_sync_arcs
        self.__update_sync_product_trans_names(sync_model_all_transitions)

        sync_prod = PetriNet('sync_prod', sync_model_all_places, sync_model_all_transitions, sync_model_all_arcs,
                             trace_len=trace_model.trace_len)

        sync_prod.init_mark = Marking(self.init_mark.places + trace_model.init_mark.places)
        sync_prod.final_mark = Marking(self.final_mark.places + trace_model.final_mark.places)

        sync_prod.trace_transitions = trace_transitions
        #         print(f'number of places={len(sync_prod.places)}, number of transitions={len(sync_prod.transitions)}, number of sync transitions={len(new_sync_transitions)} \n')
        #         print(f'numebr of arcs: {len(sync_model_all_arcs)}, the arcs are: {[(arc.source.name, arc.target.name) for arc in sync_model_all_arcs]}')
        return sync_prod

    def add_transitions_with_arcs(self, transitions):
        if isinstance(transitions, list):
            self.transitions += transitions
            for transition in transitions:
                self.arcs += list(transition.in_arcs.union(transition.out_arcs))

        else:
            self.transitions.append(transitions)
            self.arcs += list(transition.in_arcs.union(transition.out_arcs))

        self.__update_indices_t_dict(transitions)

    def add_arc_from_to(self, source, target, weight=None):
        if weight is None:
            arc = Arc(source, target)
        else:
            arc = Arc(source, target, weight)
        source.out_arcs.add(arc)
        target.in_arcs.add(arc)
        self.arcs.append(arc)

    #     def __generate_all_sync_transitions(self, trace_model):
    #         sync_transitions = []
    #         counter = 1

    #         for trans in self.transitions:
    #             # trans.label is guaranteed to be unique in the discovered model (from docs)
    #             if trans.label is not None:
    #                 # Find in the trace model all the transitions with the same label
    #                 same_label_transitions = self.__find_simillar_label_transitions(trace_model, trans.label)

    #                 for trace_trans in same_label_transitions:
    #                     new_sync_trans = self.__generate_new_trans(trans, trace_trans, counter)
    #                     sync_transitions.append(new_sync_trans)
    #                     counter += 1

    #         return sync_transitions

    def __generate_all_sync_transitions(self, model_transitions, trace_transitions):
        sync_transitions = []
        new_sync_arcs = []

        for model_tran in model_transitions:
            # trans.label is guaranteed to be unique in the discovered model (from docs)
            if model_tran.label is not None:
                # Find in the trace model all the transitions with the same label
                same_label_transitions = self.__find_simillar_label_transitions(trace_transitions, model_tran.label)

                for trace_tran in same_label_transitions:
                    new_sync_trans, new_arcs = self.__generate_new_trans(model_tran, trace_tran)
                    sync_transitions.append(new_sync_trans)
                    new_sync_arcs += new_arcs

        return sync_transitions, new_sync_arcs

    def __find_simillar_label_transitions(self, trace_transitions, activity_label):
        '''Returns all the transitions in the trace with a specified activity label'''
        same_label_trans = [transition for transition in trace_transitions if transition.label == activity_label]

        return same_label_trans

    def __generate_new_trans(self, model_tran, trace_tran):
        #         name = 'sync_transition_' + str(counter)
        name = f'sync_{trace_tran.name}'
        new_sync_transition = Transition(name=name, label=trace_tran.label, move_type='sync', prob=trace_tran.prob)
        new_sync_transition.location_idx = trace_tran.location_idx

        input_arcs = model_tran.in_arcs.union(trace_tran.in_arcs)
        new_input_arcs = []
        for arc in input_arcs:
            new_arc = Arc(arc.source, new_sync_transition, arc.weight)
            new_input_arcs.append(new_arc)

        output_arcs = model_tran.out_arcs.union(trace_tran.out_arcs)
        new_output_arcs = []
        for arc in output_arcs:
            new_arc = Arc(new_sync_transition, arc.target, arc.weight)
            new_output_arcs.append(new_arc)

        new_sync_transition.in_arcs = new_sync_transition.in_arcs.union(new_input_arcs)
        new_sync_transition.out_arcs = new_sync_transition.out_arcs.union(new_output_arcs)

        return new_sync_transition, new_input_arcs + new_output_arcs

    def __update_indices_p_dict(self, places):
        curr_idx = len(self.places_indices)
        if isinstance(places, list):
            for p in places:
                self.places_indices[p.name] = curr_idx
                curr_idx += 1
        else:
            self.places_indices[places.name] = curr_idx

    def __update_indices_t_dict(self, transitions):
        curr_idx = len(self.transitions_indices)
        if isinstance(transitions, list):
            for t in transitions:
                self.transitions_indices[t.name] = curr_idx
                curr_idx += 1
        else:
            self.transitions_indices[transitions.name] = curr_idx

    def __find_available_transitions(self, mark_tuple):
        '''Input: tuple
           Output: list'''

        available_transitions = []
        for transition in self.transitions:
            if self.__check_transition_prerequesits(transition, mark_tuple):
                available_transitions.append(transition)

        return available_transitions

    def __check_transition_prerequesits(self, transition, mark_tuple):
        for arc in transition.in_arcs:
            arc_weight = arc.weight
            source_idx = self.places_indices[arc.source.name]
            if mark_tuple[source_idx] < arc_weight:
                return False

        return True

    def __assign_trace_transitions_move_type(self, trace_model):
        trace_model_copy = copy.deepcopy(trace_model)
        for trans in trace_model_copy.transitions:
            if trans.move_type is None:
                trans.move_type = 'trace'

        return trace_model_copy

    def __assign_model_transitions_move_type(self):
        for trans in self.transitions:
            if trans.move_type is None:
                trans.move_type = 'model'

    #     def conformance_checking(self):
    #         if self.reachability_graph is None:
    #             self.construct_reachability_graph()

    #         alignment, min_cost_distance = self.__dijkstra()
    #         return alignment, min_cost_distance

    #     def conformance_checking(self, trace_model):
    #         sync_prod = self.construct_synchronous_product(trace_model)
    # #         sync_prod.construct_reachability_graph()

    #         return sync_prod.astar_extended()

    def conformance_checking(self, trace_model, hist_prob_dict=None, lamda=0.5):
        sync_prod = self.construct_synchronous_product(trace_model)
        return sync_prod._dijkstra_no_rg_construct(hist_prob_dict, lamda=lamda)

    def __dijkstra(self):
        distance_min_heap = []
        heapify(distance_min_heap)
        visited_nodes = set()
        search_graph_nodes = [search_node(node) for node in self.reachability_graph.nodes]
        nodes_idx_dict = {search_node.graph_node.marking: idx for idx, search_node in enumerate(search_graph_nodes)}

        source_node_idx = nodes_idx_dict[self.reachability_graph.starting_node.marking]
        source_node = search_graph_nodes[source_node_idx]
        source_node.dist = 0

        for node in search_graph_nodes:
            heappush(distance_min_heap, node)

        while distance_min_heap:
            min_dist_node = heappop(distance_min_heap)
            need_heapify = False

            for neighbor_transition_tuple in min_dist_node.graph_node.neighbors:
                neighbor, transition = neighbor_transition_tuple[0], neighbor_transition_tuple[1]
                alt_distance = min_dist_node.dist + transition.weight
                neighbor_search_idx = nodes_idx_dict[neighbor.marking]

                if alt_distance < search_graph_nodes[neighbor_search_idx].dist:
                    search_graph_nodes[neighbor_search_idx].dist = alt_distance
                    search_graph_nodes[neighbor_search_idx].ancestor = min_dist_node
                    search_graph_nodes[neighbor_search_idx].transition_to_ancestor = transition
                    need_heapify = True

            if need_heapify:
                heapify(distance_min_heap)

        #         print('ending marking is: ', self.reachability_graph.ending_node.marking)
        #         print('nodes_idx_dict is: ', nodes_idx_dict)
        final_mark_idx = nodes_idx_dict[self.reachability_graph.ending_node.marking]
        curr_node = search_graph_nodes[final_mark_idx]
        path = []

        while curr_node is not source_node:
            #             path.append(curr_node.transition_to_ancestor.label)
            path.append(curr_node.transition_to_ancestor.name)
            curr_node = curr_node.ancestor

        return path[::-1], search_graph_nodes[final_mark_idx].dist

    def _dijkstra_no_rg_construct(self, prob_dict=None, lamda=0.5, return_final_marking=False):
        distance_min_heap = []
        heapify(distance_min_heap)
        curr_node = search_node_new(self.init_mark, dist=0)
        heappush(distance_min_heap, curr_node)
        marking_distance_dict = {}
        visited_markings = set()

        if prob_dict is None:
            prob_dict = {}

        while distance_min_heap:
            min_dist_node = heappop(distance_min_heap)

            if min_dist_node.marking.places in visited_markings:
                continue

            if min_dist_node.marking.places == self.final_mark.places:
                break

            available_transitions = self._find_available_transitions(min_dist_node.marking.places)
            for transition in available_transitions:
                new_marking = self._fire_transition(min_dist_node.marking, transition)

                if new_marking.places in visited_markings:
                    continue

                if new_marking in visited_markings:
                    continue

                conditioned_transition_weight = self.compute_conditioned_weight(min_dist_node.path_prefix, transition,
                                                                                prob_dict, lamda=lamda)
                if new_marking.places not in marking_distance_dict or marking_distance_dict[
                    new_marking.places] > min_dist_node.dist + conditioned_transition_weight:
                    new_path_prefix = min_dist_node.path_prefix + transition.label if transition.label is not None else min_dist_node.path_prefix

                    new_node = search_node_new(new_marking,
                                               dist=min_dist_node.dist + conditioned_transition_weight,
                                               ancestor=min_dist_node,
                                               transition_to_ancestor=transition,
                                               path_prefix=new_path_prefix)

                    marking_distance_dict[new_marking.places] = new_node.dist
                    heappush(distance_min_heap, new_node)

            visited_markings.add(min_dist_node.marking.places)

        shortest_path = []
        curr_node = min_dist_node
        while curr_node.ancestor:
            shortest_path.append(curr_node.transition_to_ancestor.name)
            curr_node = curr_node.ancestor

        #         print('min_dist_node distance: ', min_dist_node.dist)
        #         print('shortest path: ', shortest_path[::-1])
        if return_final_marking:  # TO DO: need to include overlap in the code
            return shortest_path[::-1], min_dist_node.dist, self.marking.place

        return shortest_path[::-1], min_dist_node.dist

    def __fire_transition(self, mark, transition):
        '''Input: Marking object, Transition object
        Output: Marking object'''

        subtract_mark = [0] * len(mark.places)
        for arc in transition.in_arcs:
            place_idx = self.places_indices[arc.source.name]
            subtract_mark[place_idx] -= arc.weight

        add_mark = [0] * len(mark.places)
        for arc in transition.out_arcs:
            place_idx = self.places_indices[arc.target.name]
            add_mark[place_idx] += arc.weight

        new_mark = tuple([sum(x) for x in zip(list(mark.places), subtract_mark, add_mark)])
        for elem in new_mark:
            if elem < 0:
                print(
                    f'the mark was: {mark} and I subtract the following values: {subtract_mark} and adding these: {add_mark} \
                which results in this: {new_mark} and all this sh!t was by using this transition: {transition.name}')
        new_mark_obj = Marking(new_mark)

        return new_mark_obj

    def __update_sync_product_trans_names(self, transitions_list):

        for trans in transitions_list:
            if trans.move_type == 'model':
                trans.name = f'({trans.name}, >>)'
            elif trans.move_type == 'trace':
                trans.name = f'(>>, {trans.name})'
            else:
                trans.name = f'({trans.name}, {trans.name})'


def generate_probs(n_elements):
    probs = []
    for i in range(n_elements):
        res = random.sample(range(1, 10), 3)
        total = sum(res)
        norm_res = [round(item / total, 2) for item in res]
        probs.append(norm_res)
    return probs


def smooth_zero_probs(probabilities_lst):
    zero_indices = []
    non_zero_donating_indices = []
    for index, value in enumerate(probabilities_lst):
        if value == 0:
            zero_indices.append(index)

        elif value > 0.1:
            non_zero_donating_indices.append(index)

    assert non_zero_donating_indices == []

    for zero_index in zero_indices:
        donating_index = random.choice(non_zero_donating_indices)
        assert probabilities_lst[donating_index] <= 0.01
        probabilities_lst[zero_index] += 0.01
        probabilities_lst[donating_index] -= 0.01

    return probabilities_lst


def generate_argmax_probabilities(argmax_prob_is_right, n_activities):
    '''
    Given probability for argmax being the true activity the function generates probability list
    '''

    new_probs_list = []
    for i in range(n_activities):
        new_probs_list.append(random.uniform(1.0, 10.0))

    max_value = max(new_probs_list)
    max_value_idx = new_probs_list.index(max_value)
    random_uniform_threshold = random.uniform(0, 1)

    if argmax_prob_is_right > random_uniform_threshold:
        if max_value_idx != 0:
            new_probs_list[0], new_probs_list[max_value_idx] = new_probs_list[max_value_idx], new_probs_list[0]

    else:
        if max_value_idx == 0:
            random_idx_in_lst = random.choice([i for i in range(1, n_activities)])
            counter = 0
            while new_probs_list[random_idx_in_lst] == max_value:
                random_idx_in_lst = random.choice([i for i in range(1, n_activities)])
                counter += 1
                if counter > 10:
                    break

            new_probs_list[0], new_probs_list[random_idx_in_lst] = new_probs_list[random_idx_in_lst], new_probs_list[0]

    normalize_factor = sum(new_probs_list)
    new_probs_list_normalized = [round(value / normalize_factor, 2) for value in new_probs_list]

    return new_probs_list_normalized


def generate_argmax_probabilities_unique(argmax_prob_is_right, n_activities):
    while True:
        new_probs = generate_argmax_probabilities(argmax_prob_is_right, n_activities)
        if len(new_probs) == len(set(new_probs)):
            break

    return new_probs


def generate_probabilities(true_value_prob, n_activities):
    new_probs_list = [true_value_prob]
    noisy_probs = []
    for i in range(n_activities - 1):
        noisy_probs.append(random.uniform(1.0, 10.0))

    normalize_factor = sum(noisy_probs) / (1 - true_value_prob)

    noisy_probs_normalized = [item / normalize_factor for item in noisy_probs]

    new_probs_list += noisy_probs_normalized
    new_probs_list = [round(item, 2) for item in new_probs_list]

    return new_probs_list


def convert_to_integer(num):
    if isinstance(num, (np.ndarray, np.generic)):
        return num.item()

    return int(num)


def generate_new_record(trace_df, idx):
    new_dict = {}
    new_dict['concept:name'] = [[activity for activity in trace_df.iloc[idx]['concept:name']]]
    new_dict['case:concept:name'] = trace_df.iloc[idx]['case:concept:name']
    new_dict['probs'] = [[prob for prob in trace_df.iloc[idx]['probs']]]
    new_record = pd.DataFrame(new_dict)

    return new_record


def copy_trace(trace, for_determ=True):
    new_trace_df = pd.DataFrame()

    for i in range(len(trace)):
        new_trace_df = pd.concat([new_trace_df, generate_new_record(trace, i)])

    new_trace_df = new_trace_df.reset_index(drop=True)

    if for_determ:
        new_trace_df['concept:name'] = new_trace_df['concept:name'].apply(lambda x: x[0])

    return new_trace_df


def copy_dataframe(df):
    traces_df_list = []
    traces_ids = df['case:concept:name'].unique()

    for trace_id in traces_ids:
        trace_df = df[df['case:concept:name'] == trace_id]
        trace_df_copy = copy_trace(trace_df, for_determ=False)
        traces_df_list.append(trace_df_copy)

    new_df_copy = pd.concat(traces_df_list, ignore_index=True)

    return new_df_copy


def alter_activity_labels(trace_df, activities_universe, alter_fraq):
    n_activities_to_alter = int(round(len(trace_df) * alter_fraq))

    if n_activities_to_alter == 0:
        return trace_df

    rows_expansion_indexes = random.sample(range(len(trace_df)), n_activities_to_alter)

    for idx in rows_expansion_indexes:
        curr_activity = {trace_df.at[idx, 'concept:name'][0]}
        new_activity = random.choice(list(activities_universe.copy().difference(curr_activity)))
        trace_df.at[idx, 'concept:name'] = [new_activity]

    return trace_df


def swap_successor_events(trace_df, swap_fraq=0.5):
    indices = [i for i in range(len(trace_df))]
    indices_to_swap = [idx for idx in indices if random.uniform(0, 1) < swap_fraq]
    left_right_swap = [int(round(random.uniform(0, 1))) for _ in indices_to_swap]

    if len(indices_to_swap) == 0:
        return trace_df

    if indices_to_swap[0] == 0:
        left_right_swap[0] = 1

    if indices_to_swap[-1] == len(trace_df) - 1:
        left_right_swap[-1] = 0

    indices_to_swap = set(indices_to_swap)

    left_right_idx = 0
    for i, index in enumerate(indices):
        if index in indices_to_swap:
            if left_right_swap[left_right_idx] == 0:
                indices[i], indices[i - 1] = indices[i - 1], indices[i]

            elif i + 1 < len(trace_df):
                indices[i], indices[i + 1] = indices[i + 1], indices[i]

            indices_to_swap.remove(index)
            left_right_idx += 1

    trace_df = trace_df.reindex(indices).reset_index(drop=True)

    return trace_df


def duplicate_activities(trace_df, duplicate_fraq=0.5):
    #     indices_to_duplicate = [i for i in range(len(trace_df)) if random.uniform(0,1) < duplicate_fraq]

    n_indices_to_duplicate = int(round(len(trace_df) * duplicate_fraq))
    indices_to_duplicate = [i for i in range(len(trace_df))]
    indices_to_duplicate = sorted(random.sample(indices_to_duplicate, n_indices_to_duplicate))

    #     indices_to_duplicate_doesnt_work = list(random.sample([i for i in range(len(trace_df))], n_indices_to_duplicate))
    #     print(f'good indices:{indices_to_duplicate} and bad indices:{indices_to_duplicate_doesnt_work}')
    #     if n_indices_to_duplicate == 0:
    #         return trace_df

    #     if len(indices_to_duplicate) == 0:
    #         return trace_df

    #     indices_to_duplicate = random.sample(range(len(trace_df)), n_indices_to_duplicate)

    curr_idx = 0
    new_duplicated_df = pd.DataFrame(columns=trace_df.columns)

    if not indices_to_duplicate:
        return trace_df

    for idx in indices_to_duplicate:
        new_duplicated_df = pd.concat(
            [new_duplicated_df, trace_df.iloc[curr_idx:idx], generate_new_record(trace_df, idx)])
        curr_idx = idx
        new_duplicated_df = new_duplicated_df.reset_index(drop=True)

    new_duplicated_df = pd.concat([new_duplicated_df, trace_df.iloc[idx:].copy()])
    new_duplicated_df = new_duplicated_df.reset_index(drop=True)

    return new_duplicated_df


def add_noise_by_trace(df, disturbed_trans_frac=0.3, true_trans_prob=0.5, expansion_min=2, expansion_max=4, \
                       randomized_true_trans_prob=False, generate_probs_for_argmax=False, gradual_noise=True,
                       alter_labels=False, swap_events=False, duplicate_acts=False, fraq=0.1, determ_noise_only=False):
    all_activities_unique = set([activity[0] for activity in df['concept:name'].tolist()])
    unique_cases_ids = df['case:concept:name'].unique()

    new_noised_deterministic_df = pd.DataFrame(columns=df.columns)
    first_case_df = True
    for case_id in unique_cases_ids:
        case_df = df[df['case:concept:name'] == case_id]
        case_df = case_df.reset_index(drop=True)

        if alter_labels:
            case_df = alter_activity_labels(case_df, all_activities_unique, fraq)

        if swap_events:
            case_df = swap_successor_events(case_df, fraq)

        if duplicate_acts:
            case_df = duplicate_activities(case_df, fraq)

        #         new_noised_deterministic_df = pd.concat([new_noised_deterministic_df, copy_trace(case_df)])
        if determ_noise_only is False:

            if gradual_noise:
                case_df = add_gradual_noise(case_df, disturbed_trans_frac=disturbed_trans_frac,
                                            true_trans_prob=true_trans_prob,
                                            expansion_min=expansion_min, expansion_max=expansion_max,
                                            randomized_true_trans_prob=randomized_true_trans_prob,
                                            generate_probs_for_argmax=generate_probs_for_argmax,
                                            all_activities_unique=all_activities_unique)

            else:
                case_df = add_noise(case_df, disturbed_trans_frac=disturbed_trans_frac, true_trans_prob=true_trans_prob,
                                    expansion_min=expansion_min, expansion_max=expansion_max,
                                    randomized_true_trans_prob=randomized_true_trans_prob,
                                    generate_probs_for_argmax=generate_probs_for_argmax,
                                    all_activities_unique=all_activities_unique)

        if not first_case_df:
            df_new = df_new.append(case_df, ignore_index=True)

        else:
            df_new = case_df
            first_case_df = False

    #     new_noised_deterministic_df = new_noised_deterministic_df.reset_index(drop=True)

    return df_new  # , new_noised_deterministic_df


def add_noise(df, disturbed_trans_frac=0.3, true_trans_prob=0.5, expansion_min=2, expansion_max=4, \
              randomized_true_trans_prob=False, generate_probs_for_argmax=False, all_activities_unique=None):
    # print(f'disturbed_trans_frac is: {disturbed_trans_frac} and df len is: {len(df)} and their multiply is: {disturbed_trans_frac * len(df)} and rounding this number results in:{round(disturbed_trans_frac * len(df))}')
    n_rows = int(round(disturbed_trans_frac * len(df)))
    # print(f'df len is: {len(df)} and n_rows for parallel activities are: {n_rows}')
    rows_expansion_indexes = random.sample(range(len(df)), n_rows)
    all_activities_unique = set([activity[0] for activity in df[
        'concept:name'].tolist()]) if all_activities_unique is None else all_activities_unique

    for idx in rows_expansion_indexes:
        if randomized_true_trans_prob:
            true_trans_prob = np.random.uniform(0.01, 0.99)
        curr_activity = {df.at[idx, 'concept:name'][0]}

        n_expanded_trans = random.randint(expansion_min, expansion_max)
        assert n_expanded_trans < len(all_activities_unique) - 1
        noisy_activities = random.sample(all_activities_unique.copy().difference(curr_activity), n_expanded_trans - 1)

        noisy_activities = [activity for activity in noisy_activities]

        for activity in noisy_activities:
            df.at[idx, 'concept:name'].append(activity)

        if generate_probs_for_argmax:
            df.at[idx, 'probs'] = generate_argmax_probabilities_unique(true_trans_prob, n_expanded_trans)

        else:
            df.at[idx, 'probs'] = generate_probabilities(true_trans_prob, n_expanded_trans)

    return df


def add_gradual_noise(df, disturbed_trans_frac=0.3, true_trans_prob=0.5, expansion_min=2, expansion_max=4, \
                      randomized_true_trans_prob=False, generate_probs_for_argmax=False, all_activities_unique=None):
    deterministic_idxs = []
    total_disturbed_transitions = 0
    for idx, transition in enumerate(df['concept:name']):
        if len(transition) > 1:
            total_disturbed_transitions += 1
        else:
            deterministic_idxs.append(idx)

    current_disturbed_transitions_fraq = total_disturbed_transitions / len(df)

    if current_disturbed_transitions_fraq >= disturbed_trans_frac:
        return df

    n_additional_trans_to_disturb = int(round((disturbed_trans_frac - current_disturbed_transitions_fraq) * len(df)))

    if n_additional_trans_to_disturb == 0:
        return df

    rows_expansion_indexes = random.sample(deterministic_idxs, n_additional_trans_to_disturb)

    all_activities_unique = set([activity for activity_lst in df['concept:name'].tolist() for activity in
                                 activity_lst]) if all_activities_unique is None else all_activities_unique

    for idx in rows_expansion_indexes:
        if randomized_true_trans_prob:
            true_trans_prob = np.random.uniform(0.01, 0.99)
        curr_activity = {df.at[idx, 'concept:name'][0]}

        n_expanded_trans = random.randint(expansion_min, expansion_max)

        assert n_expanded_trans < len(all_activities_unique) - 1
        noisy_activities = random.sample(list(all_activities_unique.copy().difference(curr_activity)),
                                         n_expanded_trans - 1)

        noisy_activities = [activity for activity in noisy_activities]

        for activity in noisy_activities:
            df.at[idx, 'concept:name'].append(activity)

        if generate_probs_for_argmax:
            df.at[idx, 'probs'] = generate_argmax_probabilities_unique(true_trans_prob, n_expanded_trans)

        else:
            df.at[idx, 'probs'] = generate_probabilities(true_trans_prob, n_expanded_trans)

    return df


def make_all_probs_one(preprocessed_test_data):
    df_one_probs = preprocessed_test_data.copy(deep=True)
    df_one_probs['probs'] = df_one_probs['probs'].apply(lambda x: [1 for prob in x])

    return df_one_probs


def construct_trace_model(trace_df, non_sync_move_penalty=1):
    places = [Place(f'place_{i}') for i in range(len(trace_df) + 1)]
    transitions = []
    transition_to_idx_dict = {}
    curr_idx = 0

    for i in range(len(trace_df)):
        for idx, activity in enumerate(trace_df.iloc[i, 0]):
            new_transition = Transition(f'{activity}_{i + 1}', activity, prob=trace_df.iloc[i, 1][idx],
                                        weight=non_sync_move_penalty)
            transitions.append(new_transition)
            transition_to_idx_dict[f'{activity}_{i + 1}'] = curr_idx
            new_transition.location_idx = i
            curr_idx += 1

    trace_model_net = PetriNet('trace_model', places, transitions, trace_len=len(trace_df))

    for i in range(len(trace_df)):
        for activity in trace_df.iloc[i, 0]:
            trace_model_net.add_arc_from_to(places[i], transitions[transition_to_idx_dict[f'{activity}_{i + 1}']])
            trace_model_net.add_arc_from_to(transitions[transition_to_idx_dict[f'{activity}_{i + 1}']], places[i + 1])

    init_mark = tuple([1] + [0] * len(trace_df))
    final_mark = tuple([0] * len(trace_df) + [1])

    trace_model_net.init_mark = Marking(init_mark)
    trace_model_net.final_mark = Marking(final_mark)

    return trace_model_net


def pre_process_log(df_log):
    df_log = copy.deepcopy(df_log)
    clean_df_log = df_log[['concept:name', 'case:concept:name']]
    clean_df_log['probs'] = [[1.0]] * len(clean_df_log)
    clean_df_log['probs'].astype('object')
    clean_df_log['concept:name'] = clean_df_log['concept:name'].apply(lambda activity: [activity])

    return clean_df_log


def calc_conf_for_log(df_log, model, non_sync_move_penalty=1, add_heuristic=False):
    unique_cases_ids = df_log['case:concept:name'].unique()
    conformance_scores_lst = []

    for case_id in unique_cases_ids:
        case_df = df_log[df_log['case:concept:name'] == case_id]
        case_df = case_df[['concept:name', 'probs']]
        case_df = case_df.reset_index(drop=True)
        case_trace_model = construct_trace_model(case_df, non_sync_move_penalty, add_heuristic=add_heuristic)
        case_conformance_score = model.conformance_checking(case_trace_model)[1]
        conformance_scores_lst.append(case_conformance_score)

    return mean(conformance_scores_lst)


def sort_places(places):
    init_mark = [place for place in places if place.name == 'source']
    final_mark = [place for place in places if place.name == 'sink']
    inner_places = [place for place in places if place.name not in {'source', 'sink'}]
    inner_places_sorted = sorted(inner_places, key=lambda x: float(x.name[2:]))
    places_sorted = init_mark + inner_places_sorted + final_mark

    return places_sorted


def from_discovered_model_to_PetriNet(discovered_model, non_sync_move_penalty=1, name='discovered_net'):
    discovered_model = copy.deepcopy(discovered_model)
    places = sort_places(discovered_model.places)
    places = [Place(p.name) for p in places]

    petri_new_arcs = []
    transition_list = list(discovered_model.transitions)

    assert len([tran.name for tran in transition_list]) == len(set([tran.name for tran in transition_list]))
    assert len([place.name for place in places]) == len(set([place.name for place in places]))

    tran2idx = {tran.name: i for i, tran in enumerate(transition_list)}
    place2idx = {place.name: i for i, place in enumerate(places)}

    transitions = [Transition(transition.name, transition.label, transition.in_arcs, transition.out_arcs, 'model',
                              weight=non_sync_move_penalty) \
                   for transition in transition_list]
    for trans in transitions:
        if trans.label is None:
            trans.weight = 0

    for i in range(len(transitions)):
        new_in_arcs = set()
        for arc in transitions[i].in_arcs:
            new_in_arc = Arc(places[place2idx[arc.source.name]], transitions[i])
            new_in_arcs.add(new_in_arc)
            petri_new_arcs.append(new_in_arc)

        new_out_arcs = set()
        for arc in transitions[i].out_arcs:
            new_out_arc = Arc(transitions[i], places[place2idx[arc.target.name]])
            new_out_arcs.add(new_out_arc)
            petri_new_arcs.append(new_out_arc)

        transitions[i].in_arcs = new_in_arcs
        transitions[i].out_arcs = new_out_arcs

    for transition in transitions:
        if transition.label is not None:
            transition.name = transition.label

    init_mark = tuple([1] + [0] * (len(places) - 1))
    final_mark = tuple([0] * (len(places) - 1) + [1])

    new_PetriNet = PetriNet(name)
    new_PetriNet.add_places(places)
    new_PetriNet.add_transitions(transitions)
    new_PetriNet.init_mark = Marking(init_mark)
    new_PetriNet.final_mark = Marking(final_mark)
    new_PetriNet.arcs = petri_new_arcs

    return new_PetriNet


def argmax_stochastic_trace(stochastic_trace_df):
    # Initialize an empty list to hold dictionaries
    data_list = []

    for i in range(len(stochastic_trace_df)):
        # Assuming stochastic_trace_df.iloc[i, 2] is a list or similar iterable with probabilities
        max_val = max(stochastic_trace_df.iloc[i, 2])
        max_idx = stochastic_trace_df.iloc[i, 2].index(max_val)
        highest_prob_activity = stochastic_trace_df.iloc[i, 0][max_idx]
        case_id = stochastic_trace_df.iloc[i, 1]

        # Append a new dictionary for each row to be added to the DataFrame
        data_list.append({'concept:name': highest_prob_activity, 'case:concept:name': case_id, 'probs': [1.0]})

    # Create a DataFrame from the list of dictionaries
    determ_df = pd.DataFrame(data_list)

    return determ_df


def get_non_sync_non_quiet_activities(alignment, quiet_activities):
    non_sync_non_quiet_transitions = []
    for trans in alignment:
        if '>>' in trans:
            if not any(activity for activity in quiet_activities if activity in trans):
                non_sync_non_quiet_transitions.append(trans)

    return non_sync_non_quiet_transitions


def get_sync_activities(alignment):
    return [align for align in alignment if '>>' not in align]


def alignment_accuracy_helper(alignment, true_trace_df):
    stochastic_align_clean = [item.split(',')[1][1:-1] for item in alignment[0] if item.split(',')[1][1:-1] != '>>']
    determ_trace_clean = true_trace_df['concept:name'].tolist()
    #     print('true trace df is: ', true_trace_df)
    #     print('determ trace clean is: ', determ_trace_clean)
    similar_activity = 0
    for idx, item in enumerate(determ_trace_clean):
        if item in stochastic_align_clean[idx]:
            similar_activity += 1

    return similar_activity / len(determ_trace_clean), len(determ_trace_clean)


def compare_argmax_and_stochastic_alignments(stochastic_trace_df, true_trace_df, model, non_sync_penalty=1,
                                             add_heuristic=False):
    df_stochastic = stochastic_trace_df[['concept:name', 'probs']]
    df_stochastic = df_stochastic.reset_index(drop=True)
    case_trace_stochastic_model = construct_trace_model(df_stochastic, non_sync_penalty)

    #     case_trace_stochastic_model = construct_trace_model(df_stochastic, non_sync_penalty, add_heuristic=add_heuristic)
    stochastic_alignment = model.conformance_checking(case_trace_stochastic_model)
    #     print('True trace is:')
    #     display(true_trace_df)
    #     print()
    #     print('The modified trace is:')
    #     display(df_stochastic)
    #     print()
    #     print(f'The alignment is: {stochastic_alignment[0]}')
    #     print(f'The cost of the alignment is: {stochastic_alignment[1]}')
    #     print()
    argmax_trace = argmax_stochastic_trace(stochastic_trace_df)
    #     argmax_trace_preprocessed = pre_process_log(argmax_trace)
    #     df_deterministic = argmax_trace_preprocessed[['concept:name', 'probs']]
    #     df_deterministic = df_deterministic.reset_index(drop = True)
    #     case_trace_deterministic_model = construct_trace_model(df_deterministic, non_sync_penalty, add_heuristic=add_heuristic)
    #     argmax_alignment = model.conformance_checking(case_trace_deterministic_model)

    #     quiet_activities = {transition.name for transition in model.transitions if transition.label is None}
    #     non_sync_non_quiet_stochastic_alignment_activities = get_non_sync_non_quiet_activities(stochastic_alignment[0], quiet_activities)
    #     sync_stochastic_alignment_activities = get_sync_activities(stochastic_alignment[0])
    #     non_sync_non_quiet_argmax_alignment_activities = get_non_sync_non_quiet_activities(argmax_alignment[0], quiet_activities)
    #     sync_argmax_alignment_activities = get_sync_activities(argmax_alignment[0])
    stochastic_acc, trace_len = alignment_accuracy_helper(stochastic_alignment, true_trace_df)
    argmax_acc = sum(argmax_trace['concept:name'].reset_index(drop=True) == true_trace_df['concept:name'].reset_index(
        drop=True)) / len(argmax_trace)

    #     return non_sync_non_quiet_stochastic_alignment_activities, sync_stochastic_alignment_activities, \
    #            non_sync_non_quiet_argmax_alignment_activities, sync_argmax_alignment_activities, \
    #            stochastic_acc, argmax_acc, trace_len, stochastic_alignment[1]

    return stochastic_acc, argmax_acc


def generate_statistics_for_dataset(stochastic_dataset, df_test, model, shortest_path_in_model, non_sync_penalty=1,
                                    add_heuristic=False):
    '''Generate 4 arrays for plots.
       Input: stochastic dataframe with multiple traces
       Output: 4 arrays with amounts of sync and non_syc activities'''

    unique_cases_ids = stochastic_dataset['case:concept:name'].unique()

    #     non_sync_non_quiet_stochastic_alignment_len = []
    #     sync_stochastic_alignment_len = []
    #     non_sync_non_quiet_argmax_len = []
    #     sync_argmax_alignment_len = []
    #     stochastic_acc_lst = []
    #     argmax_acc_lst = []
    #     traces_length = []
    stochastic_conformance_scores_lst = []
    #     fitness_lst = []

    for case_id in unique_cases_ids:
        case_df = stochastic_dataset[stochastic_dataset['case:concept:name'] == case_id]
        case_df = case_df.reset_index(drop=True)
        true_case_df = df_test[df_test['case:concept:name'] == case_id]
        #         worst_alignment_score = shortest_path_in_model + len(case_df)

        #         non_sync_non_quiet_stochastic_alignment_activities, sync_stochastic_alignment_activities, \
        #         non_sync_non_quiet_argmax_alignment_activities, sync_argmax_alignment_activities, stochastic_acc, \
        #         argmax_acc, trace_len, stoc_conf_score = compare_argmax_and_stochastic_alignments(case_df, true_case_df, model, non_sync_penalty, add_heuristic=add_heuristic)
        #         fitness = 1 - stoc_conf_score / worst_alignment_score
        stoc_conf_score = compare_argmax_and_stochastic_alignments(case_df, true_case_df, model, non_sync_penalty,
                                                                   add_heuristic=add_heuristic)
        #         non_sync_non_quiet_stochastic_alignment_len.append(len(non_sync_non_quiet_stochastic_alignment_activities))
        #         sync_stochastic_alignment_len.append(len(sync_stochastic_alignment_activities))
        #         non_sync_non_quiet_argmax_len.append(len(non_sync_non_quiet_argmax_alignment_activities))
        #         sync_argmax_alignment_len.append(len(sync_argmax_alignment_activities))
        #         stochastic_acc_lst.append(stochastic_acc)
        #         argmax_acc_lst.append(argmax_acc)
        #         traces_length.append(trace_len)
        stochastic_conformance_scores_lst.append(stoc_conf_score)
    #         fitness_lst.append(fitness)

    #     return non_sync_non_quiet_stochastic_alignment_len, sync_stochastic_alignment_len, \
    #            non_sync_non_quiet_argmax_len, sync_argmax_alignment_len, stochastic_acc_lst, argmax_acc_lst, traces_length, \
    #            stochastic_conformance_scores_lst, fitness_lst

    return stochastic_conformance_scores_lst


def calculate_statistics_for_different_uncertainty_levels(df, non_sync_penalty=1, n_traces_for_model_building=15,
                                                          true_trans_prob=None, expansion_min=2, expansion_max=2,
                                                          uncertainty_levels=None, generate_probs_for_argmax=False,
                                                          by_trace=True, add_heuristic=False,
                                                          custom_traces_addition=False,
                                                          deterministic_evaluation=False,
                                                          return_shortest_path_only=False,
                                                          all_probs_one=False, gradual_noise=True,
                                                          eval_lower_bound=True,
                                                          alter_labels=False, swap_events=False, duplicate_acts=False,
                                                          determ_noise_fraq=0.3, determ_noised_dataset=None):
    if uncertainty_levels is None:
        uncertainty_levels = np.linspace(0, 1, 21)

    train_traces = list(df['case:concept:name'].unique())[:n_traces_for_model_building]
    if custom_traces_addition:
        additional_traces = [173823, 173793, 173754, 173739, 173736, 173778, 173763, 173847]
        additional_traces = [np.int64(item) for item in additional_traces]
        train_traces += additional_traces
        train_traces = list(set(train_traces))

    train_data = df[df['case:concept:name'].isin(train_traces)]
    test_data = df[~df['case:concept:name'].isin(train_traces)]

    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(train_data)
    discovered_net = from_discovered_model_to_PetriNet(net, non_sync_move_penalty=non_sync_penalty)
    shortest_path_in_model = discovered_net.astar()[1]

    if return_shortest_path_only:
        train_traces_length = [len(train_data[train_data['case:concept:name'] == trace_id]) for trace_id in
                               train_traces]
        mean_trace_length = mean(train_traces_length)
        return shortest_path_in_model, mean_trace_length

    if deterministic_evaluation:
        test_data_copy = test_data.copy()
        test_data_copy['probs'] = [[1.0]] * len(test_data_copy)
        test_data_copy['concept:name'] = test_data_copy['concept:name'].apply(lambda x: [x])
        return calc_conf_for_log(test_data_copy, discovered_net, non_sync_move_penalty=non_sync_penalty,
                                 add_heuristic=add_heuristic)

    if true_trans_prob is None:
        true_trans_prob = np.linspace(0, 1, 21)

    if not isinstance(true_trans_prob, (np.ndarray, list)):
        true_trans_prob = [true_trans_prob]

    #     non_sync_stochastic_avgs = []
    #     sync_stochastic_avgs = []
    #     non_sync_argmax_avgs = []
    #     sync_argmax_avgs = []
    #     stochastic_acc_avgs = []
    #     argmax_acc_avgs = []
    stochastic_conformance_avgs = []
    #     fitness_avgs = []
    lower_bound_scores = []

    original_alter_labels, original_swap_events, original_duplicate_acts = alter_labels, swap_events, duplicate_acts

    for true_prob in true_trans_prob:
        true_prob = round(true_prob, 2)

        preprocessed_test_data = None
        determ_preprocess_flag = 1

        for uncert_level in uncertainty_levels:
            uncert_level = round(uncert_level, 2)

            if determ_preprocess_flag == 1:
                alter_labels, swap_events, duplicate_acts = original_alter_labels, original_swap_events, original_duplicate_acts

            else:
                alter_labels, swap_events, duplicate_acts = False, False, False

            if preprocessed_test_data is None and determ_noised_dataset is not None:
                preprocessed_test_data = determ_noised_dataset

            elif preprocessed_test_data is None or gradual_noise is False:
                preprocessed_test_data = pre_process_log(test_data.copy(deep=True).reset_index(drop=True))

            else:
                preprocessed_test_data = copy_dataframe(preprocessed_test_data_noised)

            if by_trace is True:
                #                 print('dataset before noise:')
                #                 display(preprocessed_test_data.head(30))
                preprocessed_test_data_noised = add_noise_by_trace(preprocessed_test_data,
                                                                   disturbed_trans_frac=uncert_level,
                                                                   true_trans_prob=true_prob,
                                                                   expansion_min=expansion_min,
                                                                   expansion_max=expansion_max,
                                                                   randomized_true_trans_prob=False,
                                                                   generate_probs_for_argmax=generate_probs_for_argmax,
                                                                   gradual_noise=gradual_noise,
                                                                   alter_labels=alter_labels, swap_events=swap_events,
                                                                   duplicate_acts=duplicate_acts,
                                                                   fraq=determ_noise_fraq)
            else:
                preprocessed_test_data_noised = add_noise(preprocessed_test_data, disturbed_trans_frac=uncert_level,
                                                          true_trans_prob=true_prob, expansion_min=expansion_min,
                                                          expansion_max=expansion_max,
                                                          randomized_true_trans_prob=False,
                                                          generate_probs_for_argmax=generate_probs_for_argmax)

            return preprocessed_test_data_noised
            #            return preprocessed_test_data_noised

            #             print()
            #             print('dataset after noise')
            #             display(preprocessed_test_data_noised.head(30))
            if all_probs_one is True:
                preprocessed_test_data_noised = make_all_probs_one(preprocessed_test_data_noised)

            if eval_lower_bound is True:
                preprocessed_noised_determ = copy_dataframe(preprocessed_test_data_noised)
                preprocessed_noised_determ = make_all_probs_one(preprocessed_noised_determ)
                lower_bound_score = calc_conf_for_log(preprocessed_noised_determ, discovered_net,
                                                      non_sync_move_penalty=non_sync_penalty,
                                                      add_heuristic=add_heuristic)
                lower_bound_scores.append(lower_bound_score)

            #             non_sync_non_quiet_stochastic_alignment_length, sync_stochastic_alignment_length, \
            #             non_sync_non_quiet_argmax_length, sync_argmax_alignment_length, stochastic_acc, \
            #             argmax_acc, traces_length, stoc_conf_lst, fitness_lst = generate_statistics_for_dataset(preprocessed_test_data_noised, test_data, discovered_net, shortest_path_in_model, non_sync_penalty, add_heuristic)
            stoc_conf_lst = generate_statistics_for_dataset(preprocessed_test_data_noised, test_data, discovered_net,
                                                            shortest_path_in_model, non_sync_penalty, add_heuristic)

            #             mean_stochastic_avg = (np.array(stochastic_acc).dot(np.array(traces_length))) / sum(traces_length)
            #             mean_argmax_avg = (np.array(argmax_acc).dot(np.array(traces_length))) / sum(traces_length)

            #             non_sync_stochastic_avgs.append(mean(non_sync_non_quiet_stochastic_alignment_length))
            #             sync_stochastic_avgs.append(mean(sync_stochastic_alignment_length))
            #             non_sync_argmax_avgs.append(mean(non_sync_non_quiet_argmax_length))
            #             sync_argmax_avgs.append(mean(sync_argmax_alignment_length))
            #             stochastic_acc_avgs.append(mean_stochastic_avg)
            #             argmax_acc_avgs.append(mean_argmax_avg)
            stochastic_conformance_avgs.append(mean(stoc_conf_lst))
            #             fitness_avgs.append(mean(fitness_lst))

            determ_preprocess_flag = 0

            print(
                f'true prob: {true_prob},  uncertainty_level: {uncert_level},  mean_stochastic_conformance: {mean(stoc_conf_lst)}')
            print('--------------------------------------------------------------------------------------------------')
            print()
    #     return non_sync_stochastic_avgs, sync_stochastic_avgs, non_sync_argmax_avgs, sync_argmax_avgs, \
    #            stochastic_acc_avgs, argmax_acc_avgs, stochastic_conformance_avgs, fitness_avgs, lower_bound_scores
    return stochastic_conformance_avgs, lower_bound_scores


def generate_stats_dict_constant_stochastic_traces_frequency(history):
    '''
    Dictionary where keys are stochastic traces frequency and values are frequency
    of alignments varied by the probability of the true transition
    '''

    stats_dict = defaultdict(list)
    stochastic_traces_frequency = 0
    for i in range(21):
        j = i
        while j < 441:
            stats_dict[stochastic_traces_frequency].append(history[j])
            j += 21

        stochastic_traces_frequency += 0.05
        stochastic_traces_frequency = round(stochastic_traces_frequency, 2)

    return stats_dict


def filter_log(log, n_traces, max_len, random_selection=False, random_seed=42):
    cases_list = list(log['case:concept:name'].unique())
    accepted_cases = []

    for case in cases_list:
        case_length = len(log[log['case:concept:name'] == case])

        if case_length <= max_len:
            accepted_cases.append(case)

    if random_selection:
        random.seed(random_seed)
        final_cases = random.sample(accepted_cases, n_traces)

    else:
        final_cases = accepted_cases[:n_traces]

    filtered_df = log[log['case:concept:name'].isin(final_cases)]

    return filtered_df


def prepare_df_cols_for_discovery(df):
    df_copy = df.copy()
    df_copy.loc[:, 'order'] = df_copy.groupby('case:concept:name').cumcount()
    df_copy.loc[:, 'time:timestamp'] = pd.to_datetime(df_copy['order'])

    return df_copy


def get_df_trace_lengths(df):
    return df.groupby(["case:concept:name"])['concept:name'].count().reset_index(name='count')['count'].values


def evaluate_conformance_cost(log, model, non_sync_penalty=1):
    cases_list = list(log['case:concept:name'].unique())
    conf_costs = []
    for case in cases_list:
        trace_case = log[log['case:concept:name'] == case]
        trace_case = trace_case.drop('case:concept:name', axis=1)
        trace_case = trace_case.reset_index(drop=True)
        #         display(trace_case)
        trace_case_model = construct_trace_model(trace_case, non_sync_move_penalty=non_sync_penalty)
        trace_alignment, trace_cost = model.conformance_checking(trace_case_model)
        #         print(f'trace cost: {trace_cost}')
        #         print(f'trace alignment: {trace_alignment}')
        conf_costs.append(trace_cost)

    return conf_costs


def trace_self_loops_indices(trace_df):
    trace_df_copy = copy.copy(trace_df)
    trace_df_copy['concept:name_shifted'] = trace_df_copy['concept:name'].shift(1)
    trace_df_copy['case:concept:name_shifted'] = trace_df_copy['case:concept:name'].shift(1)
    trace_df_copy['is_duplicate'] = trace_df_copy.apply(check_duplicate_rows, axis=1)
    return trace_df_copy['is_duplicate'].to_numpy().astype(np.bool)


def sample_n_traces(df, n_traces=10, random=True):
    if random is False:
        trace_cases = list(df['case:concept:name'].unique())[:n_traces]
    else:
        trace_cases = sample(list(df['case:concept:name'].unique()), n_traces)
    return df[df['case:concept:name'].isin(trace_cases)]


def remove_self_loops_in_trace(trace_df):
    return trace_df[(trace_df['concept:name'] != trace_df.shift(1)['concept:name']) | (
            trace_df['case:concept:name'] != trace_df.shift(1)['case:concept:name'])]


def remove_self_loops_in_dataset(log_df, return_self_loops_indices=False):
    trace_cases = list(log_df['case:concept:name'].unique())

    new_traces_lst = []
    self_loop_indices_lst = []
    for trace_case in trace_cases:
        trace = log_df[log_df['case:concept:name'] == trace_case]
        curr_trace_self_loop_indices = trace_self_loops_indices(trace)
        trace = remove_self_loops_in_trace(trace)
        new_traces_lst.append(trace)
        self_loop_indices_lst += list(curr_trace_self_loop_indices)

    new_log_df = pd.concat(new_traces_lst)
    if return_self_loops_indices:
        return new_log_df, np.array(self_loop_indices_lst)
    return new_log_df


def check_duplicate_rows(row):
    if row['concept:name'] != row['concept:name_shifted'] or row['case:concept:name_shifted'] != row[
        'case:concept:name_shifted']:
        return 0
    return 1


def argmax_sk_trace(trace_df):
    trace_df['argmax_activity_label'] = trace_df.apply(get_max_prob_activity, axis=1)
    return trace_df[['argmax_activity_label', 'case:concept:name']]


def get_max_prob_activity(row):
    max_idx = row['probs'].index(max(row['probs']))
    max_prob_activity = row['concept:name'][max_idx]
    return max_prob_activity


def filter_softmax_matrice(sftm_mat, is_dup_bool_vec):
    np_sftm_mat = sftm_mat.squeeze(0).cpu().numpy()
    return np_sftm_mat[:, np.invert(is_dup_bool_vec)]


def remove_loops_in_trace_and_matrice(trace_df, sftm_mat):
    self_loops_indices = trace_self_loops_indices(trace_df)
    no_loops_trace = remove_self_loops_in_trace(trace_df)
    no_loops_sftm_mat = filter_softmax_matrice(sftm_mat, self_loops_indices)
    return no_loops_trace, no_loops_sftm_mat


def remove_loops_in_log_and_sftm_matrices_lst(log_df, sftm_mat_lst):
    trace_cases = list(log_df['case:concept:name'].unique())
    no_loops_trace_lst = []
    no_loops_sftm_mat_lst = []

    for i, case in enumerate(trace_cases):
        trace = log_df[log_df['case:concept:name'] == case]
        loop_indices = trace_self_loops_indices(trace)
        no_loops_trace = remove_self_loops_in_trace(trace)
        no_loop_stmx_mat = filter_softmax_matrice(sftm_mat_lst[i], loop_indices)
        no_loops_trace_lst.append(no_loops_trace)
        no_loops_sftm_mat_lst.append(no_loop_stmx_mat)

    return pd.concat(no_loops_trace_lst), no_loops_sftm_mat_lst


def sfmx_mat_to_sk_trace(sftm_mat, case_num, round_precision=2):
    if type(sftm_mat) is torch.Tensor:
        sftm_mat = sftm_mat.squeeze(0).cpu().numpy()

    activities_arr = np.arange(19)
    df_prob_lst = []
    df_activities_lst = []
    di = activity_map_dict()

    for i in range(sftm_mat.shape[1]):
        probs = np.round(sftm_mat[:, i], round_precision)
        activities = list(activities_arr[np.nonzero(probs)])
        activities = [di[str(act)] for act in activities]
        df_prob_lst.append(list(probs[np.nonzero(probs)]))
        df_activities_lst.append(activities)

    case_lst = [case_num] * sftm_mat.shape[1]

    df = pd.DataFrame(
        {'concept:name': df_activities_lst,
         'case:concept:name': case_lst,
         'probs': df_prob_lst
         })

    return df


def argmax_sftmx_matrice(stmx_mat):
    return np.argmax(stmx_mat, axis=0)


def train_test_log_split(log, n_traces, random_selection=False, random_seed=42):
    cases_list = list(log['case:concept:name'].unique())
    assert len(
        cases_list) >= n_traces, "Houston we've got a problem - more traces were demanded than there were available"

    if random_selection:
        random.seed(random_seed)
        final_cases = random.sample(cases_list, n_traces)

    else:
        final_cases = cases_list[:n_traces]

    train_df = log[log['case:concept:name'].isin(final_cases)]
    test_df = log[~log['case:concept:name'].isin(final_cases)]

    return train_df, test_df


def select_stmx_mats_for_test(sfmx_mats, indices_lst):
    if isinstance(indices_lst, pd.core.frame.DataFrame):
        indices = sorted([int(num) for num in indices_lst['case:concept:name'].unique().tolist()])

    elif isinstance(indices_lst, pd.core.series.Series):
        indices = sorted([int(num) for num in indices_lst.unique().tolist()])

    else:
        indices = sorted([int(num) for num in indices_lst])

    return [sfmx_mats[i] for i in indices]


def logarithmic(p):
    return -np.log(p) / 2.4


def shorten_df(df, stmx_lst, max_trace_len=4, remove_self_loops=True):
    if remove_self_loops:
        df, stmx = remove_loops_in_log_and_sftm_matrices_lst(df, stmx_lst)

    shortened_traces = []
    cases = list(df['case:concept:name'].unique())

    for idx, trace_case in enumerate(cases):
        trace_df = df[df['case:concept:name'] == trace_case]
        trace_df = trace_df.head(max_trace_len)
        shortened_traces.append(trace_df)

    stm_short = [mat[:, :max_trace_len] for mat in stmx]
    return pd.concat(shortened_traces), stm_short


def activity_map_dict():
    di = {'0': 'cut_tomato',
          '1': 'place_tomato_into_bowl',
          '2': 'cut_cheese',
          '3': 'place_cheese_into_bowl',
          '4': 'cut_lettuce',
          '5': 'place_lettuce_into_bowl',
          '6': 'add_salt',
          '7': 'add_vinegar',
          '8': 'add_oil',
          '9': 'add_pepper',
          '10': 'mix_dressing',
          '11': 'peel_cucumber',
          '12': 'cut_cucumber',
          '13': 'place_cucumber_into_bowl',
          '14': 'add_dressing',
          '15': 'mix_ingredients',
          '16': 'serve_salad_onto_plate',
          '17': 'action_start',
          '18': 'action_end'}

    return di


def compare_stochastic_vs_argmax_no_loops(df, softmax_lst, net=None, init_marking=None, final_marking=None,
                                          n_train_traces=10, cost_function=None, round_precision=2,
                                          random_trace_selection=True, random_seed=42, non_sync_penalty=1,
                                          remove_self_loops=True, max_trace_len=None):
    if cost_function is None:
        cost_function = logarithmic

    if remove_self_loops and max_trace_len is None:
        df_no_loops, stmx_lst_no_loops = remove_loops_in_log_and_sftm_matrices_lst(df, softmax_lst)

    if max_trace_len is not None:
        df_no_loops, stmx_lst_no_loops = shorten_df(df, softmax_lst, max_trace_len=max_trace_len)

    df_train, df_test = train_test_log_split(df_no_loops, n_traces=n_train_traces,
                                             random_selection=random_trace_selection, random_seed=random_seed)
    stmx_matrices_test = select_stmx_mats_for_test(stmx_lst_no_loops, df_test)

    if net is None:
        net, init_marking, final_marking = pm4py.discover_petri_net_inductive(df_train)
    #     model = from_discovered_model_to_PetriNet(net, non_sync_move_penalty=non_sync_penalty, cost_function=cost_function)
    #     gviz = pn_visualizer.apply(net, init_marking, final_marking)
    #     pn_visualizer.view(gviz)

    model = from_discovered_model_to_PetriNet(net, non_sync_move_penalty=non_sync_penalty)
    test_traces_cases = list(df_test['case:concept:name'].unique())
    stochastic_acc_lst = []
    argmax_acc_lst = []
    for idx, trace_case in enumerate(test_traces_cases):
        true_trace_df = df_test[df_test['case:concept:name'] == trace_case]
        stochastic_trace_df = sfmx_mat_to_sk_trace(stmx_matrices_test[idx], trace_case, round_precision=round_precision)
        #         display(stochastic_trace_df)
        stochastic_acc, argmax_acc = compare_argmax_and_stochastic_alignments(stochastic_trace_df, true_trace_df, model,
                                                                              non_sync_penalty=non_sync_penalty)
        stochastic_acc_lst.append(stochastic_acc)
        argmax_acc_lst.append(argmax_acc)

    return stochastic_acc_lst, argmax_acc_lst


def compare_stochastic_vs_argmax_random_indices(df, softmax_lst, cost_function=None, n_train_traces=10, n_indices=100,
                                                round_precision=2, random_trace_selection=True, random_seed=42,
                                                non_sync_penalty=1):
    if cost_function is None:
        cost_function = logarithmic

    df_random_indices, stmx_lst_random_indices = select_random_indices_in_log_and_sftm_matrices_lst(df, softmax_lst,
                                                                                                    n_indices)
    df_train, df_test = train_test_log_split(df_random_indices, n_traces=n_train_traces,
                                             random_selection=random_trace_selection, random_seed=random_seed)
    stmx_matrices_test = select_stmx_mats_for_test(stmx_lst_random_indices, df_test)

    df_train = prepare_df_cols_for_discovery(df_train)
    net, init_marking, final_marking = pm4py.discover_petri_net_inductive(df_train)
    #     model = from_discovered_model_to_PetriNet(net, non_sync_move_penalty=non_sync_penalty, cost_function=cost_function)
    model = from_discovered_model_to_PetriNet(net, non_sync_move_penalty=non_sync_penalty)

    test_traces_cases = list(df_test['case:concept:name'].unique())
    stochastic_acc_lst = []
    argmax_acc_lst = []
    for idx, trace_case in enumerate(test_traces_cases):
        true_trace_df = df_test[df_test['case:concept:name'] == trace_case].reset_index(drop=True)
        stochastic_trace_df = sfmx_mat_to_sk_trace(stmx_matrices_test[idx], trace_case, round_precision=round_precision)
        #         display(true_trace_df)
        #         display(stochastic_trace_df)
        stochastic_acc, argmax_acc = compare_argmax_and_stochastic_alignments(stochastic_trace_df, true_trace_df, model,
                                                                              non_sync_penalty=non_sync_penalty)
        stochastic_acc_lst.append(stochastic_acc)
        argmax_acc_lst.append(argmax_acc)

    return stochastic_acc_lst, argmax_acc_lst


def select_random_indices_in_log_and_sftm_matrices_lst(log_df, sftm_mat_lst, n_indices=100):
    trace_cases = list(log_df['case:concept:name'].unique())
    filtered_trace_lst = []
    filtered_sftm_mat_lst = []

    for i, case in enumerate(trace_cases):
        trace = log_df[log_df['case:concept:name'] == case]
        np_sftm_mat = sftm_mat_lst[i].squeeze(0).cpu().numpy()
        if n_indices is None:
            n_indices = np_sftm_mat.shape[1]
        selected_indices = sample(list(range(np_sftm_mat.shape[1])), n_indices)
        selected_indices_bool = np.zeros(np_sftm_mat.shape[1], dtype=bool)
        np.add.at(selected_indices_bool, selected_indices, 1)
        sftm_filtered = np_sftm_mat[:, selected_indices_bool]
        trace_df_filtered = trace[selected_indices_bool]
        filtered_trace_lst.append(trace_df_filtered)
        filtered_sftm_mat_lst.append(sftm_filtered)
    return pd.concat(filtered_trace_lst), filtered_sftm_mat_lst


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


if __name__ == "__main__":
    with open('../data/pickles/salads_softmax_lst.pickle', 'rb') as handle:
        softmax_lst = CPU_Unpickler(handle).load()

    with open('../data/pickles/salads_target_lst.pickle', 'rb') as handle:
        target_lst = CPU_Unpickler(handle).load()

    concant_tensor_lst = []
    concat_idx_lst = []

    for i, tensor in enumerate(target_lst):
        tensor_lst = tensor.tolist()
        tensor_lst = [str(elem) for elem in tensor_lst]
        idx_lst = [str(i)] * len(tensor_lst)
        concant_tensor_lst += tensor_lst
        concat_idx_lst += idx_lst

    df = pd.DataFrame(
        {'concept:name': concant_tensor_lst,
         'case:concept:name': concat_idx_lst
         })

    di = activity_map_dict()
    df["concept:name"] = df["concept:name"].replace(di)
    stochastic_acc, argmax_acc = compare_stochastic_vs_argmax_random_indices(df, softmax_lst, n_train_traces=20, n_indices=100)
    print(mean(stochastic_acc), mean(argmax_acc))
