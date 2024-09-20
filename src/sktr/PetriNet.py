import numpy as np
from collections import defaultdict


class PetriNet:
    def __init__(self, name, places=None, transitions=None, arcs=None, trace_len=None, properties=None):
        self.name = name
        self.transitions = list() if transitions is None else transitions
        self.places = list() if places is None else places
        self.arcs = list() if arcs is None else arcs
        self.properties = dict() if properties is None else properties
        self.init_mark = None
        self.final_mark = None
        self.reachability_graph = None
        self.places_indices = dict() if places is None else {self.places[i].name: i for i in range(len(self.places))}
        self.transitions_indices = dict() if transitions is None else {self.transitions[i].name: i for i in
                                                                       range(len(self.transitions))}
        self.dis_associated_indices = None
        self.trace_transitions = None
        self.transitions_weights = list() if transitions is None else np.array([t.weight for t in transitions])
        self.trace_len = trace_len
        self.cost_function = None

    def update_transitions_weights(self):
        transitions_weights = np.array([t.weight for t in self.transitions])
        return transitions_weights

    def compute_conditioned_weight(self, path_prefix, transition, prob_dict, lamda=0.5):

        if prob_dict is None:
            return transition.weight

        if transition.label is None:
            return 0

        transition_weight = transition.weight
        transition_label = transition.label
        full_path = path_prefix + transition_label

        if path_prefix == '':
            return transition_weight

        if full_path in prob_dict:
            return (1 - lamda) * ((1 - prob_dict[full_path]) * transition_weight) + lamda * transition_weight

        longest_prefix = self.find_longest_prefix(full_path, prob_dict)

        if longest_prefix:
            return (1 - lamda) * ((1 - prob_dict[longest_prefix]) * transition_weight) + lamda * transition_weight

        return transition_weight

    @staticmethod
    def find_longest_prefix(full_path, prob_dict):
        longest_prefix = None
        for i in range(len(full_path) - 1):
            if full_path[i:] in prob_dict:
                return full_path[i:]

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
