import numpy as np


class SearchNode:
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
