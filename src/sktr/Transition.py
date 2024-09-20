import numpy as np


class Transition:
    def __init__(self, name, label, in_arcs=None, out_arcs=None, move_type=None, prob=None, weight=None,
                 location_idx=None, cost_function=None, properties=None):
        self.name = name
        self.label = label
        self.in_arcs = set() if in_arcs is None else in_arcs
        self.out_arcs = set() if out_arcs is None else out_arcs
        self.move_type = move_type
        self.prob = prob
        self.cost_function = cost_function
        self.weight = self.__initialize_weight(weight)
        self.properties = dict() if properties is None else properties
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
