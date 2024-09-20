class Place:
    def __init__(self, name, in_arcs=None, out_arcs=None, properties=None):
        self.name = name
        self.in_arcs = set() if in_arcs is None else in_arcs
        self.out_arcs = set() if out_arcs is None else out_arcs
        self.properties = dict() if properties is None else properties

    def __repr__(self):
        return self.name
