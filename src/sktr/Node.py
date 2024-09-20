class Node:
    def __init__(self, marking):
        self.marking = marking
        self.neighbors = set()

    def __repr__(self):
        return str(self.marking)

    def add_neighbor(self, node, transition):
        self.neighbors.add((node, transition))
