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
