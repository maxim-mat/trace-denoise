class Edge:
    def __init__(self, name, source_marking, target_marking, move_type):
        self.name = name
        self.source_marking = source_marking
        self.target_marking = target_marking
        self.move_type = move_type

    def __repr__(self):
        return f'{self.source_marking} -> {self.name} -> {self.target_marking}'
