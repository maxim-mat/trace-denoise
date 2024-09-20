class Marking:
    def __init__(self, places=None):
        self.places = (0, 0) if places is None else places

    def __repr__(self):
        return str(self.places)

    def __eq__(self, other):
        return self.places == other.places

    def __hash__(self):
        return hash(self.places)
