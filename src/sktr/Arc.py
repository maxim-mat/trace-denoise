class Arc:
    def __init__(self, source, target, weight=1, properties=None):
        self.source = source
        self.target = target
        self.weight = weight
        self.properties = dict() if properties is None else properties

    def __repr__(self):
        return self.source.name + ' -> ' + self.target.name
