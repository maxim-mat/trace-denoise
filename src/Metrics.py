from dataclasses import dataclass


@dataclass
class Metrics:
    accuracy: float = None
    precision: float = None
    recall: float = None
    f1: float = None
    w2_dist: float = None
    auc: float = None
    conformance: float = None
