from memory_retrieving.value_base import ValueBase

class TagEqual(ValueBase):

    def __init__(self, k: int = 1, value_threshold: float = 0.9):
        self.value_threshold = value_threshold
        self.k = k  

    def compute_importance(self, Importance: float) -> float:
       return 0

    def compute_relevance(self, Relevance: float) -> float:
        return Relevance

    def compute_recency(self, Recency: float) -> float:
        return 0

    def compute_return(self, Return: float) -> float:
        return 0

    def compute_credibility(self, Credibility: float) -> float:
        return 0

class TagRelevant(ValueBase):

    def __init__(self, k: int = 1, value_threshold: float = 0.8):
        self.value_threshold = value_threshold
        self.k = k

    def compute_importance(self, Importance: float) -> float:
        return 0

    def compute_relevance(self, Relevance: float) -> float:
        return Relevance

    def compute_recency(self, Recency: float) -> float:
        return 0

    def compute_return(self, Return: float) -> float:
        return 0

    def compute_credibility(self, Credibility: float) -> float:
        return 0

class SemanticEqual(ValueBase):

    def __init__(self, k: int = 1, value_threshold: float = 0.9):
        self.value_threshold = value_threshold
        self.k = k

    def compute_importance(self, Importance: float) -> float:
        return 0

    def compute_relevance(self, Relevance: float) -> float:
        return Relevance

    def compute_recency(self, Recency: float) -> float:
        return 0

    def compute_return(self, Return: float) -> float:
        return 0
    
    def compute_credibility(self, Credibility: float) -> float:
        return 0

class SemanticRelevant(ValueBase):

    def __init__(self, k: int = 10, value_threshold: float = 0.0):
        self.value_threshold = value_threshold
        self.k = k

    def compute_importance(self, Importance: float) -> float:
        return 0

    def compute_relevance(self, Relevance: float) -> float:
        return Relevance

    def compute_recency(self, Recency: float) -> float:
        return 0

    def compute_return(self, Return: float) -> float:
        return 0

    def compute_credibility(self, Credibility: float) -> float:
        return 0


class SemanticRelevant4Episodic(ValueBase):

    def __init__(self, k: int = 30, value_threshold: float = 0.0):
        self.value_threshold = value_threshold
        self.k = k

    def compute_importance(self, Importance: float) -> float:
        return 0

    def compute_relevance(self, Relevance: float) -> float:
        return Relevance

    def compute_recency(self, Recency: float) -> float:
        return 0

    def compute_return(self, Return: float) -> float:
        return 0

    def compute_credibility(self, Credibility: float) -> float:
        return 0

class SubgoalEqual(ValueBase):

    def __init__(self, k: int = 1, value_threshold: float = 0.8):
        self.value_threshold = value_threshold
        self.k = k

    def compute_importance(self, Importance: float) -> float:
        return 0

    def compute_relevance(self, Relevance: float) -> float:
        return Relevance

    def compute_recency(self, Recency: float) -> float:
        return 0

    def compute_return(self, Return: float) -> float:
        return 0

    def compute_credibility(self, Credibility: float) -> float:
        return 0

class SubgoalRelevant(ValueBase):

    def __init__(self, k: int = 1, value_threshold: float = 0.1):
        self.value_threshold = value_threshold
        self.k = k

    def compute_importance(self, Importance: float) -> float:
        return 0

    def compute_relevance(self, Relevance: float) -> float:
        return Relevance

    def compute_recency(self, Recency: float) -> float:
        return 0

    def compute_return(self, Return: float) -> float:
        return 0
    
    def compute_credibility(self, Credibility: float) -> float:
        return 0

class ProceduralEqual(ValueBase):

    def __init__(self, k: int = 1, value_threshold: float = 0.8):
        self.value_threshold = value_threshold
        self.k = k

    def compute_importance(self, Importance: float) -> float:
       return 0

    def compute_relevance(self, Relevance: float) -> float:
        return Relevance

    def compute_recency(self, Recency: float) -> float:
        return 0

    def compute_return(self, Return: float) -> float:
       return 0
    
    def compute_credibility(self, Credibility: float) -> float:
        return 0

class ProceduralRelevant(ValueBase):

    def __init__(self, k: int = 1, value_threshold: float = 0.1):
        self.value_threshold = value_threshold
        self.k = k

    def compute_importance(self, Importance: float) -> float:
       return 0

    def compute_relevance(self, Relevance: float) -> float:
        return Relevance

    def compute_recency(self, Recency: float) -> float:
        return 0

    def compute_return(self, Return: float) -> float:
       return 0
    
    def compute_credibility(self, Credibility: float) -> float:
        return 0