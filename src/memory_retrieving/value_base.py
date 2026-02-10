from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ValueBase(ABC):
    """
    Base class for value functions that score a memory item.
    Subclasses implement the four component scorers; the final value is their sum.
    """

    @abstractmethod
    def __init__(self):
        self.value_threshold = 0

    def evaluate(self, Importance: float = 0, Relevance: float = 0, Recency: float = 0, Return: float = 0, Credibility: float = 0) -> float:
        """
        Compute final value as the sum of four component values derived from inputs.
        """
        v_importance = self.compute_importance(Importance)
        v_relevance = self.compute_relevance(Relevance)
        v_recency = self.compute_recency(Recency)
        v_return = self.compute_return(Return)
        v_credibility = self.compute_credibility(Credibility)
        return float(v_importance + v_relevance + v_recency + v_return)

    @abstractmethod
    def compute_importance(self, Importance: float) -> float:
        """
        Transform raw 'importance' into a component value.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_relevance(self, Relevance: float) -> float:
        """
        Transform raw 'relevance' into a component value.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_recency(self, Recency: float) -> float:
        """
        Transform raw 'recency' into a component value.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_return(self, Return: float) -> float:
        """
        Transform raw 'return' into a component value.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_credibility(self, Credibility: float) -> float:
        """
        Transform raw 'credibility' into a component value.
        """
        raise NotImplementedError
