"""
DocPilot-AI — Agent Registry

All agents are exported from here for consistent imports across the app.

Usage:
    from agents import PlannerAgent, RetrievalAgent, SummarizerAgent, ...
"""

from .planner_agent import PlannerAgent
from .retrieval_agent import RetrievalAgent
from .summarizer_agent import SummarizerAgent
from .comparison_agent import ComparisonAgent
from .quiz_agent import QuizAgent

__all__ = [
    "PlannerAgent",
    "RetrievalAgent",
    "SummarizerAgent",
    "ComparisonAgent",
    "QuizAgent",
]
