"""
Core Components
===============

Core components of the context engineering system including neural fields,
protocol orchestration, and cognitive processing.
"""

from .field import ContextField, FieldManager
from .protocol_orchestrator import ProtocolOrchestrator  
from .cognitive_processor import CognitiveProcessor

__all__ = [
    "ContextField",
    "FieldManager",
    "ProtocolOrchestrator", 
    "CognitiveProcessor"
]