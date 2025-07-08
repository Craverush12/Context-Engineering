"""
Context Engineering System
========================

A comprehensive system for field-based context engineering with neural fields,
protocol shells, cognitive tools, and meta-recursive capabilities.

This system represents a paradigm shift from discrete token-based context
management to continuous semantic field dynamics with emergent properties.
"""

__version__ = "0.1.0"
__author__ = "Context Engineering Team"

from .core.field import ContextField, FieldManager
from .core.protocol_orchestrator import ProtocolOrchestrator
from .core.cognitive_processor import CognitiveProcessor
from .parsers.pareto_lang import ParetoLangParser
from .visualizations.field_visualizer import FieldVisualizer

__all__ = [
    "ContextField",
    "FieldManager", 
    "ProtocolOrchestrator",
    "CognitiveProcessor",
    "ParetoLangParser",
    "FieldVisualizer"
]