"""
Interpretability Scaffolding
===========================

Provides transparent explanations of system behavior, decision-making,
and emergent properties. Enables users to understand:
- Why the system made specific decisions
- How different components contributed to outcomes
- What causal chains led to results
- Which symbolic transformations occurred
- How emergent behaviors arose
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
import logging
import json
from collections import defaultdict

from ..core.field import FieldManager, ContextField
from ..protocols.multi_protocol_orchestrator import ProtocolResult


class ExplanationType(Enum):
    """Types of explanations the system can generate."""
    DECISION = "decision"              # Why a specific decision was made
    BEHAVIOR = "behavior"              # How a behavior emerged
    TRANSFORMATION = "transformation"  # How data was transformed
    CAUSALITY = "causality"           # Causal chain of events
    CONTRIBUTION = "contribution"      # Component contributions
    EMERGENCE = "emergence"           # Emergent property explanation


class AttributionLevel(Enum):
    """Levels of attribution granularity."""
    COMPONENT = "component"    # High-level component attribution
    OPERATION = "operation"    # Operation-level attribution
    ELEMENT = "element"        # Individual element attribution
    FIELD = "field"           # Field-level attribution


@dataclass
class Attribution:
    """Attribution of outcome to specific system component or operation."""
    source: str                    # Component/operation that contributed
    contribution_type: str         # Type of contribution
    contribution_score: float      # Quantified contribution (0-1)
    evidence: List[Dict[str, Any]]  # Supporting evidence
    confidence: float              # Confidence in attribution
    timestamp: float = dataclass_field(default_factory=time.time)


@dataclass
class CausalLink:
    """Represents a causal relationship between events or states."""
    cause: Dict[str, Any]          # Causing event/state
    effect: Dict[str, Any]         # Resulting event/state
    relationship_type: str         # Type of causal relationship
    strength: float               # Strength of causal connection (0-1)
    latency: float               # Time delay between cause and effect
    evidence: List[Dict[str, Any]]  # Supporting evidence


@dataclass
class SymbolicTransformation:
    """Tracks transformation of symbolic representations."""
    input_symbol: str
    output_symbol: str
    transformation_type: str
    transformation_rule: str
    intermediate_steps: List[str]
    residue: Dict[str, Any]  # What was lost/gained in transformation
    timestamp: float = dataclass_field(default_factory=time.time)


@dataclass
class Explanation:
    """Complete explanation of system behavior or decision."""
    explanation_id: str
    explanation_type: ExplanationType
    target: str  # What is being explained
    summary: str  # Human-readable summary
    detailed_explanation: Dict[str, Any]
    attributions: List[Attribution]
    causal_chain: List[CausalLink]
    confidence: float
    timestamp: float = dataclass_field(default_factory=time.time)


class InterpretabilityScaffold:
    """
    Main interpretability framework providing comprehensive explanations
    of system behavior and decision-making.
    """
    
    def __init__(self, field_manager: FieldManager):
        """Initialize interpretability scaffold."""
        self.field_manager = field_manager
        self.attribution_tracer = AttributionTracer(self)
        self.causal_mapper = CausalMapper(self)
        self.symbolic_tracker = SymbolicResidueTracker(self)
        self.explanation_generator = ExplanationGenerator(self)
        
        # Tracking
        self.operation_history: List[Dict[str, Any]] = []
        self.decision_history: List[Dict[str, Any]] = []
        self.generated_explanations: Dict[str, Explanation] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def explain(self, target: str, explanation_type: ExplanationType,
                context: Optional[Dict[str, Any]] = None) -> Explanation:
        """
        Generate explanation for specified target and type.
        
        Args:
            target: What to explain (operation id, decision id, behavior name, etc.)
            explanation_type: Type of explanation to generate
            context: Additional context for explanation
            
        Returns:
            Complete explanation with attributions and causal chains
        """
        self.logger.info(f"Generating {explanation_type.value} explanation for: {target}")
        
        # Gather relevant data
        relevant_data = self._gather_relevant_data(target, explanation_type, context)
        
        # Trace attributions
        attributions = self.attribution_tracer.trace_attributions(
            target, relevant_data, explanation_type
        )
        
        # Map causal relationships
        causal_chain = self.causal_mapper.map_causality(
            target, relevant_data, explanation_type
        )
        
        # Track symbolic transformations if applicable
        symbolic_info = None
        if explanation_type in [ExplanationType.TRANSFORMATION, ExplanationType.EMERGENCE]:
            symbolic_info = self.symbolic_tracker.track_transformations(
                target, relevant_data
            )
        
        # Generate comprehensive explanation
        explanation = self.explanation_generator.generate_explanation(
            target=target,
            explanation_type=explanation_type,
            attributions=attributions,
            causal_chain=causal_chain,
            symbolic_info=symbolic_info,
            context=context
        )
        
        # Store explanation
        self.generated_explanations[explanation.explanation_id] = explanation
        
        return explanation
    
    def record_operation(self, operation_data: Dict[str, Any]):
        """Record operation for interpretability tracking."""
        self.operation_history.append({
            **operation_data,
            "timestamp": time.time()
        })
    
    def record_decision(self, decision_data: Dict[str, Any]):
        """Record decision for interpretability tracking."""
        self.decision_history.append({
            **decision_data,
            "timestamp": time.time()
        })
    
    def _gather_relevant_data(self, target: str, explanation_type: ExplanationType,
                            context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Gather all data relevant to generating explanation."""
        relevant_data = {
            "target": target,
            "type": explanation_type,
            "context": context or {},
            "operations": [],
            "decisions": [],
            "field_states": {}
        }
        
        # Gather relevant operations
        if explanation_type in [ExplanationType.DECISION, ExplanationType.BEHAVIOR]:
            relevant_data["operations"] = [
                op for op in self.operation_history
                if self._is_operation_relevant(op, target)
            ]
        
        # Gather relevant decisions
        if explanation_type == ExplanationType.DECISION:
            relevant_data["decisions"] = [
                dec for dec in self.decision_history
                if dec.get("decision_id") == target or 
                   dec.get("decision_name") == target
            ]
        
        # Gather field states
        for field_id, field in self.field_manager.active_fields.items():
            if self._is_field_relevant(field, target, explanation_type):
                relevant_data["field_states"][field_id] = {
                    "coherence": field.measure_field_coherence(),
                    "element_count": len(field.elements),
                    "attractor_count": len(field.attractors),
                    "recent_changes": self._get_recent_field_changes(field_id)
                }
        
        return relevant_data
    
    def _is_operation_relevant(self, operation: Dict[str, Any], target: str) -> bool:
        """Determine if operation is relevant to explanation target."""
        # Check direct match
        if operation.get("operation_id") == target:
            return True
        
        # Check if operation contributed to target
        if target in operation.get("outputs", []):
            return True
        
        # Check temporal proximity (within 5 seconds)
        if "timestamp" in operation:
            target_time = self._get_target_timestamp(target)
            if target_time and abs(operation["timestamp"] - target_time) < 5.0:
                return True
        
        return False
    
    def _is_field_relevant(self, field: ContextField, target: str,
                          explanation_type: ExplanationType) -> bool:
        """Determine if field is relevant to explanation."""
        # For emergence explanations, all active fields might be relevant
        if explanation_type == ExplanationType.EMERGENCE:
            return len(field.elements) > 0
        
        # For other types, check if field contains relevant elements
        return any(
            target in element_id or element_id in target
            for element_id in field.elements
        )
    
    def _get_recent_field_changes(self, field_id: str) -> List[Dict[str, Any]]:
        """Get recent changes to specified field."""
        # This would track actual field modifications
        return []
    
    def _get_target_timestamp(self, target: str) -> Optional[float]:
        """Get timestamp associated with target."""
        # Check operations
        for op in self.operation_history:
            if op.get("operation_id") == target:
                return op.get("timestamp")
        
        # Check decisions
        for dec in self.decision_history:
            if dec.get("decision_id") == target:
                return dec.get("timestamp")
        
        return None


class AttributionTracer:
    """Traces attributions of outcomes to system components."""
    
    def __init__(self, scaffold: InterpretabilityScaffold):
        self.scaffold = scaffold
        self.logger = logging.getLogger(__name__)
    
    def trace_attributions(self, target: str, relevant_data: Dict[str, Any],
                          explanation_type: ExplanationType) -> List[Attribution]:
        """
        Trace attributions for the target outcome.
        
        Args:
            target: What to attribute
            relevant_data: Data relevant to the target
            explanation_type: Type of explanation being generated
            
        Returns:
            List of attributions with contribution scores
        """
        attributions = []
        
        # Trace based on explanation type
        if explanation_type == ExplanationType.DECISION:
            attributions.extend(self._trace_decision_attributions(target, relevant_data))
        elif explanation_type == ExplanationType.BEHAVIOR:
            attributions.extend(self._trace_behavior_attributions(target, relevant_data))
        elif explanation_type == ExplanationType.TRANSFORMATION:
            attributions.extend(self._trace_transformation_attributions(target, relevant_data))
        elif explanation_type == ExplanationType.EMERGENCE:
            attributions.extend(self._trace_emergence_attributions(target, relevant_data))
        
        # Normalize contribution scores
        total_score = sum(attr.contribution_score for attr in attributions)
        if total_score > 0:
            for attr in attributions:
                attr.contribution_score /= total_score
        
        # Sort by contribution score
        attributions.sort(key=lambda x: x.contribution_score, reverse=True)
        
        return attributions
    
    def _trace_decision_attributions(self, target: str,
                                   relevant_data: Dict[str, Any]) -> List[Attribution]:
        """Trace attributions for a decision."""
        attributions = []
        
        # Find the decision
        decisions = relevant_data.get("decisions", [])
        if not decisions:
            return attributions
        
        decision = decisions[0]  # Assume first is most relevant
        
        # Attribute to decision factors
        for factor in decision.get("factors", []):
            attribution = Attribution(
                source=f"decision_factor_{factor['name']}",
                contribution_type="decision_input",
                contribution_score=factor.get("weight", 0.5),
                evidence=[{"factor": factor}],
                confidence=0.8
            )
            attributions.append(attribution)
        
        # Attribute to relevant operations
        for operation in relevant_data.get("operations", []):
            if self._operation_influenced_decision(operation, decision):
                attribution = Attribution(
                    source=f"operation_{operation.get('operation_id', 'unknown')}",
                    contribution_type="operational_influence",
                    contribution_score=0.3,
                    evidence=[{"operation": operation}],
                    confidence=0.6
                )
                attributions.append(attribution)
        
        return attributions
    
    def _trace_behavior_attributions(self, target: str,
                                   relevant_data: Dict[str, Any]) -> List[Attribution]:
        """Trace attributions for a behavior."""
        attributions = []
        
        # Attribute to field states
        for field_id, field_state in relevant_data.get("field_states", {}).items():
            if field_state["element_count"] > 0:
                # Calculate field's contribution based on activity
                contribution = self._calculate_field_contribution(field_state)
                
                attribution = Attribution(
                    source=f"field_{field_id}",
                    contribution_type="field_dynamics",
                    contribution_score=contribution,
                    evidence=[{"field_state": field_state}],
                    confidence=0.7
                )
                attributions.append(attribution)
        
        # Attribute to operations that shaped the behavior
        for operation in relevant_data.get("operations", []):
            attribution = Attribution(
                source=f"operation_{operation.get('operation_id', 'unknown')}",
                contribution_type="behavior_shaping",
                contribution_score=0.2,
                evidence=[{"operation": operation}],
                confidence=0.5
            )
            attributions.append(attribution)
        
        return attributions
    
    def _trace_transformation_attributions(self, target: str,
                                         relevant_data: Dict[str, Any]) -> List[Attribution]:
        """Trace attributions for a transformation."""
        attributions = []
        
        # Find transformation operations
        for operation in relevant_data.get("operations", []):
            if operation.get("operation_type") == "transformation":
                attribution = Attribution(
                    source=f"transformer_{operation.get('transformer_id', 'unknown')}",
                    contribution_type="transformation_execution",
                    contribution_score=0.8,
                    evidence=[{"operation": operation}],
                    confidence=0.9
                )
                attributions.append(attribution)
        
        return attributions
    
    def _trace_emergence_attributions(self, target: str,
                                    relevant_data: Dict[str, Any]) -> List[Attribution]:
        """Trace attributions for emergent properties."""
        attributions = []
        
        # Emergent properties arise from field interactions
        field_states = relevant_data.get("field_states", {})
        
        # Calculate field interaction contributions
        field_ids = list(field_states.keys())
        for i, field_id_a in enumerate(field_ids):
            for field_id_b in field_ids[i+1:]:
                interaction_strength = self._calculate_field_interaction(
                    field_states[field_id_a], field_states[field_id_b]
                )
                
                if interaction_strength > 0.1:
                    attribution = Attribution(
                        source=f"field_interaction_{field_id_a}_{field_id_b}",
                        contribution_type="emergent_interaction",
                        contribution_score=interaction_strength,
                        evidence=[{
                            "fields": [field_id_a, field_id_b],
                            "interaction_strength": interaction_strength
                        }],
                        confidence=0.6
                    )
                    attributions.append(attribution)
        
        # Attribute to system-level properties
        if len(field_states) > 3:
            attribution = Attribution(
                source="system_complexity",
                contribution_type="complexity_threshold",
                contribution_score=0.3,
                evidence=[{"active_fields": len(field_states)}],
                confidence=0.5
            )
            attributions.append(attribution)
        
        return attributions
    
    def _operation_influenced_decision(self, operation: Dict[str, Any],
                                     decision: Dict[str, Any]) -> bool:
        """Check if operation influenced decision."""
        # Temporal check - operation before decision
        op_time = operation.get("timestamp", 0)
        dec_time = decision.get("timestamp", 0)
        
        if op_time >= dec_time:
            return False
        
        # Check if operation outputs are decision inputs
        op_outputs = set(operation.get("outputs", []))
        dec_inputs = set(decision.get("inputs", []))
        
        return bool(op_outputs & dec_inputs)
    
    def _calculate_field_contribution(self, field_state: Dict[str, Any]) -> float:
        """Calculate field's contribution score."""
        # Based on coherence, element count, and attractor count
        coherence = field_state.get("coherence", 0)
        element_factor = min(1.0, field_state.get("element_count", 0) / 50)
        attractor_factor = min(1.0, field_state.get("attractor_count", 0) / 10)
        
        return (coherence * 0.5 + element_factor * 0.3 + attractor_factor * 0.2)
    
    def _calculate_field_interaction(self, field_a: Dict[str, Any],
                                   field_b: Dict[str, Any]) -> float:
        """Calculate interaction strength between two fields."""
        # Simplified calculation based on field properties
        coherence_product = field_a.get("coherence", 0) * field_b.get("coherence", 0)
        size_factor = min(field_a.get("element_count", 0), 
                         field_b.get("element_count", 0)) / 50
        
        return coherence_product * size_factor


class CausalMapper:
    """Maps causal relationships in system behavior."""
    
    def __init__(self, scaffold: InterpretabilityScaffold):
        self.scaffold = scaffold
        self.logger = logging.getLogger(__name__)
    
    def map_causality(self, target: str, relevant_data: Dict[str, Any],
                     explanation_type: ExplanationType) -> List[CausalLink]:
        """
        Map causal relationships leading to target outcome.
        
        Args:
            target: Target outcome
            relevant_data: Relevant data for analysis
            explanation_type: Type of explanation
            
        Returns:
            List of causal links forming causal chain
        """
        causal_chain = []
        
        # Build causal graph
        causal_graph = self._build_causal_graph(relevant_data)
        
        # Find causal paths to target
        target_node = self._find_target_node(target, causal_graph)
        if target_node:
            causal_paths = self._trace_causal_paths(target_node, causal_graph)
            
            # Convert paths to causal links
            for path in causal_paths:
                for i in range(len(path) - 1):
                    link = self._create_causal_link(path[i], path[i+1], causal_graph)
                    if link and link not in causal_chain:
                        causal_chain.append(link)
        
        # Sort by temporal order
        causal_chain.sort(key=lambda x: x.cause.get("timestamp", 0))
        
        return causal_chain
    
    def _build_causal_graph(self, relevant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build causal graph from relevant data."""
        graph = {
            "nodes": {},
            "edges": []
        }
        
        # Add operation nodes
        for operation in relevant_data.get("operations", []):
            node_id = f"op_{operation.get('operation_id', 'unknown')}"
            graph["nodes"][node_id] = {
                "type": "operation",
                "data": operation
            }
        
        # Add decision nodes
        for decision in relevant_data.get("decisions", []):
            node_id = f"dec_{decision.get('decision_id', 'unknown')}"
            graph["nodes"][node_id] = {
                "type": "decision",
                "data": decision
            }
        
        # Add field state nodes
        for field_id, field_state in relevant_data.get("field_states", {}).items():
            node_id = f"field_{field_id}"
            graph["nodes"][node_id] = {
                "type": "field_state",
                "data": field_state
            }
        
        # Build edges based on dependencies and temporal relationships
        self._build_causal_edges(graph)
        
        return graph
    
    def _build_causal_edges(self, graph: Dict[str, Any]):
        """Build causal edges in the graph."""
        nodes = graph["nodes"]
        
        # Connect operations based on input/output relationships
        for node_id_a, node_a in nodes.items():
            if node_a["type"] == "operation":
                outputs = set(node_a["data"].get("outputs", []))
                
                for node_id_b, node_b in nodes.items():
                    if node_id_a != node_id_b and node_b["type"] == "operation":
                        inputs = set(node_b["data"].get("inputs", []))
                        
                        if outputs & inputs:  # Shared elements indicate causality
                            graph["edges"].append({
                                "from": node_id_a,
                                "to": node_id_b,
                                "type": "data_flow",
                                "strength": len(outputs & inputs) / len(outputs | inputs)
                            })
        
        # Connect operations to decisions they influence
        for op_id, op_node in nodes.items():
            if op_node["type"] == "operation":
                for dec_id, dec_node in nodes.items():
                    if dec_node["type"] == "decision":
                        if self._operation_influences_decision(op_node["data"], dec_node["data"]):
                            graph["edges"].append({
                                "from": op_id,
                                "to": dec_id,
                                "type": "influence",
                                "strength": 0.7
                            })
    
    def _find_target_node(self, target: str, graph: Dict[str, Any]) -> Optional[str]:
        """Find node corresponding to target."""
        for node_id, node in graph["nodes"].items():
            if target in node_id:
                return node_id
            
            # Check if node data contains target
            if node["type"] == "operation":
                if node["data"].get("operation_id") == target:
                    return node_id
            elif node["type"] == "decision":
                if node["data"].get("decision_id") == target:
                    return node_id
        
        return None
    
    def _trace_causal_paths(self, target_node: str,
                           graph: Dict[str, Any]) -> List[List[str]]:
        """Trace all causal paths leading to target node."""
        paths = []
        
        # Use DFS to find all paths
        def dfs_paths(node: str, current_path: List[str], visited: Set[str]):
            if len(current_path) > 10:  # Limit depth
                return
            
            # Find edges leading to this node
            incoming_edges = [
                edge for edge in graph["edges"]
                if edge["to"] == node and edge["from"] not in visited
            ]
            
            if not incoming_edges:
                # Reached a root cause
                if len(current_path) > 1:
                    paths.append(list(reversed(current_path)))
            else:
                for edge in incoming_edges:
                    new_visited = visited | {edge["from"]}
                    dfs_paths(edge["from"], current_path + [edge["from"]], new_visited)
        
        dfs_paths(target_node, [target_node], {target_node})
        
        return paths
    
    def _create_causal_link(self, from_node: str, to_node: str,
                           graph: Dict[str, Any]) -> Optional[CausalLink]:
        """Create causal link between two nodes."""
        # Find edge
        edge = None
        for e in graph["edges"]:
            if e["from"] == from_node and e["to"] == to_node:
                edge = e
                break
        
        if not edge:
            return None
        
        # Get node data
        cause_data = graph["nodes"][from_node]["data"]
        effect_data = graph["nodes"][to_node]["data"]
        
        # Calculate latency
        cause_time = cause_data.get("timestamp", 0)
        effect_time = effect_data.get("timestamp", 0)
        latency = effect_time - cause_time if effect_time > cause_time else 0
        
        return CausalLink(
            cause={
                "node_id": from_node,
                "type": graph["nodes"][from_node]["type"],
                "summary": self._summarize_node(from_node, graph["nodes"][from_node])
            },
            effect={
                "node_id": to_node,
                "type": graph["nodes"][to_node]["type"],
                "summary": self._summarize_node(to_node, graph["nodes"][to_node])
            },
            relationship_type=edge["type"],
            strength=edge["strength"],
            latency=latency,
            evidence=[{"edge": edge}]
        )
    
    def _operation_influences_decision(self, operation: Dict[str, Any],
                                     decision: Dict[str, Any]) -> bool:
        """Check if operation influences decision."""
        # Temporal check
        if operation.get("timestamp", 0) >= decision.get("timestamp", 0):
            return False
        
        # Data flow check
        op_outputs = set(operation.get("outputs", []))
        dec_inputs = set(decision.get("inputs", []))
        
        return bool(op_outputs & dec_inputs)
    
    def _summarize_node(self, node_id: str, node: Dict[str, Any]) -> str:
        """Create summary of node for causal link."""
        if node["type"] == "operation":
            op_type = node["data"].get("operation_type", "unknown")
            return f"Operation {op_type}"
        elif node["type"] == "decision":
            dec_name = node["data"].get("decision_name", "unknown")
            return f"Decision: {dec_name}"
        elif node["type"] == "field_state":
            return f"Field state: {node_id}"
        else:
            return f"Unknown node: {node_id}"


class SymbolicResidueTracker:
    """Tracks symbolic transformations and their residues."""
    
    def __init__(self, scaffold: InterpretabilityScaffold):
        self.scaffold = scaffold
        self.transformation_history: List[SymbolicTransformation] = []
        self.logger = logging.getLogger(__name__)
    
    def track_transformations(self, target: str,
                            relevant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track symbolic transformations related to target.
        
        Args:
            target: Target to track transformations for
            relevant_data: Relevant data for analysis
            
        Returns:
            Dictionary of transformation information and residues
        """
        transformations = []
        total_residue = defaultdict(float)
        
        # Find transformation operations
        for operation in relevant_data.get("operations", []):
            if operation.get("operation_type") == "transformation":
                transformation = self._analyze_transformation(operation)
                if transformation:
                    transformations.append(transformation)
                    self.transformation_history.append(transformation)
                    
                    # Accumulate residue
                    for key, value in transformation.residue.items():
                        total_residue[key] += value
        
        # Analyze transformation chain
        chain_analysis = self._analyze_transformation_chain(transformations)
        
        return {
            "transformations": transformations,
            "total_residue": dict(total_residue),
            "chain_analysis": chain_analysis,
            "information_preserved": self._calculate_information_preservation(transformations)
        }
    
    def _analyze_transformation(self, operation: Dict[str, Any]) -> Optional[SymbolicTransformation]:
        """Analyze a single transformation operation."""
        if "input_symbol" not in operation or "output_symbol" not in operation:
            return None
        
        # Determine transformation type
        trans_type = self._determine_transformation_type(
            operation["input_symbol"], operation["output_symbol"]
        )
        
        # Extract transformation rule
        trans_rule = operation.get("transformation_rule", "implicit")
        
        # Calculate residue (what was lost or gained)
        residue = self._calculate_residue(operation)
        
        return SymbolicTransformation(
            input_symbol=operation["input_symbol"],
            output_symbol=operation["output_symbol"],
            transformation_type=trans_type,
            transformation_rule=trans_rule,
            intermediate_steps=operation.get("intermediate_steps", []),
            residue=residue
        )
    
    def _determine_transformation_type(self, input_symbol: str,
                                     output_symbol: str) -> str:
        """Determine type of transformation."""
        # Simple heuristics
        if len(output_symbol) > len(input_symbol):
            return "expansion"
        elif len(output_symbol) < len(input_symbol):
            return "compression"
        elif input_symbol.lower() != output_symbol.lower():
            return "mutation"
        else:
            return "refinement"
    
    def _calculate_residue(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate what was lost or gained in transformation."""
        residue = {}
        
        # Information content change (simplified)
        input_info = len(operation.get("input_symbol", ""))
        output_info = len(operation.get("output_symbol", ""))
        residue["information_delta"] = output_info - input_info
        
        # Semantic drift (would use embeddings in practice)
        residue["semantic_drift"] = 0.1  # Placeholder
        
        # Structural changes
        input_structure = operation.get("input_structure", {})
        output_structure = operation.get("output_structure", {})
        residue["structural_changes"] = len(set(output_structure.keys()) - 
                                          set(input_structure.keys()))
        
        return residue
    
    def _analyze_transformation_chain(self,
                                    transformations: List[SymbolicTransformation]) -> Dict[str, Any]:
        """Analyze chain of transformations."""
        if not transformations:
            return {"status": "no_transformations"}
        
        # Calculate cumulative effects
        total_expansion = sum(
            1 for t in transformations 
            if t.transformation_type == "expansion"
        )
        total_compression = sum(
            1 for t in transformations 
            if t.transformation_type == "compression"
        )
        
        # Check for cycles
        symbols = [t.input_symbol for t in transformations]
        symbols.append(transformations[-1].output_symbol if transformations else "")
        has_cycles = len(symbols) != len(set(symbols))
        
        return {
            "chain_length": len(transformations),
            "total_expansion": total_expansion,
            "total_compression": total_compression,
            "has_cycles": has_cycles,
            "dominant_type": max(
                set(t.transformation_type for t in transformations),
                key=lambda x: sum(1 for t in transformations if t.transformation_type == x)
            ) if transformations else "none"
        }
    
    def _calculate_information_preservation(self,
                                          transformations: List[SymbolicTransformation]) -> float:
        """Calculate how much information was preserved through transformations."""
        if not transformations:
            return 1.0
        
        # Simple calculation based on residues
        total_loss = sum(
            abs(t.residue.get("information_delta", 0))
            for t in transformations
        )
        
        # Normalize to 0-1 scale
        preservation = max(0, 1 - (total_loss / (len(transformations) * 10)))
        
        return preservation


class ExplanationGenerator:
    """Generates human-readable explanations from analysis results."""
    
    def __init__(self, scaffold: InterpretabilityScaffold):
        self.scaffold = scaffold
        self.template_engine = ExplanationTemplateEngine()
        self.logger = logging.getLogger(__name__)
    
    def generate_explanation(self, target: str, explanation_type: ExplanationType,
                           attributions: List[Attribution], causal_chain: List[CausalLink],
                           symbolic_info: Optional[Dict[str, Any]],
                           context: Optional[Dict[str, Any]]) -> Explanation:
        """
        Generate comprehensive explanation from analysis components.
        
        Args:
            target: What is being explained
            explanation_type: Type of explanation
            attributions: Attribution analysis results
            causal_chain: Causal analysis results
            symbolic_info: Symbolic transformation analysis
            context: Additional context
            
        Returns:
            Complete human-readable explanation
        """
        # Generate summary
        summary = self._generate_summary(
            target, explanation_type, attributions, causal_chain
        )
        
        # Generate detailed explanation
        detailed = self._generate_detailed_explanation(
            target, explanation_type, attributions, causal_chain, symbolic_info
        )
        
        # Calculate overall confidence
        confidence = self._calculate_explanation_confidence(
            attributions, causal_chain
        )
        
        # Create explanation
        explanation = Explanation(
            explanation_id=f"exp_{target}_{time.time()}",
            explanation_type=explanation_type,
            target=target,
            summary=summary,
            detailed_explanation=detailed,
            attributions=attributions,
            causal_chain=causal_chain,
            confidence=confidence
        )
        
        return explanation
    
    def _generate_summary(self, target: str, explanation_type: ExplanationType,
                         attributions: List[Attribution],
                         causal_chain: List[CausalLink]) -> str:
        """Generate human-readable summary."""
        if explanation_type == ExplanationType.DECISION:
            return self._generate_decision_summary(target, attributions, causal_chain)
        elif explanation_type == ExplanationType.BEHAVIOR:
            return self._generate_behavior_summary(target, attributions, causal_chain)
        elif explanation_type == ExplanationType.TRANSFORMATION:
            return self._generate_transformation_summary(target, attributions, causal_chain)
        elif explanation_type == ExplanationType.EMERGENCE:
            return self._generate_emergence_summary(target, attributions, causal_chain)
        else:
            return f"Explanation for {target}"
    
    def _generate_decision_summary(self, target: str, attributions: List[Attribution],
                                  causal_chain: List[CausalLink]) -> str:
        """Generate summary for decision explanation."""
        top_contributors = attributions[:3]
        contributor_text = ", ".join([
            f"{attr.source} ({attr.contribution_score:.1%})"
            for attr in top_contributors
        ])
        
        return (f"Decision '{target}' was primarily influenced by: {contributor_text}. "
                f"The causal chain involved {len(causal_chain)} steps.")
    
    def _generate_behavior_summary(self, target: str, attributions: List[Attribution],
                                  causal_chain: List[CausalLink]) -> str:
        """Generate summary for behavior explanation."""
        field_attributions = [
            attr for attr in attributions
            if attr.source.startswith("field_")
        ]
        
        if field_attributions:
            return (f"Behavior '{target}' emerged from interactions between "
                   f"{len(field_attributions)} fields with average contribution "
                   f"of {np.mean([a.contribution_score for a in field_attributions]):.1%}.")
        else:
            return f"Behavior '{target}' resulted from system operations."
    
    def _generate_transformation_summary(self, target: str, attributions: List[Attribution],
                                       causal_chain: List[CausalLink]) -> str:
        """Generate summary for transformation explanation."""
        transformer_attrs = [
            attr for attr in attributions
            if attr.source.startswith("transformer_")
        ]
        
        if transformer_attrs:
            return (f"Transformation '{target}' was executed by "
                   f"{len(transformer_attrs)} transformer(s) with "
                   f"{transformer_attrs[0].contribution_score:.1%} primary contribution.")
        else:
            return f"Transformation '{target}' completed."
    
    def _generate_emergence_summary(self, target: str, attributions: List[Attribution],
                                   causal_chain: List[CausalLink]) -> str:
        """Generate summary for emergence explanation."""
        interaction_attrs = [
            attr for attr in attributions
            if attr.contribution_type == "emergent_interaction"
        ]
        
        return (f"Emergent property '{target}' arose from "
               f"{len(interaction_attrs)} field interactions with "
               f"combined contribution of "
               f"{sum(a.contribution_score for a in interaction_attrs):.1%}.")
    
    def _generate_detailed_explanation(self, target: str, explanation_type: ExplanationType,
                                     attributions: List[Attribution],
                                     causal_chain: List[CausalLink],
                                     symbolic_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed explanation structure."""
        detailed = {
            "target": target,
            "type": explanation_type.value,
            "attribution_analysis": self._format_attribution_analysis(attributions),
            "causal_analysis": self._format_causal_analysis(causal_chain),
            "contributing_factors": self._identify_contributing_factors(attributions),
            "key_insights": self._extract_key_insights(attributions, causal_chain)
        }
        
        if symbolic_info:
            detailed["symbolic_analysis"] = self._format_symbolic_analysis(symbolic_info)
        
        return detailed
    
    def _format_attribution_analysis(self, attributions: List[Attribution]) -> Dict[str, Any]:
        """Format attribution analysis for explanation."""
        return {
            "top_contributors": [
                {
                    "source": attr.source,
                    "contribution": f"{attr.contribution_score:.1%}",
                    "type": attr.contribution_type,
                    "confidence": f"{attr.confidence:.1%}"
                }
                for attr in attributions[:5]  # Top 5
            ],
            "total_attributions": len(attributions),
            "attribution_distribution": self._calculate_attribution_distribution(attributions)
        }
    
    def _format_causal_analysis(self, causal_chain: List[CausalLink]) -> Dict[str, Any]:
        """Format causal analysis for explanation."""
        return {
            "causal_steps": [
                {
                    "step": i + 1,
                    "cause": link.cause["summary"],
                    "effect": link.effect["summary"],
                    "relationship": link.relationship_type,
                    "strength": f"{link.strength:.1%}",
                    "latency": f"{link.latency:.2f}s"
                }
                for i, link in enumerate(causal_chain)
            ],
            "total_steps": len(causal_chain),
            "total_latency": f"{sum(link.latency for link in causal_chain):.2f}s"
        }
    
    def _format_symbolic_analysis(self, symbolic_info: Dict[str, Any]) -> Dict[str, Any]:
        """Format symbolic analysis for explanation."""
        return {
            "transformations": len(symbolic_info.get("transformations", [])),
            "information_preserved": f"{symbolic_info.get('information_preserved', 1.0):.1%}",
            "dominant_transformation": symbolic_info.get("chain_analysis", {}).get("dominant_type", "none"),
            "total_residue": symbolic_info.get("total_residue", {})
        }
    
    def _identify_contributing_factors(self, attributions: List[Attribution]) -> List[str]:
        """Identify key contributing factors."""
        factors = []
        
        # Group by contribution type
        type_groups = defaultdict(list)
        for attr in attributions:
            type_groups[attr.contribution_type].append(attr)
        
        # Identify significant factor types
        for contrib_type, attrs in type_groups.items():
            total_contribution = sum(a.contribution_score for a in attrs)
            if total_contribution > 0.2:  # 20% threshold
                factors.append(
                    f"{contrib_type.replace('_', ' ').title()}: "
                    f"{total_contribution:.1%} total contribution"
                )
        
        return factors
    
    def _extract_key_insights(self, attributions: List[Attribution],
                            causal_chain: List[CausalLink]) -> List[str]:
        """Extract key insights from analysis."""
        insights = []
        
        # Insight about attribution concentration
        if attributions:
            top_3_contribution = sum(a.contribution_score for a in attributions[:3])
            if top_3_contribution > 0.7:
                insights.append(
                    f"Highly concentrated influence: top 3 factors account for "
                    f"{top_3_contribution:.1%} of outcome"
                )
        
        # Insight about causal complexity
        if len(causal_chain) > 5:
            insights.append(
                f"Complex causal chain with {len(causal_chain)} steps indicates "
                f"intricate system behavior"
            )
        
        # Insight about temporal dynamics
        if causal_chain:
            total_latency = sum(link.latency for link in causal_chain)
            if total_latency > 10:
                insights.append(
                    f"Significant temporal dynamics: {total_latency:.1f}s total latency"
                )
        
        return insights
    
    def _calculate_attribution_distribution(self,
                                          attributions: List[Attribution]) -> Dict[str, float]:
        """Calculate distribution of attributions by type."""
        distribution = defaultdict(float)
        
        for attr in attributions:
            distribution[attr.contribution_type] += attr.contribution_score
        
        return dict(distribution)
    
    def _calculate_explanation_confidence(self, attributions: List[Attribution],
                                        causal_chain: List[CausalLink]) -> float:
        """Calculate overall confidence in explanation."""
        # Average attribution confidence
        attr_confidence = np.mean([a.confidence for a in attributions]) if attributions else 0.5
        
        # Causal chain strength
        causal_confidence = np.mean([l.strength for l in causal_chain]) if causal_chain else 0.5
        
        # Weight factors
        overall_confidence = attr_confidence * 0.6 + causal_confidence * 0.4
        
        return overall_confidence


class ExplanationTemplateEngine:
    """Template engine for generating natural language explanations."""
    
    def __init__(self):
        self.templates = {
            "decision": {
                "intro": "The decision to {action} was made based on the following analysis:",
                "factors": "Key factors influencing this decision were: {factors}",
                "rationale": "The primary rationale was {rationale}",
                "confidence": "This explanation has {confidence} confidence"
            },
            "behavior": {
                "intro": "The observed behavior '{behavior}' emerged from system dynamics:",
                "components": "Contributing components included: {components}",
                "pattern": "This represents a {pattern_type} pattern",
                "significance": "The behavior is {significance} to system operation"
            },
            "emergence": {
                "intro": "The emergent property '{property}' arose from complex interactions:",
                "interactions": "Key interactions involved: {interactions}",
                "threshold": "Emergence occurred when {threshold_condition}",
                "implications": "This emergence implies {implications}"
            }
        }
    
    def generate_from_template(self, template_type: str, **kwargs) -> str:
        """Generate text from template with provided values."""
        if template_type not in self.templates:
            return f"No template available for {template_type}"
        
        template = self.templates[template_type]
        result = []
        
        for key, template_str in template.items():
            if key in kwargs:
                result.append(template_str.format(**{key: kwargs[key]}))
        
        return " ".join(result)