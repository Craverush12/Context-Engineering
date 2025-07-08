"""
Unified Context Orchestrator
===========================

Phase 3 unified orchestration engine that coordinates:
- Multi-protocol integration and execution
- Advanced field operations
- System-level emergence detection and management
- Hierarchical field management
- Organizational intelligence emergence
"""

import time
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

from ..core.field import ContextField, FieldManager
from .multi_protocol import MultiProtocolOrchestrator, ProtocolComposition, ExecutionMode
from .field_operations import AdvancedFieldOperationsEngine, ScanMode, BoundaryOperation
from .system_level import SystemLevelPropertiesEngine


@dataclass
class UnifiedOperationRequest:
    """Request for unified system operation."""
    operation_id: str
    operation_type: str
    target_components: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: float = 1.0
    timeout: Optional[float] = None


@dataclass
class UnifiedOperationResult:
    """Result of unified system operation."""
    operation_id: str
    operation_type: str
    success: bool
    results: Dict[str, Any]
    execution_time: float
    system_impact: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class HierarchicalFieldManager:
    """Manages hierarchical organization of fields across organizational boundaries."""
    
    def __init__(self, base_field_manager: FieldManager):
        self.base_manager = base_field_manager
        self.field_hierarchy: Dict[str, Dict[str, Any]] = {}
        self.organizational_layers: Dict[str, List[str]] = {}
        self.cross_boundary_connections: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def create_hierarchical_structure(self,
                                    organization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create hierarchical field organization."""
        
        # Create organizational layers
        for layer_name, layer_config in organization_config.get("layers", {}).items():
            field_ids = []
            
            for field_config in layer_config.get("fields", []):
                field_id = f"{layer_name}_{field_config['name']}"
                field = self.base_manager.create_field(
                    field_id,
                    dimensions=field_config.get("dimensions", 2),
                    **field_config.get("field_params", {})
                )
                field_ids.append(field_id)
                
                # Store hierarchy information
                self.field_hierarchy[field_id] = {
                    "layer": layer_name,
                    "level": layer_config.get("level", 0),
                    "parent_layer": layer_config.get("parent_layer"),
                    "organizational_role": field_config.get("role", "worker")
                }
            
            self.organizational_layers[layer_name] = field_ids
        
        # Create cross-boundary connections
        self._establish_cross_boundary_connections(organization_config)
        
        return {
            "created_layers": len(self.organizational_layers),
            "total_fields": sum(len(fields) for fields in self.organizational_layers.values()),
            "cross_boundary_connections": len(self.cross_boundary_connections)
        }
    
    def _establish_cross_boundary_connections(self, config: Dict[str, Any]):
        """Establish connections across organizational boundaries."""
        connections = config.get("cross_boundary_connections", [])
        
        for connection in connections:
            source_layer = connection["source_layer"]
            target_layer = connection["target_layer"]
            connection_type = connection.get("type", "bidirectional")
            strength = connection.get("strength", 0.5)
            
            self.cross_boundary_connections.append({
                "source_layer": source_layer,
                "target_layer": target_layer,
                "connection_type": connection_type,
                "strength": strength,
                "established_time": time.time()
            })
    
    def get_organizational_state(self) -> Dict[str, Any]:
        """Get current state of organizational field hierarchy."""
        return {
            "hierarchy": self.field_hierarchy,
            "layers": self.organizational_layers,
            "connections": self.cross_boundary_connections,
            "field_states": {
                field_id: field.get_field_state() 
                for field_id, field in self.base_manager.active_fields.items()
            }
        }


class SystemIntelligenceEngine:
    """Manages system-wide intelligence emergence and coordination."""
    
    def __init__(self,
                 field_manager: FieldManager,
                 system_properties_engine: SystemLevelPropertiesEngine):
        self.field_manager = field_manager
        self.system_properties = system_properties_engine
        self.intelligence_patterns: List[Dict[str, Any]] = []
        self.collective_insights: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    def detect_system_intelligence(self) -> Dict[str, Any]:
        """Detect emergent intelligence patterns in the system."""
        
        # Get current system analysis
        system_analysis = self.system_properties.execute_comprehensive_system_analysis()
        
        # Analyze for intelligence indicators
        intelligence_indicators = {
            "collective_problem_solving": self._detect_collective_problem_solving(),
            "adaptive_learning": self._detect_adaptive_learning(),
            "emergent_coordination": self._detect_emergent_coordination(),
            "meta_cognitive_patterns": self._detect_meta_cognitive_patterns()
        }
        
        # Calculate overall intelligence emergence score
        intelligence_scores = [
            indicators.get("strength", 0.0) 
            for indicators in intelligence_indicators.values()
        ]
        overall_intelligence = sum(intelligence_scores) / len(intelligence_scores) if intelligence_scores else 0.0
        
        intelligence_assessment = {
            "overall_intelligence_level": overall_intelligence,
            "intelligence_indicators": intelligence_indicators,
            "system_analysis": system_analysis,
            "assessment_timestamp": time.time()
        }
        
        self.intelligence_patterns.append(intelligence_assessment)
        return intelligence_assessment
    
    def _detect_collective_problem_solving(self) -> Dict[str, Any]:
        """Detect collective problem-solving capabilities."""
        # Analyze field coordination patterns
        fields = list(self.field_manager.active_fields.values())
        
        if len(fields) < 2:
            return {"strength": 0.0, "indicators": []}
        
        # Look for complementary attractor patterns
        complementary_patterns = 0
        total_comparisons = 0
        
        for i, field_a in enumerate(fields):
            for field_b in fields[i+1:]:
                total_comparisons += 1
                
                # Check if fields have complementary attractors
                if self._are_fields_complementary(field_a, field_b):
                    complementary_patterns += 1
        
        complementarity_ratio = complementary_patterns / total_comparisons if total_comparisons > 0 else 0.0
        
        return {
            "strength": complementarity_ratio,
            "indicators": [
                f"Complementary patterns: {complementary_patterns}/{total_comparisons}",
                f"Field coordination level: {complementarity_ratio:.2f}"
            ]
        }
    
    def _detect_adaptive_learning(self) -> Dict[str, Any]:
        """Detect adaptive learning patterns."""
        # Analyze field evolution over time
        learning_indicators = []
        total_learning_strength = 0.0
        field_count = 0
        
        for field_id, field_history in self.field_manager.field_history.items():
            if len(field_history) >= 3:
                field_count += 1
                learning_trend = self._analyze_field_learning_trend(field_history)
                total_learning_strength += learning_trend
                
                if learning_trend > 0.1:
                    learning_indicators.append(f"Field {field_id}: positive learning trend")
        
        avg_learning_strength = total_learning_strength / field_count if field_count > 0 else 0.0
        
        return {
            "strength": avg_learning_strength,
            "indicators": learning_indicators
        }
    
    def _detect_emergent_coordination(self) -> Dict[str, Any]:
        """Detect emergent coordination patterns."""
        fields = list(self.field_manager.active_fields.values())
        
        # Calculate field synchronization
        field_coherences = [field.measure_field_coherence() for field in fields]
        
        if len(field_coherences) < 2:
            return {"strength": 0.0, "indicators": []}
        
        # High coordination = low variance in coherence + high average coherence
        avg_coherence = sum(field_coherences) / len(field_coherences)
        coherence_variance = sum((c - avg_coherence)**2 for c in field_coherences) / len(field_coherences)
        
        coordination_strength = avg_coherence * (1 - coherence_variance)
        
        return {
            "strength": coordination_strength,
            "indicators": [
                f"Average field coherence: {avg_coherence:.3f}",
                f"Coherence variance: {coherence_variance:.3f}",
                f"Coordination level: {coordination_strength:.3f}"
            ]
        }
    
    def _detect_meta_cognitive_patterns(self) -> Dict[str, Any]:
        """Detect meta-cognitive patterns (thinking about thinking)."""
        # Look for recursive patterns and self-reference
        meta_patterns = 0
        
        for field in self.field_manager.active_fields.values():
            # Check for self-referential attractors
            for attractor in field.attractors.values():
                if "meta" in attractor.name.lower() or "self" in attractor.name.lower():
                    meta_patterns += 1
        
        # Normalize by total number of attractors
        total_attractors = sum(len(field.attractors) for field in self.field_manager.active_fields.values())
        meta_ratio = meta_patterns / total_attractors if total_attractors > 0 else 0.0
        
        return {
            "strength": meta_ratio,
            "indicators": [
                f"Meta-cognitive attractors: {meta_patterns}",
                f"Meta-cognitive ratio: {meta_ratio:.3f}"
            ]
        }
    
    def _are_fields_complementary(self, field_a: ContextField, field_b: ContextField) -> bool:
        """Check if two fields have complementary patterns."""
        # Simplified complementarity check
        coherence_a = field_a.measure_field_coherence()
        coherence_b = field_b.measure_field_coherence()
        
        # Different coherence levels can be complementary
        coherence_difference = abs(coherence_a - coherence_b)
        
        # Different attractor counts can be complementary
        attractor_diff = abs(len(field_a.attractors) - len(field_b.attractors))
        
        return coherence_difference > 0.2 and attractor_diff > 0
    
    def _analyze_field_learning_trend(self, field_history: List[Dict[str, Any]]) -> float:
        """Analyze learning trend in field evolution."""
        if len(field_history) < 3:
            return 0.0
        
        # Extract coherence progression
        coherences = []
        for state in field_history[-5:]:  # Last 5 states
            field_props = state.get("field_properties", {})
            coherence = field_props.get("coherence", 0.0)
            coherences.append(coherence)
        
        if len(coherences) < 2:
            return 0.0
        
        # Calculate learning trend (improvement over time)
        trend = (coherences[-1] - coherences[0]) / len(coherences)
        return max(0.0, trend)  # Only positive learning


class OrganizationalEmergenceEngine:
    """Manages emergence at organizational scale."""
    
    def __init__(self, hierarchical_manager: HierarchicalFieldManager):
        self.hierarchical_manager = hierarchical_manager
        self.organizational_patterns: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def detect_organizational_emergence(self) -> Dict[str, Any]:
        """Detect emergence patterns at organizational level."""
        
        org_state = self.hierarchical_manager.get_organizational_state()
        
        emergence_patterns = {
            "cross_layer_coordination": self._analyze_cross_layer_coordination(org_state),
            "organizational_learning": self._analyze_organizational_learning(org_state),
            "adaptive_restructuring": self._analyze_adaptive_restructuring(org_state),
            "collective_intelligence": self._analyze_collective_intelligence(org_state)
        }
        
        # Overall organizational emergence
        pattern_strengths = [p.get("strength", 0.0) for p in emergence_patterns.values()]
        overall_emergence = sum(pattern_strengths) / len(pattern_strengths) if pattern_strengths else 0.0
        
        organizational_analysis = {
            "overall_emergence_level": overall_emergence,
            "emergence_patterns": emergence_patterns,
            "organizational_state": org_state,
            "analysis_timestamp": time.time()
        }
        
        self.organizational_patterns.append(organizational_analysis)
        return organizational_analysis
    
    def _analyze_cross_layer_coordination(self, org_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coordination across organizational layers."""
        layers = org_state.get("layers", {})
        connections = org_state.get("connections", [])
        
        if len(layers) < 2:
            return {"strength": 0.0, "description": "Insufficient layers for coordination"}
        
        # Calculate coordination strength based on connections
        total_possible_connections = len(layers) * (len(layers) - 1)
        actual_connections = len(connections)
        
        coordination_ratio = actual_connections / total_possible_connections if total_possible_connections > 0 else 0.0
        
        return {
            "strength": coordination_ratio,
            "description": f"Cross-layer coordination: {actual_connections}/{total_possible_connections} connections"
        }
    
    def _analyze_organizational_learning(self, org_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze learning patterns across the organization."""
        field_states = org_state.get("field_states", {})
        
        # Analyze coherence patterns across organizational levels
        layer_coherences = {}
        for field_id, field_state in field_states.items():
            hierarchy_info = self.hierarchical_manager.field_hierarchy.get(field_id, {})
            layer = hierarchy_info.get("layer", "unknown")
            
            field_coherence = field_state.get("field_properties", {}).get("coherence", 0.0)
            
            if layer not in layer_coherences:
                layer_coherences[layer] = []
            layer_coherences[layer].append(field_coherence)
        
        # Calculate learning as coherence improvement across layers
        learning_indicators = []
        for layer, coherences in layer_coherences.items():
            if coherences:
                avg_coherence = sum(coherences) / len(coherences)
                learning_indicators.append(avg_coherence)
        
        overall_learning = sum(learning_indicators) / len(learning_indicators) if learning_indicators else 0.0
        
        return {
            "strength": overall_learning,
            "description": f"Organizational learning level: {overall_learning:.3f}"
        }
    
    def _analyze_adaptive_restructuring(self, org_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze adaptive restructuring capabilities."""
        # Simple analysis based on connection flexibility
        connections = org_state.get("connections", [])
        
        if not connections:
            return {"strength": 0.0, "description": "No connections to analyze"}
        
        # Calculate connection diversity and strength
        connection_types = set(conn.get("connection_type", "default") for conn in connections)
        avg_strength = sum(conn.get("strength", 0.5) for conn in connections) / len(connections)
        
        # More connection types + higher strength = more adaptive
        adaptability = (len(connection_types) / 3.0) * avg_strength  # Normalize to 3 types max
        
        return {
            "strength": min(1.0, adaptability),
            "description": f"Adaptive restructuring: {len(connection_types)} connection types, avg strength {avg_strength:.3f}"
        }
    
    def _analyze_collective_intelligence(self, org_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze collective intelligence emergence."""
        field_states = org_state.get("field_states", {})
        
        # Collective intelligence indicators
        total_attractors = sum(
            len(state.get("attractors", {})) 
            for state in field_states.values()
        )
        
        total_resonance = sum(
            len(state.get("resonance_patterns", {}))
            for state in field_states.values()
        )
        
        # Normalize intelligence indicators
        field_count = len(field_states)
        if field_count == 0:
            return {"strength": 0.0, "description": "No fields to analyze"}
        
        attractor_density = total_attractors / field_count
        resonance_density = total_resonance / field_count
        
        collective_intelligence = (attractor_density + resonance_density) / 10.0  # Normalize
        
        return {
            "strength": min(1.0, collective_intelligence),
            "description": f"Collective intelligence: {attractor_density:.1f} attractors/field, {resonance_density:.1f} patterns/field"
        }


class UnifiedContextOrchestrator:
    """Main orchestrator for the complete Phase 3 context engineering system."""
    
    def __init__(self):
        """Initialize the unified orchestrator with all Phase 3 components."""
        # Core components
        self.field_manager = FieldManager()
        self.multi_protocol_orchestrator = MultiProtocolOrchestrator(self.field_manager)
        self.field_operations_engine = AdvancedFieldOperationsEngine(self.field_manager)
        self.system_properties_engine = SystemLevelPropertiesEngine(self.field_manager)
        
        # Hierarchical and intelligence components
        self.hierarchical_manager = HierarchicalFieldManager(self.field_manager)
        self.system_intelligence = SystemIntelligenceEngine(self.field_manager, self.system_properties_engine)
        self.organizational_emergence = OrganizationalEmergenceEngine(self.hierarchical_manager)
        
        # Operation tracking
        self.operation_history: List[UnifiedOperationResult] = []
        self.active_operations: Dict[str, UnifiedOperationRequest] = {}
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Unified Context Orchestrator initialized with all Phase 3 components")
    
    async def execute_unified_operation(self, request: UnifiedOperationRequest) -> UnifiedOperationResult:
        """Execute a unified operation across all system components."""
        
        start_time = time.time()
        self.active_operations[request.operation_id] = request
        
        try:
            if request.operation_type == "comprehensive_analysis":
                results = await self._execute_comprehensive_analysis(request)
            elif request.operation_type == "multi_protocol_execution":
                results = await self._execute_multi_protocol_operation(request)
            elif request.operation_type == "field_operations":
                results = await self._execute_field_operations(request)
            elif request.operation_type == "system_intelligence_assessment":
                results = await self._execute_intelligence_assessment(request)
            elif request.operation_type == "organizational_emergence_analysis":
                results = await self._execute_organizational_analysis(request)
            else:
                raise ValueError(f"Unknown operation type: {request.operation_type}")
            
            execution_time = time.time() - start_time
            
            # Analyze system impact
            system_impact = await self._analyze_system_impact(request, results)
            
            result = UnifiedOperationResult(
                operation_id=request.operation_id,
                operation_type=request.operation_type,
                success=True,
                results=results,
                execution_time=execution_time,
                system_impact=system_impact
            )
            
            self.logger.info(f"Unified operation {request.operation_id} completed successfully in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = UnifiedOperationResult(
                operation_id=request.operation_id,
                operation_type=request.operation_type,
                success=False,
                results={"error": str(e)},
                execution_time=execution_time,
                system_impact={"error_impact": "operation_failed"}
            )
            
            self.logger.error(f"Unified operation {request.operation_id} failed: {e}")
        
        finally:
            # Clean up
            if request.operation_id in self.active_operations:
                del self.active_operations[request.operation_id]
            
            self.operation_history.append(result)
        
        return result
    
    async def _execute_comprehensive_analysis(self, request: UnifiedOperationRequest) -> Dict[str, Any]:
        """Execute comprehensive analysis across all components."""
        
        # Field operations analysis
        field_analysis = self.field_operations_engine.execute_comprehensive_analysis(
            request.target_components, request.parameters.get("field_params", {})
        )
        
        # System-level analysis
        system_analysis = self.system_properties_engine.execute_comprehensive_system_analysis(
            request.parameters.get("protocol_data", {})
        )
        
        # Intelligence assessment
        intelligence_analysis = self.system_intelligence.detect_system_intelligence()
        
        # Organizational emergence
        org_analysis = self.organizational_emergence.detect_organizational_emergence()
        
        return {
            "field_analysis": field_analysis,
            "system_analysis": system_analysis,
            "intelligence_analysis": intelligence_analysis,
            "organizational_analysis": org_analysis,
            "analysis_summary": self._create_analysis_summary(
                field_analysis, system_analysis, intelligence_analysis, org_analysis
            )
        }
    
    async def _execute_multi_protocol_operation(self, request: UnifiedOperationRequest) -> Dict[str, Any]:
        """Execute multi-protocol operations."""
        
        protocols = request.parameters.get("protocols", [])
        execution_strategy = ExecutionMode(request.parameters.get("execution_strategy", "sequential"))
        
        # Create composition
        composition = self.multi_protocol_orchestrator.create_protocol_composition(
            request.operation_id,
            f"Unified Operation {request.operation_id}",
            execution_strategy
        )
        
        # Add protocols to composition
        for protocol_config in protocols:
            self.multi_protocol_orchestrator.add_protocol_to_composition(
                request.operation_id,
                protocol_config["name"],
                protocol_config.get("parameters", {}),
                protocol_config.get("dependencies", []),
                protocol_config.get("priority", 1.0)
            )
        
        # Get target fields
        target_fields = [
            self.field_manager.get_field(field_id) 
            for field_id in request.target_components
            if self.field_manager.get_field(field_id)
        ]
        
        # Execute composition
        execution_results = await self.multi_protocol_orchestrator.execute_composition(
            request.operation_id,
            target_fields,
            request.parameters.get("execution_context", {})
        )
        
        return {
            "composition_id": request.operation_id,
            "execution_strategy": execution_strategy.value,
            "protocols_executed": len(protocols),
            "execution_results": execution_results,
            "performance_metrics": self.multi_protocol_orchestrator.get_performance_metrics()
        }
    
    async def _execute_field_operations(self, request: UnifiedOperationRequest) -> Dict[str, Any]:
        """Execute advanced field operations."""
        
        operation_type = request.parameters.get("field_operation", "comprehensive_analysis")
        target_fields = [
            self.field_manager.get_field(field_id)
            for field_id in request.target_components
            if self.field_manager.get_field(field_id)
        ]
        
        if operation_type == "attractor_scan":
            scan_mode = ScanMode(request.parameters.get("scan_mode", "surface"))
            results = {}
            for i, field in enumerate(target_fields):
                scan_result = self.field_operations_engine.attractor_scanner.scan_attractors(
                    field, scan_mode, request.parameters.get("scan_params", {})
                )
                results[f"field_{i}"] = scan_result
        
        elif operation_type == "boundary_manipulation":
            operation = BoundaryOperation(request.parameters.get("boundary_operation", "expand"))
            results = {}
            for i, field in enumerate(target_fields):
                manipulation_result = self.field_operations_engine.boundary_manipulator.manipulate_boundary(
                    field, operation, request.parameters.get("boundary_params", {})
                )
                results[f"field_{i}"] = manipulation_result
        
        else:
            # Default comprehensive analysis
            results = self.field_operations_engine.execute_comprehensive_analysis(
                request.target_components, request.parameters
            )
        
        return results
    
    async def _execute_intelligence_assessment(self, request: UnifiedOperationRequest) -> Dict[str, Any]:
        """Execute system intelligence assessment."""
        
        intelligence_analysis = self.system_intelligence.detect_system_intelligence()
        system_health = self.system_properties_engine.get_system_health_summary()
        
        return {
            "intelligence_analysis": intelligence_analysis,
            "system_health": system_health,
            "intelligence_trends": self._analyze_intelligence_trends(),
            "recommendations": self._generate_intelligence_recommendations(intelligence_analysis)
        }
    
    async def _execute_organizational_analysis(self, request: UnifiedOperationRequest) -> Dict[str, Any]:
        """Execute organizational emergence analysis."""
        
        org_config = request.parameters.get("organization_config", {})
        
        # Create or update organizational structure if config provided
        if org_config:
            structure_result = self.hierarchical_manager.create_hierarchical_structure(org_config)
        else:
            structure_result = {"message": "Using existing organizational structure"}
        
        # Analyze organizational emergence
        emergence_analysis = self.organizational_emergence.detect_organizational_emergence()
        
        return {
            "structure_result": structure_result,
            "emergence_analysis": emergence_analysis,
            "organizational_state": self.hierarchical_manager.get_organizational_state(),
            "organizational_recommendations": self._generate_organizational_recommendations(emergence_analysis)
        }
    
    async def _analyze_system_impact(self, request: UnifiedOperationRequest, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the impact of the operation on the overall system."""
        
        # Get system state before and after (simplified)
        current_health = self.system_properties_engine.get_system_health_summary()
        
        return {
            "operation_type": request.operation_type,
            "target_components_affected": len(request.target_components),
            "system_health_after": current_health,
            "performance_impact": {
                "execution_time": time.time(),
                "resource_utilization": "normal",  # Simplified
                "system_stability": current_health.get("stability", 0.5)
            }
        }
    
    def _create_analysis_summary(self, field_analysis: Dict, system_analysis: Dict, 
                                intelligence_analysis: Dict, org_analysis: Dict) -> Dict[str, Any]:
        """Create a comprehensive analysis summary."""
        
        return {
            "overall_system_health": system_analysis.get("coherence_metrics", {}).get("overall_coherence", 0.0),
            "intelligence_level": intelligence_analysis.get("overall_intelligence_level", 0.0),
            "organizational_emergence": org_analysis.get("overall_emergence_level", 0.0),
            "field_operations_summary": {
                "total_operations": field_analysis.get("total_operations", 0),
                "emergence_events": len(field_analysis.get("results", {}).get("emergence_detection_field_0", []))
            },
            "key_insights": [
                f"System coherence: {system_analysis.get('coherence_metrics', {}).get('overall_coherence', 0.0):.3f}",
                f"Intelligence emergence: {intelligence_analysis.get('overall_intelligence_level', 0.0):.3f}",
                f"Organizational emergence: {org_analysis.get('overall_emergence_level', 0.0):.3f}"
            ]
        }
    
    def _analyze_intelligence_trends(self) -> Dict[str, Any]:
        """Analyze trends in system intelligence over time."""
        if len(self.system_intelligence.intelligence_patterns) < 2:
            return {"trend": "insufficient_data"}
        
        recent_patterns = self.system_intelligence.intelligence_patterns[-5:]
        intelligence_levels = [p["overall_intelligence_level"] for p in recent_patterns]
        
        if len(intelligence_levels) >= 2:
            trend = intelligence_levels[-1] - intelligence_levels[0]
            return {
                "trend": "increasing" if trend > 0.05 else "decreasing" if trend < -0.05 else "stable",
                "trend_magnitude": abs(trend),
                "current_level": intelligence_levels[-1],
                "change_rate": trend / len(intelligence_levels)
            }
        
        return {"trend": "stable"}
    
    def _generate_intelligence_recommendations(self, intelligence_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on intelligence analysis."""
        recommendations = []
        
        intelligence_level = intelligence_analysis.get("overall_intelligence_level", 0.0)
        
        if intelligence_level < 0.3:
            recommendations.append("Increase field interaction and cross-field resonance")
            recommendations.append("Implement more adaptive protocols")
        elif intelligence_level < 0.6:
            recommendations.append("Enhance multi-protocol coordination")
            recommendations.append("Optimize attractor configurations")
        else:
            recommendations.append("Maintain current intelligence patterns")
            recommendations.append("Explore advanced emergence opportunities")
        
        return recommendations
    
    def _generate_organizational_recommendations(self, org_analysis: Dict[str, Any]) -> List[str]:
        """Generate organizational recommendations."""
        recommendations = []
        
        emergence_level = org_analysis.get("overall_emergence_level", 0.0)
        
        if emergence_level < 0.4:
            recommendations.append("Strengthen cross-layer connections")
            recommendations.append("Implement organizational learning protocols")
        elif emergence_level < 0.7:
            recommendations.append("Enhance adaptive restructuring capabilities")
            recommendations.append("Develop collective intelligence patterns")
        else:
            recommendations.append("Leverage high organizational emergence for innovation")
            recommendations.append("Scale successful patterns to other domains")
        
        return recommendations
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive overview of the entire system."""
        
        return {
            "system_components": {
                "active_fields": len(self.field_manager.active_fields),
                "organizational_layers": len(self.hierarchical_manager.organizational_layers),
                "active_operations": len(self.active_operations),
                "total_operations_completed": len(self.operation_history)
            },
            "system_health": self.system_properties_engine.get_system_health_summary(),
            "intelligence_status": {
                "patterns_detected": len(self.system_intelligence.intelligence_patterns),
                "latest_intelligence_level": (
                    self.system_intelligence.intelligence_patterns[-1].get("overall_intelligence_level", 0.0)
                    if self.system_intelligence.intelligence_patterns else 0.0
                )
            },
            "organizational_status": {
                "emergence_patterns": len(self.organizational_emergence.organizational_patterns),
                "latest_emergence_level": (
                    self.organizational_emergence.organizational_patterns[-1].get("overall_emergence_level", 0.0)
                    if self.organizational_emergence.organizational_patterns else 0.0
                )
            },
            "performance_metrics": {
                "average_operation_time": (
                    sum(op.execution_time for op in self.operation_history) / len(self.operation_history)
                    if self.operation_history else 0.0
                ),
                "success_rate": (
                    sum(1 for op in self.operation_history if op.success) / len(self.operation_history)
                    if self.operation_history else 0.0
                )
            }
        }