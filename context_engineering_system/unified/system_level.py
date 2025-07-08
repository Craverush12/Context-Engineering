"""
System-Level Properties Engine
=============================

Phase 3 implementation of system-wide emergence and stability analysis:
- Cross-protocol emergence identification
- System-level attractor formation
- Coherence measurement across layers  
- Stability and resilience metrics
- Organizational emergence patterns
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..core.field import ContextField, FieldManager


class EmergenceType(Enum):
    """Types of system-level emergence."""
    CROSS_FIELD = "cross_field"
    PROTOCOL_SYNERGY = "protocol_synergy"
    HIERARCHICAL = "hierarchical"
    TEMPORAL = "temporal"
    ORGANIZATIONAL = "organizational"


class StabilityMetric(Enum):
    """Types of stability measurements."""
    COHERENCE_STABILITY = "coherence_stability"
    ATTRACTOR_STABILITY = "attractor_stability"
    RESONANCE_STABILITY = "resonance_stability"
    BOUNDARY_STABILITY = "boundary_stability"


@dataclass
class SystemEmergenceEvent:
    """System-level emergence event."""
    event_id: str
    emergence_type: EmergenceType
    strength: float
    participating_components: List[str]
    emergence_indicators: Dict[str, Any]
    confidence: float
    persistence_duration: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemAttractor:
    """System-level attractor spanning multiple components."""
    attractor_id: str
    name: str
    component_attractors: List[str]  # References to component attractors
    system_center: Dict[str, Any]  # Multi-dimensional center
    strength: float
    influence_radius: float
    formation_time: float = field(default_factory=time.time)
    stability_score: float = 0.8


@dataclass
class CoherenceMetrics:
    """System-wide coherence measurements."""
    overall_coherence: float
    field_coherences: Dict[str, float]
    cross_field_coherence: float
    protocol_coherence: float
    temporal_coherence: float
    stability_index: float
    measurement_timestamp: float = field(default_factory=time.time)


class EmergenceIdentifier:
    """Identifies system-level emergent patterns and behaviors."""
    
    def __init__(self):
        self.emergence_history: List[SystemEmergenceEvent] = []
        self.active_emergences: Dict[str, SystemEmergenceEvent] = {}
        self.logger = logging.getLogger(__name__)
    
    def identify_system_emergence(self,
                                field_manager: FieldManager,
                                protocol_execution_data: Dict[str, Any] = None,
                                sensitivity: float = 0.6) -> List[SystemEmergenceEvent]:
        """Identify emergent patterns at the system level."""
        
        events = []
        protocol_data = protocol_execution_data or {}
        
        # Cross-field emergence
        events.extend(self._detect_cross_field_emergence(field_manager, sensitivity))
        
        # Protocol synergy emergence
        events.extend(self._detect_protocol_synergy(protocol_data, sensitivity))
        
        # Hierarchical emergence
        events.extend(self._detect_hierarchical_emergence(field_manager, sensitivity))
        
        # Temporal emergence patterns
        events.extend(self._detect_temporal_emergence(field_manager, sensitivity))
        
        # Update active emergences
        self._update_active_emergences(events)
        
        # Record in history
        self.emergence_history.extend(events)
        
        return events
    
    def _detect_cross_field_emergence(self, field_manager: FieldManager, sensitivity: float) -> List[SystemEmergenceEvent]:
        """Detect emergence patterns across multiple fields."""
        events = []
        fields = list(field_manager.active_fields.values())
        
        if len(fields) < 2:
            return events
        
        # Calculate cross-field coherence
        field_coherences = [field.measure_field_coherence() for field in fields]
        avg_coherence = np.mean(field_coherences)
        coherence_variance = np.var(field_coherences)
        
        # Low variance + high average = system coherence emergence
        if avg_coherence > sensitivity and coherence_variance < 0.1:
            event = SystemEmergenceEvent(
                event_id=f"cross_field_coherence_{int(time.time() * 1000)}",
                emergence_type=EmergenceType.CROSS_FIELD,
                strength=avg_coherence * (1 - coherence_variance),
                participating_components=list(field_manager.active_fields.keys()),
                emergence_indicators={
                    "average_coherence": avg_coherence,
                    "coherence_variance": coherence_variance,
                    "synchronized_fields": len(fields)
                },
                confidence=0.85
            )
            events.append(event)
        
        # Cross-field attractor alignment
        attractor_alignments = self._calculate_attractor_alignments(fields)
        if attractor_alignments["alignment_strength"] > sensitivity:
            event = SystemEmergenceEvent(
                event_id=f"attractor_alignment_{int(time.time() * 1000)}",
                emergence_type=EmergenceType.CROSS_FIELD,
                strength=attractor_alignments["alignment_strength"],
                participating_components=attractor_alignments["participating_fields"],
                emergence_indicators=attractor_alignments,
                confidence=0.8
            )
            events.append(event)
        
        return events
    
    def _detect_protocol_synergy(self, protocol_data: Dict[str, Any], sensitivity: float) -> List[SystemEmergenceEvent]:
        """Detect synergistic emergence between protocols."""
        events = []
        
        if not protocol_data or "execution_results" not in protocol_data:
            return events
        
        execution_results = protocol_data["execution_results"]
        
        # Look for performance amplification when protocols run together
        performance_metrics = []
        for result in execution_results:
            if "performance_improvement" in result:
                performance_metrics.append(result["performance_improvement"])
        
        if performance_metrics:
            avg_improvement = np.mean(performance_metrics)
            if avg_improvement > sensitivity:
                event = SystemEmergenceEvent(
                    event_id=f"protocol_synergy_{int(time.time() * 1000)}",
                    emergence_type=EmergenceType.PROTOCOL_SYNERGY,
                    strength=avg_improvement,
                    participating_components=[r.get("protocol_id", "unknown") for r in execution_results],
                    emergence_indicators={
                        "average_improvement": avg_improvement,
                        "protocol_count": len(execution_results),
                        "synergy_type": "performance_amplification"
                    },
                    confidence=0.75
                )
                events.append(event)
        
        return events
    
    def _detect_hierarchical_emergence(self, field_manager: FieldManager, sensitivity: float) -> List[SystemEmergenceEvent]:
        """Detect hierarchical emergence patterns."""
        events = []
        
        # Analyze field hierarchy and meta-patterns
        fields = list(field_manager.active_fields.values())
        
        # Look for higher-order patterns
        meta_patterns = self._identify_meta_patterns(fields)
        
        for pattern in meta_patterns:
            if pattern["strength"] > sensitivity:
                event = SystemEmergenceEvent(
                    event_id=f"hierarchical_{pattern['type']}_{int(time.time() * 1000)}",
                    emergence_type=EmergenceType.HIERARCHICAL,
                    strength=pattern["strength"],
                    participating_components=pattern["participating_fields"],
                    emergence_indicators=pattern,
                    confidence=0.7
                )
                events.append(event)
        
        return events
    
    def _detect_temporal_emergence(self, field_manager: FieldManager, sensitivity: float) -> List[SystemEmergenceEvent]:
        """Detect temporal emergence patterns."""
        events = []
        
        # Analyze evolution patterns across field history
        for field_id, field_history in field_manager.field_history.items():
            if len(field_history) >= 3:  # Need enough history
                temporal_pattern = self._analyze_temporal_evolution(field_history)
                
                if temporal_pattern["emergence_strength"] > sensitivity:
                    event = SystemEmergenceEvent(
                        event_id=f"temporal_{field_id}_{int(time.time() * 1000)}",
                        emergence_type=EmergenceType.TEMPORAL,
                        strength=temporal_pattern["emergence_strength"],
                        participating_components=[field_id],
                        emergence_indicators=temporal_pattern,
                        confidence=0.75
                    )
                    events.append(event)
        
        return events
    
    def _calculate_attractor_alignments(self, fields: List[ContextField]) -> Dict[str, Any]:
        """Calculate alignment between attractors across fields."""
        alignments = []
        participating_fields = []
        
        for i, field_a in enumerate(fields):
            for j, field_b in enumerate(fields[i+1:], i+1):
                alignment = self._calculate_field_attractor_alignment(field_a, field_b)
                if alignment > 0.5:
                    alignments.append(alignment)
                    participating_fields.extend([f"field_{i}", f"field_{j}"])
        
        return {
            "alignment_strength": np.mean(alignments) if alignments else 0.0,
            "alignment_count": len(alignments),
            "participating_fields": list(set(participating_fields))
        }
    
    def _calculate_field_attractor_alignment(self, field_a: ContextField, field_b: ContextField) -> float:
        """Calculate attractor alignment between two fields."""
        attractors_a = list(field_a.attractors.values())
        attractors_b = list(field_b.attractors.values())
        
        if not attractors_a or not attractors_b:
            return 0.0
        
        # Simple alignment based on position and strength correlation
        alignments = []
        for attr_a in attractors_a:
            for attr_b in attractors_b:
                # Position alignment (assuming normalized coordinates)
                pos_distance = np.sqrt(sum((a - b)**2 for a, b in zip(attr_a.center, attr_b.center)))
                pos_alignment = max(0, 1 - pos_distance)
                
                # Strength alignment
                strength_diff = abs(attr_a.strength - attr_b.strength)
                strength_alignment = max(0, 1 - strength_diff)
                
                alignment = (pos_alignment + strength_alignment) / 2
                alignments.append(alignment)
        
        return max(alignments) if alignments else 0.0
    
    def _identify_meta_patterns(self, fields: List[ContextField]) -> List[Dict[str, Any]]:
        """Identify meta-patterns across fields."""
        patterns = []
        
        # Pattern 1: Distributed attractor networks
        total_attractors = sum(len(field.attractors) for field in fields)
        if total_attractors > 5:
            network_strength = min(1.0, total_attractors / 20)  # Normalize
            patterns.append({
                "type": "distributed_attractor_network",
                "strength": network_strength,
                "participating_fields": [f"field_{i}" for i in range(len(fields))],
                "attractor_count": total_attractors
            })
        
        # Pattern 2: Resonance cascades
        cascade_strength = self._calculate_resonance_cascade_strength(fields)
        if cascade_strength > 0.3:
            patterns.append({
                "type": "resonance_cascade",
                "strength": cascade_strength,
                "participating_fields": [f"field_{i}" for i in range(len(fields))],
                "cascade_indicators": {"strength": cascade_strength}
            })
        
        return patterns
    
    def _calculate_resonance_cascade_strength(self, fields: List[ContextField]) -> float:
        """Calculate strength of resonance cascades across fields."""
        total_resonance = sum(len(field.resonance_patterns) for field in fields)
        if total_resonance == 0:
            return 0.0
        
        # Average resonance coherence across all fields
        coherences = []
        for field in fields:
            for pattern in field.resonance_patterns.values():
                coherences.append(pattern.coherence_score)
        
        return np.mean(coherences) if coherences else 0.0
    
    def _analyze_temporal_evolution(self, field_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal evolution patterns in field history."""
        if len(field_history) < 3:
            return {"emergence_strength": 0.0}
        
        # Track coherence evolution
        coherences = []
        for state in field_history:
            field_props = state.get("field_properties", {})
            coherence = field_props.get("coherence", 0.0)
            coherences.append(coherence)
        
        # Calculate trend strength
        if len(coherences) >= 3:
            # Simple trend analysis
            x = np.arange(len(coherences))
            slope, _ = np.polyfit(x, coherences, 1)
            
            # Positive trend indicates temporal emergence
            trend_strength = max(0, slope * 10)  # Scale for interpretation
            
            return {
                "emergence_strength": min(1.0, trend_strength),
                "coherence_trend": slope,
                "history_length": len(field_history),
                "final_coherence": coherences[-1]
            }
        
        return {"emergence_strength": 0.0}
    
    def _update_active_emergences(self, new_events: List[SystemEmergenceEvent]):
        """Update tracking of active emergence events."""
        current_time = time.time()
        
        # Remove expired emergences (older than 30 seconds)
        expired_events = []
        for event_id, event in self.active_emergences.items():
            if current_time - event.timestamp > 30.0:
                expired_events.append(event_id)
        
        for event_id in expired_events:
            del self.active_emergences[event_id]
        
        # Add new events
        for event in new_events:
            self.active_emergences[event.event_id] = event


class SystemAttractorFormation:
    """Manages formation and evolution of system-level attractors."""
    
    def __init__(self):
        self.system_attractors: Dict[str, SystemAttractor] = {}
        self.formation_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def detect_system_attractors(self,
                               field_manager: FieldManager,
                               emergence_events: List[SystemEmergenceEvent]) -> List[SystemAttractor]:
        """Detect and form system-level attractors."""
        
        new_attractors = []
        
        # Analyze cross-field attractor clusters
        cross_field_attractors = self._detect_cross_field_attractors(field_manager)
        new_attractors.extend(cross_field_attractors)
        
        # Form attractors from emergence events
        emergence_attractors = self._form_attractors_from_emergence(emergence_events, field_manager)
        new_attractors.extend(emergence_attractors)
        
        # Update system attractor registry
        for attractor in new_attractors:
            self.system_attractors[attractor.attractor_id] = attractor
        
        # Record formation
        if new_attractors:
            formation_record = {
                "timestamp": time.time(),
                "new_attractors": len(new_attractors),
                "total_system_attractors": len(self.system_attractors),
                "attractor_ids": [a.attractor_id for a in new_attractors]
            }
            self.formation_history.append(formation_record)
        
        return new_attractors
    
    def _detect_cross_field_attractors(self, field_manager: FieldManager) -> List[SystemAttractor]:
        """Detect attractors that span multiple fields."""
        system_attractors = []
        fields = list(field_manager.active_fields.items())
        
        if len(fields) < 2:
            return system_attractors
        
        # Find attractor clusters across fields
        for i, (field_id_a, field_a) in enumerate(fields):
            for j, (field_id_b, field_b) in enumerate(fields[i+1:], i+1):
                cross_attractor = self._find_cross_field_attractor(
                    field_id_a, field_a, field_id_b, field_b
                )
                if cross_attractor:
                    system_attractors.append(cross_attractor)
        
        return system_attractors
    
    def _find_cross_field_attractor(self, field_id_a: str, field_a: ContextField,
                                  field_id_b: str, field_b: ContextField) -> Optional[SystemAttractor]:
        """Find attractor spanning two specific fields."""
        
        # Look for aligned attractors
        for attr_a in field_a.attractors.values():
            for attr_b in field_b.attractors.values():
                # Check alignment criteria
                pos_distance = np.sqrt(sum((a - b)**2 for a, b in zip(attr_a.center, attr_b.center)))
                strength_similarity = 1 - abs(attr_a.strength - attr_b.strength)
                
                if pos_distance < 0.2 and strength_similarity > 0.7:
                    # Form system attractor
                    system_attractor = SystemAttractor(
                        attractor_id=f"system_{attr_a.id}_{attr_b.id}",
                        name=f"Cross-Field Attractor {attr_a.name}-{attr_b.name}",
                        component_attractors=[f"{field_id_a}:{attr_a.id}", f"{field_id_b}:{attr_b.id}"],
                        system_center={
                            "position": tuple((a + b) / 2 for a, b in zip(attr_a.center, attr_b.center)),
                            "field_weights": {field_id_a: 0.5, field_id_b: 0.5}
                        },
                        strength=(attr_a.strength + attr_b.strength) / 2,
                        influence_radius=max(attr_a.radius, attr_b.radius) * 1.5
                    )
                    return system_attractor
        
        return None
    
    def _form_attractors_from_emergence(self, events: List[SystemEmergenceEvent],
                                       field_manager: FieldManager) -> List[SystemAttractor]:
        """Form system attractors from emergence events."""
        attractors = []
        
        for event in events:
            if event.strength > 0.8 and len(event.participating_components) > 1:
                # Strong emergence with multiple components
                attractor = SystemAttractor(
                    attractor_id=f"emergence_{event.event_id}",
                    name=f"Emergent Attractor {event.emergence_type.value}",
                    component_attractors=event.participating_components,
                    system_center={
                        "emergence_type": event.emergence_type.value,
                        "emergence_indicators": event.emergence_indicators
                    },
                    strength=event.strength,
                    influence_radius=0.3 * event.strength
                )
                attractors.append(attractor)
        
        return attractors


class CoherenceMeasurement:
    """Measures and tracks system-wide coherence across all components."""
    
    def __init__(self):
        self.measurement_history: List[CoherenceMetrics] = []
        self.logger = logging.getLogger(__name__)
    
    def measure_system_coherence(self,
                                field_manager: FieldManager,
                                protocol_data: Dict[str, Any] = None) -> CoherenceMetrics:
        """Measure comprehensive system coherence."""
        
        # Field-level coherences
        field_coherences = {}
        for field_id, field in field_manager.active_fields.items():
            field_coherences[field_id] = field.measure_field_coherence()
        
        # Overall field coherence
        overall_coherence = np.mean(list(field_coherences.values())) if field_coherences else 0.0
        
        # Cross-field coherence
        cross_field_coherence = self._calculate_cross_field_coherence(field_manager)
        
        # Protocol coherence
        protocol_coherence = self._calculate_protocol_coherence(protocol_data or {})
        
        # Temporal coherence
        temporal_coherence = self._calculate_temporal_coherence(field_manager)
        
        # Stability index
        stability_index = self._calculate_stability_index(
            overall_coherence, cross_field_coherence, temporal_coherence
        )
        
        metrics = CoherenceMetrics(
            overall_coherence=overall_coherence,
            field_coherences=field_coherences,
            cross_field_coherence=cross_field_coherence,
            protocol_coherence=protocol_coherence,
            temporal_coherence=temporal_coherence,
            stability_index=stability_index
        )
        
        self.measurement_history.append(metrics)
        return metrics
    
    def _calculate_cross_field_coherence(self, field_manager: FieldManager) -> float:
        """Calculate coherence between fields."""
        fields = list(field_manager.active_fields.values())
        
        if len(fields) < 2:
            return 1.0  # Single field is perfectly coherent with itself
        
        coherences = [field.measure_field_coherence() for field in fields]
        
        # Cross-field coherence is inverse of variance
        coherence_variance = np.var(coherences)
        return max(0.0, 1.0 - coherence_variance)
    
    def _calculate_protocol_coherence(self, protocol_data: Dict[str, Any]) -> float:
        """Calculate coherence in protocol execution."""
        if not protocol_data or "execution_results" not in protocol_data:
            return 0.5  # Neutral when no data
        
        execution_results = protocol_data["execution_results"]
        success_rates = []
        
        for result in execution_results:
            success_rate = result.get("success_rate", 0.5)
            success_rates.append(success_rate)
        
        return np.mean(success_rates) if success_rates else 0.5
    
    def _calculate_temporal_coherence(self, field_manager: FieldManager) -> float:
        """Calculate temporal coherence across field evolution."""
        temporal_coherences = []
        
        for field_id, field_history in field_manager.field_history.items():
            if len(field_history) >= 2:
                # Calculate coherence stability over time
                coherences = []
                for state in field_history[-5:]:  # Last 5 states
                    field_props = state.get("field_properties", {})
                    coherence = field_props.get("coherence", 0.0)
                    coherences.append(coherence)
                
                if coherences:
                    # Temporal coherence is inverse of variance
                    variance = np.var(coherences)
                    temporal_coherence = max(0.0, 1.0 - variance)
                    temporal_coherences.append(temporal_coherence)
        
        return np.mean(temporal_coherences) if temporal_coherences else 0.5
    
    def _calculate_stability_index(self, overall: float, cross_field: float, temporal: float) -> float:
        """Calculate overall stability index."""
        return (overall * 0.4 + cross_field * 0.3 + temporal * 0.3)


class StabilityAnalyzer:
    """Analyzes system stability and resilience properties."""
    
    def __init__(self):
        self.stability_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def analyze_system_stability(self,
                                field_manager: FieldManager,
                                coherence_metrics: CoherenceMetrics,
                                time_window: float = 30.0) -> Dict[str, Any]:
        """Analyze comprehensive system stability."""
        
        current_time = time.time()
        
        stability_analysis = {
            "analysis_timestamp": current_time,
            "coherence_stability": self._analyze_coherence_stability(coherence_metrics),
            "attractor_stability": self._analyze_attractor_stability(field_manager),
            "resonance_stability": self._analyze_resonance_stability(field_manager),
            "boundary_stability": self._analyze_boundary_stability(field_manager)
        }
        
        # Overall stability score
        stability_scores = [
            stability_analysis["coherence_stability"]["stability_score"],
            stability_analysis["attractor_stability"]["stability_score"],
            stability_analysis["resonance_stability"]["stability_score"],
            stability_analysis["boundary_stability"]["stability_score"]
        ]
        
        stability_analysis["overall_stability"] = np.mean(stability_scores)
        stability_analysis["stability_variance"] = np.var(stability_scores)
        
        self.stability_history.append(stability_analysis)
        return stability_analysis
    
    def _analyze_coherence_stability(self, metrics: CoherenceMetrics) -> Dict[str, Any]:
        """Analyze stability of coherence measures."""
        # Simple analysis based on current metrics
        stability_indicators = {
            "overall_coherence": metrics.overall_coherence,
            "cross_field_alignment": metrics.cross_field_coherence,
            "temporal_consistency": metrics.temporal_coherence
        }
        
        stability_score = (
            metrics.overall_coherence * 0.4 +
            metrics.cross_field_coherence * 0.3 +
            metrics.temporal_coherence * 0.3
        )
        
        return {
            "stability_score": stability_score,
            "indicators": stability_indicators,
            "assessment": "stable" if stability_score > 0.7 else "moderate" if stability_score > 0.4 else "unstable"
        }
    
    def _analyze_attractor_stability(self, field_manager: FieldManager) -> Dict[str, Any]:
        """Analyze stability of attractor configurations."""
        total_attractors = 0
        stable_attractors = 0
        avg_strength = 0.0
        
        for field in field_manager.active_fields.values():
            for attractor in field.attractors.values():
                total_attractors += 1
                avg_strength += attractor.strength
                
                # Consider attractor stable if strength > 0.5 and persistence > 0.7
                if attractor.strength > 0.5 and attractor.persistence_factor > 0.7:
                    stable_attractors += 1
        
        stability_ratio = stable_attractors / total_attractors if total_attractors > 0 else 0.0
        avg_strength = avg_strength / total_attractors if total_attractors > 0 else 0.0
        
        return {
            "stability_score": (stability_ratio + avg_strength) / 2,
            "indicators": {
                "total_attractors": total_attractors,
                "stable_attractors": stable_attractors,
                "stability_ratio": stability_ratio,
                "average_strength": avg_strength
            },
            "assessment": "stable" if stability_ratio > 0.7 else "moderate" if stability_ratio > 0.4 else "unstable"
        }
    
    def _analyze_resonance_stability(self, field_manager: FieldManager) -> Dict[str, Any]:
        """Analyze stability of resonance patterns."""
        total_patterns = 0
        stable_patterns = 0
        avg_coherence = 0.0
        
        for field in field_manager.active_fields.values():
            for pattern in field.resonance_patterns.values():
                total_patterns += 1
                avg_coherence += pattern.coherence_score
                
                # Consider pattern stable if coherence > 0.6 and amplitude > 0.3
                if pattern.coherence_score > 0.6 and pattern.amplitude > 0.3:
                    stable_patterns += 1
        
        stability_ratio = stable_patterns / total_patterns if total_patterns > 0 else 0.0
        avg_coherence = avg_coherence / total_patterns if total_patterns > 0 else 0.0
        
        return {
            "stability_score": (stability_ratio + avg_coherence) / 2,
            "indicators": {
                "total_patterns": total_patterns,
                "stable_patterns": stable_patterns,
                "stability_ratio": stability_ratio,
                "average_coherence": avg_coherence
            },
            "assessment": "stable" if stability_ratio > 0.7 else "moderate" if stability_ratio > 0.4 else "unstable"
        }
    
    def _analyze_boundary_stability(self, field_manager: FieldManager) -> Dict[str, Any]:
        """Analyze stability of field boundaries."""
        boundary_permeabilities = []
        
        for field in field_manager.active_fields.values():
            boundary_permeabilities.append(field.boundary_permeability)
        
        if boundary_permeabilities:
            avg_permeability = np.mean(boundary_permeabilities)
            permeability_variance = np.var(boundary_permeabilities)
            
            # Stable boundaries have consistent permeability
            stability_score = max(0.0, 1.0 - permeability_variance)
        else:
            avg_permeability = 0.5
            stability_score = 0.5
        
        return {
            "stability_score": stability_score,
            "indicators": {
                "average_permeability": avg_permeability,
                "permeability_variance": permeability_variance if boundary_permeabilities else 0.0,
                "field_count": len(boundary_permeabilities)
            },
            "assessment": "stable" if stability_score > 0.7 else "moderate" if stability_score > 0.4 else "unstable"
        }


class ResilienceMetrics:
    """Calculates and tracks system resilience properties."""
    
    def __init__(self):
        self.resilience_history: List[Dict[str, Any]] = []
        self.disturbance_responses: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def calculate_resilience_metrics(self,
                                   field_manager: FieldManager,
                                   stability_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive resilience metrics."""
        
        current_time = time.time()
        
        resilience_metrics = {
            "measurement_timestamp": current_time,
            "adaptive_capacity": self._calculate_adaptive_capacity(field_manager),
            "recovery_potential": self._calculate_recovery_potential(stability_analysis),
            "redundancy_factor": self._calculate_redundancy_factor(field_manager),
            "response_flexibility": self._calculate_response_flexibility()
        }
        
        # Overall resilience score
        component_scores = [
            resilience_metrics["adaptive_capacity"],
            resilience_metrics["recovery_potential"], 
            resilience_metrics["redundancy_factor"],
            resilience_metrics["response_flexibility"]
        ]
        
        resilience_metrics["overall_resilience"] = np.mean(component_scores)
        resilience_metrics["resilience_confidence"] = 1.0 - np.var(component_scores)
        
        self.resilience_history.append(resilience_metrics)
        return resilience_metrics
    
    def _calculate_adaptive_capacity(self, field_manager: FieldManager) -> float:
        """Calculate system's capacity to adapt to changes."""
        # Based on field diversity and boundary flexibility
        field_count = len(field_manager.active_fields)
        avg_permeability = np.mean([
            field.boundary_permeability 
            for field in field_manager.active_fields.values()
        ]) if field_count > 0 else 0.5
        
        # More fields and higher permeability indicate higher adaptive capacity
        field_diversity = min(1.0, field_count / 5.0)  # Normalize to 5 fields max
        
        return (field_diversity + avg_permeability) / 2
    
    def _calculate_recovery_potential(self, stability_analysis: Dict[str, Any]) -> float:
        """Calculate potential for recovery from disturbances."""
        # Based on current stability - more stable systems recover better
        overall_stability = stability_analysis.get("overall_stability", 0.5)
        
        # Also consider stability variance - lower variance indicates more robust recovery
        stability_variance = stability_analysis.get("stability_variance", 0.3)
        variance_factor = max(0.0, 1.0 - stability_variance)
        
        return (overall_stability + variance_factor) / 2
    
    def _calculate_redundancy_factor(self, field_manager: FieldManager) -> float:
        """Calculate redundancy in system components."""
        total_redundancy = 0.0
        component_count = 0
        
        for field in field_manager.active_fields.values():
            # Attractor redundancy
            if len(field.attractors) > 1:
                attractor_redundancy = min(1.0, len(field.attractors) / 5.0)
                total_redundancy += attractor_redundancy
                component_count += 1
            
            # Resonance pattern redundancy
            if len(field.resonance_patterns) > 1:
                pattern_redundancy = min(1.0, len(field.resonance_patterns) / 10.0)
                total_redundancy += pattern_redundancy
                component_count += 1
        
        return total_redundancy / component_count if component_count > 0 else 0.5
    
    def _calculate_response_flexibility(self) -> float:
        """Calculate flexibility in system responses."""
        # Based on historical response diversity
        if len(self.disturbance_responses) < 2:
            return 0.5  # Default for insufficient data
        
        # Measure diversity in response types
        response_types = set()
        for response in self.disturbance_responses[-10:]:  # Last 10 responses
            response_type = response.get("response_type", "default")
            response_types.add(response_type)
        
        flexibility = min(1.0, len(response_types) / 5.0)  # Normalize to 5 types max
        return flexibility
    
    def record_disturbance_response(self, disturbance_type: str, response_data: Dict[str, Any]):
        """Record a system response to disturbance for resilience analysis."""
        response_record = {
            "timestamp": time.time(),
            "disturbance_type": disturbance_type,
            "response_type": response_data.get("response_type", "adaptive"),
            "response_effectiveness": response_data.get("effectiveness", 0.5),
            "recovery_time": response_data.get("recovery_time", 0.0),
            "full_response_data": response_data
        }
        
        self.disturbance_responses.append(response_record)


class SystemLevelPropertiesEngine:
    """Main engine coordinating all system-level analysis and properties."""
    
    def __init__(self, field_manager: FieldManager):
        self.field_manager = field_manager
        self.emergence_identifier = EmergenceIdentifier()
        self.system_attractor_formation = SystemAttractorFormation()
        self.coherence_measurement = CoherenceMeasurement()
        self.stability_analyzer = StabilityAnalyzer()
        self.resilience_metrics = ResilienceMetrics()
        
        self.analysis_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def execute_comprehensive_system_analysis(self,
                                            protocol_execution_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute comprehensive system-level analysis."""
        
        analysis_id = f"system_analysis_{int(time.time() * 1000)}"
        protocol_data = protocol_execution_data or {}
        
        # Emergence identification
        emergence_events = self.emergence_identifier.identify_system_emergence(
            self.field_manager, protocol_data
        )
        
        # System attractor formation
        system_attractors = self.system_attractor_formation.detect_system_attractors(
            self.field_manager, emergence_events
        )
        
        # Coherence measurement
        coherence_metrics = self.coherence_measurement.measure_system_coherence(
            self.field_manager, protocol_data
        )
        
        # Stability analysis
        stability_analysis = self.stability_analyzer.analyze_system_stability(
            self.field_manager, coherence_metrics
        )
        
        # Resilience metrics
        resilience_metrics = self.resilience_metrics.calculate_resilience_metrics(
            self.field_manager, stability_analysis
        )
        
        # Compile comprehensive analysis
        analysis = {
            "analysis_id": analysis_id,
            "timestamp": time.time(),
            "emergence_events": [
                {
                    "event_id": e.event_id,
                    "type": e.emergence_type.value,
                    "strength": e.strength,
                    "confidence": e.confidence
                } for e in emergence_events
            ],
            "system_attractors": [
                {
                    "attractor_id": a.attractor_id,
                    "name": a.name,
                    "strength": a.strength,
                    "component_count": len(a.component_attractors)
                } for a in system_attractors
            ],
            "coherence_metrics": {
                "overall_coherence": coherence_metrics.overall_coherence,
                "cross_field_coherence": coherence_metrics.cross_field_coherence,
                "stability_index": coherence_metrics.stability_index
            },
            "stability_analysis": {
                "overall_stability": stability_analysis["overall_stability"],
                "coherence_stability": stability_analysis["coherence_stability"]["stability_score"],
                "attractor_stability": stability_analysis["attractor_stability"]["stability_score"]
            },
            "resilience_metrics": {
                "overall_resilience": resilience_metrics["overall_resilience"],
                "adaptive_capacity": resilience_metrics["adaptive_capacity"],
                "recovery_potential": resilience_metrics["recovery_potential"]
            }
        }
        
        self.analysis_history.append(analysis)
        return analysis
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get a summary of overall system health."""
        if not self.analysis_history:
            return {"status": "no_data", "message": "No analysis data available"}
        
        latest_analysis = self.analysis_history[-1]
        
        # Overall health score
        coherence_score = latest_analysis["coherence_metrics"]["overall_coherence"]
        stability_score = latest_analysis["stability_analysis"]["overall_stability"]
        resilience_score = latest_analysis["resilience_metrics"]["overall_resilience"]
        
        health_score = (coherence_score + stability_score + resilience_score) / 3
        
        # Health assessment
        if health_score > 0.8:
            health_status = "excellent"
        elif health_score > 0.6:
            health_status = "good"
        elif health_score > 0.4:
            health_status = "fair"
        else:
            health_status = "poor"
        
        return {
            "health_score": health_score,
            "health_status": health_status,
            "coherence": coherence_score,
            "stability": stability_score,
            "resilience": resilience_score,
            "emergence_events": len(latest_analysis["emergence_events"]),
            "system_attractors": len(latest_analysis["system_attractors"]),
            "last_analysis": latest_analysis["timestamp"]
        }