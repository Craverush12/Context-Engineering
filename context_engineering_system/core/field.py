"""
Neural Field Implementation
==========================

Core implementation of continuous semantic fields with attractors, resonance,
persistence, and boundary dynamics. This represents the fundamental paradigm
shift from discrete token management to continuous field operations.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import math


class FieldElementType(Enum):
    """Types of elements that can exist in a context field."""
    CONTENT = "content"
    PATTERN = "pattern"
    ATTRACTOR = "attractor"
    PATHWAY = "pathway"
    RESIDUE = "residue"


@dataclass
class FieldElement:
    """An element within the context field."""
    id: str
    element_type: FieldElementType
    content: str
    position: Tuple[float, float] = (0.0, 0.0)
    strength: float = 1.0
    last_update: float = field(default_factory=time.time)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Attractor:
    """A stable semantic configuration in the field."""
    id: str
    name: str
    center: Tuple[float, float]
    strength: float
    radius: float
    elements: List[str] = field(default_factory=list)
    formation_time: float = field(default_factory=time.time)
    persistence_factor: float = 0.8
    resonance_patterns: List[str] = field(default_factory=list)


@dataclass
class ResonancePattern:
    """A pattern of mutual reinforcement between field elements."""
    id: str
    participating_elements: List[str]
    resonance_frequency: float
    amplitude: float
    coherence_score: float
    last_measured: float = field(default_factory=time.time)


class ContextField:
    """
    Continuous semantic field with attractors, resonance, and persistence.
    
    This represents the fundamental paradigm shift from discrete token management
    to continuous field operations with emergent properties.
    """
    
    def __init__(self, 
                 dimensions: int = 2,
                 decay_rate: float = 0.05,
                 boundary_permeability: float = 0.8,
                 attractor_threshold: float = 0.7,
                 resonance_bandwidth: float = 0.6):
        """
        Initialize a context field.
        
        Args:
            dimensions: Dimensionality of the field space
            decay_rate: Rate at which field elements decay over time
            boundary_permeability: How easily information crosses boundaries
            attractor_threshold: Minimum strength for attractor formation
            resonance_bandwidth: Frequency range for resonance detection
        """
        self.dimensions = dimensions
        self.decay_rate = decay_rate
        self.boundary_permeability = boundary_permeability
        self.attractor_threshold = attractor_threshold
        self.resonance_bandwidth = resonance_bandwidth
        
        # Field state
        self.elements: Dict[str, FieldElement] = {}
        self.attractors: Dict[str, Attractor] = {}
        self.resonance_patterns: Dict[str, ResonancePattern] = {}
        self.pathways: Dict[str, List[str]] = {}
        
        # Field properties
        self.creation_time = time.time()
        self.last_update = time.time()
        self.field_grid = np.zeros((100, 100))  # For visualization
        self.boundary_grid = np.ones((100, 100)) * boundary_permeability
        
        # Statistics
        self.total_injections = 0
        self.attractor_formations = 0
        self.resonance_events = 0
    
    def inject(self, 
               content: str, 
               strength: float = 1.0, 
               position: Optional[Tuple[float, float]] = None,
               element_type: FieldElementType = FieldElementType.CONTENT) -> str:
        """
        Inject content into the field with boundary filtering.
        
        Args:
            content: The semantic content to inject
            strength: Initial strength of the injection
            position: Position in field space (random if None)
            element_type: Type of field element
            
        Returns:
            ID of the created field element
        """
        # Generate position if not provided
        if position is None:
            position = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        
        # Apply boundary filtering
        boundary_strength = self._get_boundary_strength(position)
        effective_strength = strength * boundary_strength * self.boundary_permeability
        
        # Create field element
        element_id = f"{element_type.value}_{len(self.elements)}_{int(time.time() * 1000)}"
        element = FieldElement(
            id=element_id,
            element_type=element_type,
            content=content,
            position=position,
            strength=effective_strength,
            properties={"boundary_strength": boundary_strength}
        )
        
        self.elements[element_id] = element
        self.total_injections += 1
        self.last_update = time.time()
        
        # Update field grid for visualization
        self._update_field_grid(element)
        
        # Check for pattern formation and attractor creation
        self._detect_patterns()
        self._check_attractor_formation(element_id)
        self._detect_resonance()
        
        return element_id
    
    def decay(self) -> None:
        """Apply natural decay to field elements."""
        current_time = time.time()
        elements_to_remove = []
        
        for element_id, element in self.elements.items():
            # Calculate time-based decay
            time_since_update = current_time - element.last_update
            decay_factor = math.exp(-self.decay_rate * time_since_update)
            
            # Apply decay
            element.strength *= decay_factor
            
            # Remove very weak elements
            if element.strength < 0.01:
                elements_to_remove.append(element_id)
        
        # Remove decayed elements
        for element_id in elements_to_remove:
            self._remove_element(element_id)
        
        # Decay attractors with slower rate
        self._decay_attractors(current_time)
        
        # Update field grid
        self._recalculate_field_grid()
        
        self.last_update = current_time
    
    def get_attractors(self) -> List[Attractor]:
        """Get all active attractors in the field."""
        return list(self.attractors.values())
    
    def get_resonance_patterns(self) -> List[ResonancePattern]:
        """Get all active resonance patterns."""
        return list(self.resonance_patterns.values())
    
    def measure_field_coherence(self) -> float:
        """Measure overall coherence of the field."""
        if not self.resonance_patterns:
            return 0.0
        
        total_coherence = sum(pattern.coherence_score for pattern in self.resonance_patterns.values())
        return total_coherence / len(self.resonance_patterns)
    
    def get_field_state(self) -> Dict[str, Any]:
        """Get comprehensive field state for protocol operations."""
        return {
            "elements": {eid: self._serialize_element(elem) for eid, elem in self.elements.items()},
            "attractors": {aid: self._serialize_attractor(attr) for aid, attr in self.attractors.items()},
            "resonance_patterns": {rid: self._serialize_resonance(res) for rid, res in self.resonance_patterns.items()},
            "pathways": self.pathways,
            "field_properties": {
                "coherence": self.measure_field_coherence(),
                "total_elements": len(self.elements),
                "total_attractors": len(self.attractors),
                "field_age": time.time() - self.creation_time,
                "last_update": self.last_update
            },
            "field_grid": self.field_grid.tolist(),
            "boundary_grid": self.boundary_grid.tolist()
        }
    
    def _get_boundary_strength(self, position: Tuple[float, float]) -> float:
        """Calculate boundary strength at given position."""
        x, y = position
        grid_x = int(x * 99)
        grid_y = int(y * 99)
        return self.boundary_grid[grid_x, grid_y]
    
    def _update_field_grid(self, element: FieldElement) -> None:
        """Update the field grid with new element."""
        x, y = element.position
        grid_x = int(x * 99)
        grid_y = int(y * 99)
        
        # Add Gaussian influence around element position
        for i in range(max(0, grid_x - 5), min(100, grid_x + 6)):
            for j in range(max(0, grid_y - 5), min(100, grid_y + 6)):
                distance = math.sqrt((i - grid_x)**2 + (j - grid_y)**2)
                influence = element.strength * math.exp(-distance**2 / 10)
                self.field_grid[i, j] += influence
    
    def _detect_patterns(self) -> None:
        """Detect emerging patterns in the field."""
        # Simple pattern detection based on spatial clustering
        positions = [elem.position for elem in self.elements.values()]
        if len(positions) < 3:
            return
        
        # TODO: Implement sophisticated pattern detection algorithms
        # For now, use simple spatial clustering
        pass
    
    def _check_attractor_formation(self, new_element_id: str) -> None:
        """Check if a new attractor should form around the new element."""
        new_element = self.elements[new_element_id]
        
        if new_element.strength < self.attractor_threshold:
            return
        
        # Find nearby elements
        nearby_elements = []
        for element_id, element in self.elements.items():
            if element_id == new_element_id:
                continue
            
            distance = self._calculate_distance(new_element.position, element.position)
            if distance < 0.1 and element.strength > 0.3:  # Within attractor radius
                nearby_elements.append(element_id)
        
        # Create attractor if sufficient nearby elements
        if len(nearby_elements) >= 2:
            attractor_id = f"attractor_{len(self.attractors)}_{int(time.time() * 1000)}"
            attractor = Attractor(
                id=attractor_id,
                name=f"Attractor_{len(self.attractors) + 1}",
                center=new_element.position,
                strength=new_element.strength,
                radius=0.1,
                elements=[new_element_id] + nearby_elements
            )
            
            self.attractors[attractor_id] = attractor
            self.attractor_formations += 1
    
    def _detect_resonance(self) -> None:
        """Detect resonance patterns between field elements."""
        current_time = time.time()
        
        # Simple resonance detection based on content similarity and proximity
        element_list = list(self.elements.values())
        
        for i, elem1 in enumerate(element_list):
            for elem2 in element_list[i+1:]:
                # Calculate content similarity (simplified)
                content_similarity = self._calculate_content_similarity(elem1.content, elem2.content)
                
                # Calculate spatial proximity
                distance = self._calculate_distance(elem1.position, elem2.position)
                proximity_score = max(0, 1 - distance * 2)  # Closer = higher score
                
                # Calculate resonance strength
                resonance_strength = content_similarity * proximity_score
                
                if resonance_strength > 0.5:  # Threshold for resonance
                    pattern_id = f"resonance_{elem1.id}_{elem2.id}"
                    if pattern_id not in self.resonance_patterns:
                        pattern = ResonancePattern(
                            id=pattern_id,
                            participating_elements=[elem1.id, elem2.id],
                            resonance_frequency=resonance_strength,
                            amplitude=min(elem1.strength, elem2.strength),
                            coherence_score=resonance_strength,
                            last_measured=current_time
                        )
                        self.resonance_patterns[pattern_id] = pattern
                        self.resonance_events += 1
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity (simplified implementation)."""
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _remove_element(self, element_id: str) -> None:
        """Remove an element and update related structures."""
        if element_id not in self.elements:
            return
        
        # Remove from elements
        del self.elements[element_id]
        
        # Remove from attractors
        attractors_to_update = []
        for attractor_id, attractor in self.attractors.items():
            if element_id in attractor.elements:
                attractor.elements.remove(element_id)
                if len(attractor.elements) < 2:  # Attractor becomes unstable
                    attractors_to_update.append(attractor_id)
        
        # Remove unstable attractors
        for attractor_id in attractors_to_update:
            del self.attractors[attractor_id]
        
        # Remove from resonance patterns
        patterns_to_remove = []
        for pattern_id, pattern in self.resonance_patterns.items():
            if element_id in pattern.participating_elements:
                patterns_to_remove.append(pattern_id)
        
        for pattern_id in patterns_to_remove:
            del self.resonance_patterns[pattern_id]
    
    def _decay_attractors(self, current_time: float) -> None:
        """Apply decay to attractors with slower rate."""
        attractors_to_remove = []
        
        for attractor_id, attractor in self.attractors.items():
            # Attractors decay slower than regular elements
            time_since_formation = current_time - attractor.formation_time
            decay_factor = math.exp(-self.decay_rate * 0.3 * time_since_formation)  # 30% of normal decay
            
            attractor.strength *= decay_factor * attractor.persistence_factor
            
            if attractor.strength < 0.1:
                attractors_to_remove.append(attractor_id)
        
        for attractor_id in attractors_to_remove:
            del self.attractors[attractor_id]
    
    def _recalculate_field_grid(self) -> None:
        """Recalculate the entire field grid."""
        self.field_grid = np.zeros((100, 100))
        
        for element in self.elements.values():
            self._update_field_grid(element)
    
    def _serialize_element(self, element: FieldElement) -> Dict[str, Any]:
        """Serialize a field element for JSON output."""
        return {
            "id": element.id,
            "type": element.element_type.value,
            "content": element.content,
            "position": element.position,
            "strength": element.strength,
            "last_update": element.last_update,
            "properties": element.properties
        }
    
    def _serialize_attractor(self, attractor: Attractor) -> Dict[str, Any]:
        """Serialize an attractor for JSON output."""
        return {
            "id": attractor.id,
            "name": attractor.name,
            "center": attractor.center,
            "strength": attractor.strength,
            "radius": attractor.radius,
            "elements": attractor.elements,
            "formation_time": attractor.formation_time,
            "persistence_factor": attractor.persistence_factor,
            "resonance_patterns": attractor.resonance_patterns
        }
    
    def _serialize_resonance(self, resonance: ResonancePattern) -> Dict[str, Any]:
        """Serialize a resonance pattern for JSON output."""
        return {
            "id": resonance.id,
            "participating_elements": resonance.participating_elements,
            "resonance_frequency": resonance.resonance_frequency,
            "amplitude": resonance.amplitude,
            "coherence_score": resonance.coherence_score,
            "last_measured": resonance.last_measured
        }


class FieldManager:
    """
    Manager for context field operations and lifecycle.
    
    Handles field creation, persistence, evolution, and cross-field operations.
    """
    
    def __init__(self):
        self.active_fields: Dict[str, ContextField] = {}
        self.field_history: Dict[str, List[Dict[str, Any]]] = {}
        self.global_attractors: Dict[str, Attractor] = {}
        
    def create_field(self, 
                     field_id: str,
                     dimensions: int = 2,
                     **field_params) -> ContextField:
        """Create a new context field."""
        field = ContextField(dimensions=dimensions, **field_params)
        self.active_fields[field_id] = field
        self.field_history[field_id] = []
        return field
    
    def get_field(self, field_id: str) -> Optional[ContextField]:
        """Get an active field by ID."""
        return self.active_fields.get(field_id)
    
    def persist_field_state(self, field_id: str) -> None:
        """Persist current field state to history."""
        if field_id in self.active_fields:
            field_state = self.active_fields[field_id].get_field_state()
            field_state["timestamp"] = time.time()
            self.field_history[field_id].append(field_state)
    
    def evolve_field(self, field_id: str, evolution_steps: int = 1) -> None:
        """Evolve a field through multiple decay/update cycles."""
        if field_id not in self.active_fields:
            return
        
        field = self.active_fields[field_id]
        for _ in range(evolution_steps):
            field.decay()
            self.persist_field_state(field_id)
    
    def merge_fields(self, field_ids: List[str], new_field_id: str) -> ContextField:
        """Merge multiple fields into a new field."""
        merged_field = self.create_field(new_field_id)
        
        for field_id in field_ids:
            if field_id in self.active_fields:
                source_field = self.active_fields[field_id]
                
                # Transfer elements
                for element in source_field.elements.values():
                    merged_field.inject(
                        content=element.content,
                        strength=element.strength,
                        position=element.position,
                        element_type=element.element_type
                    )
        
        return merged_field
    
    def get_global_field_state(self) -> Dict[str, Any]:
        """Get the state of all active fields."""
        return {
            "active_fields": {fid: field.get_field_state() for fid, field in self.active_fields.items()},
            "field_count": len(self.active_fields),
            "total_elements": sum(len(field.elements) for field in self.active_fields.values()),
            "total_attractors": sum(len(field.attractors) for field in self.active_fields.values()),
            "global_attractors": {aid: self._serialize_attractor(attr) for aid, attr in self.global_attractors.items()}
        }
    
    def _serialize_attractor(self, attractor: Attractor) -> Dict[str, Any]:
        """Serialize an attractor for JSON output."""
        return {
            "id": attractor.id,
            "name": attractor.name,
            "center": attractor.center,
            "strength": attractor.strength,
            "radius": attractor.radius,
            "elements": attractor.elements,
            "formation_time": attractor.formation_time,
            "persistence_factor": attractor.persistence_factor,
            "resonance_patterns": attractor.resonance_patterns
        }