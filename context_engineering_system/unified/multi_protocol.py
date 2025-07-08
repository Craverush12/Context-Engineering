"""
Multi-Protocol Integration System
=================================

Advanced protocol orchestration capabilities enabling:
- Sequential and parallel protocol composition
- Hierarchical protocol organization
- Adaptive protocol selection based on field states
- Cross-protocol resonance detection
- Complex workflow automation
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, Future

from ..core.protocol_orchestrator import ProtocolOrchestrator
from ..core.field import ContextField, FieldManager


class ExecutionMode(Enum):
    """Modes of protocol execution."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


class ProtocolStatus(Enum):
    """Status of protocol execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProtocolNode:
    """A protocol execution node in the orchestration graph."""
    protocol_id: str
    protocol_name: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    priority: float = 1.0
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    status: ProtocolStatus = ProtocolStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None


@dataclass
class ProtocolComposition:
    """A composition of multiple protocols with execution strategy."""
    composition_id: str
    name: str
    protocols: List[ProtocolNode]
    execution_strategy: ExecutionMode
    composition_parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_protocol(self, protocol: ProtocolNode):
        """Add a protocol to the composition."""
        self.protocols.append(protocol)
    
    def add_dependency(self, dependent_id: str, dependency_id: str):
        """Add a dependency between protocols."""
        for protocol in self.protocols:
            if protocol.protocol_id == dependent_id:
                if dependency_id not in protocol.dependencies:
                    protocol.dependencies.append(dependency_id)
                break
    
    def get_execution_order(self) -> List[List[str]]:
        """Get the execution order based on dependencies."""
        # Topological sort for dependency resolution
        execution_layers = []
        remaining_protocols = {p.protocol_id: p.dependencies.copy() for p in self.protocols}
        
        while remaining_protocols:
            # Find protocols with no remaining dependencies
            ready_protocols = [
                pid for pid, deps in remaining_protocols.items() 
                if not deps
            ]
            
            if not ready_protocols:
                # Circular dependency detected
                raise ValueError(f"Circular dependency detected in protocols: {list(remaining_protocols.keys())}")
            
            execution_layers.append(ready_protocols)
            
            # Remove completed protocols from dependencies
            for pid in ready_protocols:
                del remaining_protocols[pid]
            
            for deps in remaining_protocols.values():
                for completed_pid in ready_protocols:
                    if completed_pid in deps:
                        deps.remove(completed_pid)
        
        return execution_layers


@dataclass
class ProtocolChain:
    """A chain of protocol compositions for complex workflows."""
    chain_id: str
    name: str
    compositions: List[ProtocolComposition]
    chain_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def add_composition(self, composition: ProtocolComposition):
        """Add a composition to the chain."""
        self.compositions.append(composition)


class MultiProtocolOrchestrator:
    """
    Advanced multi-protocol orchestration engine.
    
    Capabilities:
    - Sequential and parallel protocol execution
    - Hierarchical protocol organization
    - Adaptive protocol selection based on field states
    - Cross-protocol resonance detection
    - Complex workflow automation
    """
    
    def __init__(self, field_manager: FieldManager):
        """Initialize multi-protocol orchestrator."""
        self.field_manager = field_manager
        self.protocol_orchestrator = ProtocolOrchestrator()
        self.adaptive_selector = AdaptiveProtocolSelector(field_manager)
        self.resonance_detector = CrossProtocolResonanceDetector()
        
        self.active_compositions: Dict[str, ProtocolComposition] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def create_protocol_composition(self,
                                  composition_id: str,
                                  name: str,
                                  execution_strategy: ExecutionMode = ExecutionMode.SEQUENTIAL) -> ProtocolComposition:
        """Create a new protocol composition."""
        composition = ProtocolComposition(
            composition_id=composition_id,
            name=name,
            protocols=[],
            execution_strategy=execution_strategy
        )
        
        self.active_compositions[composition_id] = composition
        return composition
    
    def add_protocol_to_composition(self,
                                  composition_id: str,
                                  protocol_name: str,
                                  parameters: Dict[str, Any],
                                  dependencies: Optional[List[str]] = None,
                                  priority: float = 1.0) -> str:
        """Add a protocol to an existing composition."""
        if composition_id not in self.active_compositions:
            raise ValueError(f"Composition {composition_id} not found")
        
        protocol_id = f"{composition_id}_{protocol_name}_{len(self.active_compositions[composition_id].protocols)}"
        
        protocol_node = ProtocolNode(
            protocol_id=protocol_id,
            protocol_name=protocol_name,
            parameters=parameters,
            dependencies=dependencies or [],
            priority=priority
        )
        
        self.active_compositions[composition_id].add_protocol(protocol_node)
        return protocol_id
    
    async def execute_composition(self,
                                composition_id: str,
                                target_fields: List[ContextField],
                                execution_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a protocol composition with specified strategy."""
        if composition_id not in self.active_compositions:
            raise ValueError(f"Composition {composition_id} not found")
        
        composition = self.active_compositions[composition_id]
        execution_context = execution_context or {}
        
        self.logger.info(f"Starting execution of composition: {composition.name}")
        
        start_time = time.time()
        
        try:
            if composition.execution_strategy == ExecutionMode.SEQUENTIAL:
                results = await self._execute_sequential(composition, target_fields, execution_context)
            elif composition.execution_strategy == ExecutionMode.PARALLEL:
                results = await self._execute_parallel(composition, target_fields, execution_context)
            elif composition.execution_strategy == ExecutionMode.HIERARCHICAL:
                results = await self._execute_hierarchical(composition, target_fields, execution_context)
            elif composition.execution_strategy == ExecutionMode.ADAPTIVE:
                results = await self._execute_adaptive(composition, target_fields, execution_context)
            else:
                raise ValueError(f"Unknown execution strategy: {composition.execution_strategy}")
            
            execution_time = time.time() - start_time
            
            # Record execution history
            execution_record = {
                'composition_id': composition_id,
                'composition_name': composition.name,
                'execution_strategy': composition.execution_strategy.value,
                'execution_time': execution_time,
                'protocols_executed': len(composition.protocols),
                'successful_protocols': len([p for p in composition.protocols if p.status == ProtocolStatus.COMPLETED]),
                'failed_protocols': len([p for p in composition.protocols if p.status == ProtocolStatus.FAILED]),
                'results': results,
                'timestamp': time.time()
            }
            
            self.execution_history.append(execution_record)
            
            # Update performance metrics
            self._update_performance_metrics(execution_record)
            
            self.logger.info(f"Completed execution of composition: {composition.name} in {execution_time:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to execute composition {composition.name}: {e}")
            raise
    
    async def _execute_sequential(self,
                                composition: ProtocolComposition,
                                target_fields: List[ContextField],
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute protocols sequentially."""
        results = {}
        
        # Get execution order based on dependencies
        execution_layers = composition.get_execution_order()
        
        for layer in execution_layers:
            layer_results = {}
            
            for protocol_id in layer:
                protocol = self._get_protocol_by_id(composition, protocol_id)
                
                try:
                    protocol.status = ProtocolStatus.RUNNING
                    protocol.start_time = time.time()
                    
                    # Execute protocol
                    result = await self._execute_single_protocol(protocol, target_fields, context)
                    
                    protocol.result = result
                    protocol.status = ProtocolStatus.COMPLETED
                    protocol.end_time = time.time()
                    
                    layer_results[protocol_id] = result
                    
                    # Update context with results for next protocols
                    context[f"protocol_{protocol_id}_result"] = result
                    
                except Exception as e:
                    protocol.error = e
                    protocol.status = ProtocolStatus.FAILED
                    protocol.end_time = time.time()
                    
                    self.logger.error(f"Protocol {protocol.protocol_name} failed: {e}")
                    
                    if protocol.retry_count < protocol.max_retries:
                        protocol.retry_count += 1
                        protocol.status = ProtocolStatus.PENDING
                        # Retry logic could be implemented here
            
            results.update(layer_results)
        
        return results
    
    async def _execute_parallel(self,
                              composition: ProtocolComposition,
                              target_fields: List[ContextField],
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute protocols in parallel where possible."""
        results = {}
        
        # Get execution order based on dependencies
        execution_layers = composition.get_execution_order()
        
        for layer in execution_layers:
            # Execute all protocols in this layer in parallel
            if len(layer) == 1:
                # Single protocol, execute directly
                protocol_id = layer[0]
                protocol = self._get_protocol_by_id(composition, protocol_id)
                
                try:
                    protocol.status = ProtocolStatus.RUNNING
                    protocol.start_time = time.time()
                    
                    result = await self._execute_single_protocol(protocol, target_fields, context)
                    
                    protocol.result = result
                    protocol.status = ProtocolStatus.COMPLETED
                    protocol.end_time = time.time()
                    
                    results[protocol_id] = result
                    context[f"protocol_{protocol_id}_result"] = result
                    
                except Exception as e:
                    protocol.error = e
                    protocol.status = ProtocolStatus.FAILED
                    protocol.end_time = time.time()
                    
                    self.logger.error(f"Protocol {protocol.protocol_name} failed: {e}")
            else:
                # Multiple protocols, execute in parallel
                tasks = []
                
                for protocol_id in layer:
                    protocol = self._get_protocol_by_id(composition, protocol_id)
                    protocol.status = ProtocolStatus.RUNNING
                    protocol.start_time = time.time()
                    
                    task = asyncio.create_task(
                        self._execute_single_protocol(protocol, target_fields, context)
                    )
                    tasks.append((protocol_id, protocol, task))
                
                # Wait for all parallel tasks to complete
                for protocol_id, protocol, task in tasks:
                    try:
                        result = await task
                        
                        protocol.result = result
                        protocol.status = ProtocolStatus.COMPLETED
                        protocol.end_time = time.time()
                        
                        results[protocol_id] = result
                        context[f"protocol_{protocol_id}_result"] = result
                        
                    except Exception as e:
                        protocol.error = e
                        protocol.status = ProtocolStatus.FAILED
                        protocol.end_time = time.time()
                        
                        self.logger.error(f"Protocol {protocol.protocol_name} failed: {e}")
        
        return results
    
    async def _execute_hierarchical(self,
                                  composition: ProtocolComposition,
                                  target_fields: List[ContextField],
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute protocols in hierarchical order with priority consideration."""
        results = {}
        
        # Sort protocols by priority within each dependency layer
        execution_layers = composition.get_execution_order()
        
        for layer in execution_layers:
            # Sort protocols in this layer by priority (highest first)
            layer_protocols = [self._get_protocol_by_id(composition, pid) for pid in layer]
            layer_protocols.sort(key=lambda p: p.priority, reverse=True)
            
            # Execute high-priority protocols first, then parallelize lower priority
            high_priority = [p for p in layer_protocols if p.priority >= 0.8]
            medium_priority = [p for p in layer_protocols if 0.5 <= p.priority < 0.8]
            low_priority = [p for p in layer_protocols if p.priority < 0.5]
            
            # Execute high priority sequentially
            for protocol in high_priority:
                try:
                    protocol.status = ProtocolStatus.RUNNING
                    protocol.start_time = time.time()
                    
                    result = await self._execute_single_protocol(protocol, target_fields, context)
                    
                    protocol.result = result
                    protocol.status = ProtocolStatus.COMPLETED
                    protocol.end_time = time.time()
                    
                    results[protocol.protocol_id] = result
                    context[f"protocol_{protocol.protocol_id}_result"] = result
                    
                except Exception as e:
                    protocol.error = e
                    protocol.status = ProtocolStatus.FAILED
                    protocol.end_time = time.time()
                    
                    self.logger.error(f"Protocol {protocol.protocol_name} failed: {e}")
            
            # Execute medium and low priority in parallel
            remaining_protocols = medium_priority + low_priority
            if remaining_protocols:
                tasks = []
                
                for protocol in remaining_protocols:
                    protocol.status = ProtocolStatus.RUNNING
                    protocol.start_time = time.time()
                    
                    task = asyncio.create_task(
                        self._execute_single_protocol(protocol, target_fields, context)
                    )
                    tasks.append((protocol, task))
                
                # Wait for parallel tasks
                for protocol, task in tasks:
                    try:
                        result = await task
                        
                        protocol.result = result
                        protocol.status = ProtocolStatus.COMPLETED
                        protocol.end_time = time.time()
                        
                        results[protocol.protocol_id] = result
                        context[f"protocol_{protocol.protocol_id}_result"] = result
                        
                    except Exception as e:
                        protocol.error = e
                        protocol.status = ProtocolStatus.FAILED
                        protocol.end_time = time.time()
                        
                        self.logger.error(f"Protocol {protocol.protocol_name} failed: {e}")
        
        return results
    
    async def _execute_adaptive(self,
                              composition: ProtocolComposition,
                              target_fields: List[ContextField],
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute protocols with adaptive selection based on field states."""
        results = {}
        
        # Use adaptive selector to determine optimal execution strategy
        adaptive_strategy = self.adaptive_selector.select_optimal_strategy(
            composition, target_fields, context
        )
        
        # Update composition execution strategy
        original_strategy = composition.execution_strategy
        composition.execution_strategy = adaptive_strategy
        
        self.logger.info(f"Adaptive execution selected strategy: {adaptive_strategy.value}")
        
        try:
            # Execute with selected strategy
            if adaptive_strategy == ExecutionMode.SEQUENTIAL:
                results = await self._execute_sequential(composition, target_fields, context)
            elif adaptive_strategy == ExecutionMode.PARALLEL:
                results = await self._execute_parallel(composition, target_fields, context)
            elif adaptive_strategy == ExecutionMode.HIERARCHICAL:
                results = await self._execute_hierarchical(composition, target_fields, context)
            
            # Analyze results and potentially adapt further
            adaptation_feedback = self.adaptive_selector.analyze_execution_results(
                composition, results, target_fields
            )
            
            if adaptation_feedback.suggests_strategy_change():
                self.logger.info(f"Adaptive feedback suggests strategy change: {adaptation_feedback}")
                # Could implement mid-execution strategy adaptation here
            
        finally:
            # Restore original strategy
            composition.execution_strategy = original_strategy
        
        return results
    
    async def _execute_single_protocol(self,
                                     protocol: ProtocolNode,
                                     target_fields: List[ContextField],
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single protocol."""
        self.logger.debug(f"Executing protocol: {protocol.protocol_name}")
        
        # Merge protocol parameters with context
        execution_parameters = {**protocol.parameters, **context}
        
        # Execute protocol using base orchestrator
        # For Phase 3, we'll implement a simplified protocol execution
        # that works with our multi-protocol system
        result = self._execute_protocol_by_name(
            protocol.protocol_name,
            target_fields,
            execution_parameters
        )
        
        return result
    
    def _execute_protocol_by_name(self,
                                 protocol_name: str,
                                 target_fields: List[ContextField],
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a protocol by name with simplified interface for multi-protocol system."""
        # For Phase 3, we implement basic protocol operations
        # This can be expanded to integrate with the full protocol shell system
        
        if protocol_name == "field_analysis":
            # Analyze field state
            analysis = {}
            for i, field in enumerate(target_fields):
                analysis[f"field_{i}"] = {
                    "element_count": len(field.elements),
                    "attractor_count": len(field.attractors),
                    "coherence": field.measure_field_coherence(),
                    "resonance_patterns": len(field.resonance_patterns)
                }
            return {"field_analysis": analysis}
        
        elif protocol_name == "attractor_optimization":
            # Optimize attractors
            optimized_count = 0
            for field in target_fields:
                for attractor in field.attractors.values():
                    strength_boost = parameters.get("strength_boost", 1.1)
                    attractor.strength *= strength_boost
                    optimized_count += 1
            return {"optimized_attractors": optimized_count}
        
        elif protocol_name == "resonance_amplification":
            # Amplify resonance patterns
            amplified_count = 0
            for field in target_fields:
                for pattern in field.resonance_patterns.values():
                    amplification_factor = parameters.get("amplification_factor", 1.2)
                    pattern.amplitude *= amplification_factor
                    amplified_count += 1
            return {"amplified_patterns": amplified_count}
        
        elif protocol_name == "field_decay":
            # Apply field decay
            for field in target_fields:
                field.decay()
            return {"decay_applied": True, "field_count": len(target_fields)}
        
        elif protocol_name == "emergence_detection":
            # Detect emergent patterns
            emergence_indicators = []
            for field in target_fields:
                coherence = field.measure_field_coherence()
                if coherence > 0.8:
                    emergence_indicators.append({
                        "type": "high_coherence",
                        "strength": coherence,
                        "field_elements": len(field.elements)
                    })
            return {"emergence_indicators": emergence_indicators}
        
        else:
            # Default protocol execution
            return {
                "protocol": protocol_name,
                "status": "executed",
                "parameters": parameters,
                "target_fields": len(target_fields),
                "timestamp": time.time()
            }
    
    def _get_protocol_by_id(self, composition: ProtocolComposition, protocol_id: str) -> ProtocolNode:
        """Get protocol node by ID."""
        for protocol in composition.protocols:
            if protocol.protocol_id == protocol_id:
                return protocol
        raise ValueError(f"Protocol {protocol_id} not found in composition")
    
    def _update_performance_metrics(self, execution_record: Dict[str, Any]):
        """Update performance metrics based on execution record."""
        composition_name = execution_record['composition_name']
        
        if composition_name not in self.performance_metrics:
            self.performance_metrics[composition_name] = {
                'total_executions': 0,
                'total_time': 0.0,
                'success_rate': 0.0,
                'average_time': 0.0,
                'protocols_per_execution': 0.0
            }
        
        metrics = self.performance_metrics[composition_name]
        metrics['total_executions'] += 1
        metrics['total_time'] += execution_record['execution_time']
        metrics['average_time'] = metrics['total_time'] / metrics['total_executions']
        
        # Calculate success rate
        successful = execution_record['successful_protocols']
        total = execution_record['protocols_executed']
        current_success_rate = successful / total if total > 0 else 0
        
        # Update rolling average success rate
        metrics['success_rate'] = (
            (metrics['success_rate'] * (metrics['total_executions'] - 1) + current_success_rate) 
            / metrics['total_executions']
        )
        
        metrics['protocols_per_execution'] = (
            (metrics['protocols_per_execution'] * (metrics['total_executions'] - 1) + total)
            / metrics['total_executions']
        )
    
    def get_composition_status(self, composition_id: str) -> Dict[str, Any]:
        """Get current status of a composition."""
        if composition_id not in self.active_compositions:
            raise ValueError(f"Composition {composition_id} not found")
        
        composition = self.active_compositions[composition_id]
        
        status_summary = {
            'composition_id': composition_id,
            'name': composition.name,
            'total_protocols': len(composition.protocols),
            'pending': len([p for p in composition.protocols if p.status == ProtocolStatus.PENDING]),
            'running': len([p for p in composition.protocols if p.status == ProtocolStatus.RUNNING]),
            'completed': len([p for p in composition.protocols if p.status == ProtocolStatus.COMPLETED]),
            'failed': len([p for p in composition.protocols if p.status == ProtocolStatus.FAILED]),
            'protocols': [
                {
                    'protocol_id': p.protocol_id,
                    'protocol_name': p.protocol_name,
                    'status': p.status.value,
                    'start_time': p.start_time,
                    'end_time': p.end_time,
                    'retry_count': p.retry_count
                }
                for p in composition.protocols
            ]
        }
        
        return status_summary
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all compositions."""
        return {
            'composition_metrics': self.performance_metrics.copy(),
            'total_executions': len(self.execution_history),
            'average_execution_time': (
                sum(record['execution_time'] for record in self.execution_history) 
                / len(self.execution_history)
            ) if self.execution_history else 0,
            'overall_success_rate': (
                sum(
                    record['successful_protocols'] / record['protocols_executed'] 
                    for record in self.execution_history 
                    if record['protocols_executed'] > 0
                ) / len(self.execution_history)
            ) if self.execution_history else 0
        }


class AdaptiveProtocolSelector:
    """Selects optimal protocol execution strategies based on field states and context."""
    
    def __init__(self, field_manager: FieldManager):
        self.field_manager = field_manager
        self.strategy_history: List[Dict[str, Any]] = []
    
    def select_optimal_strategy(self,
                              composition: ProtocolComposition,
                              target_fields: List[ContextField],
                              context: Dict[str, Any]) -> ExecutionMode:
        """Select optimal execution strategy based on current conditions."""
        
        # Analyze field states
        field_analysis = self._analyze_field_states(target_fields)
        
        # Analyze composition characteristics
        composition_analysis = self._analyze_composition(composition)
        
        # Analyze context requirements
        context_analysis = self._analyze_context_requirements(context)
        
        # Calculate strategy scores
        strategy_scores = self._calculate_strategy_scores(
            field_analysis, composition_analysis, context_analysis
        )
        
        # Select best strategy
        optimal_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        
        # Record selection for learning
        self._record_strategy_selection(
            composition, optimal_strategy, strategy_scores, 
            field_analysis, composition_analysis, context_analysis
        )
        
        return ExecutionMode(optimal_strategy)
    
    def _analyze_field_states(self, target_fields: List[ContextField]) -> Dict[str, Any]:
        """Analyze current state of target fields."""
        if not target_fields:
            return {'complexity': 'low', 'interactions': 'minimal', 'stability': 'high'}
        
        total_elements = sum(len(field.elements) for field in target_fields)
        total_attractors = sum(len(field.attractors) for field in target_fields)
        
        # Calculate field interaction complexity
        interaction_complexity = 0
        for i, field_a in enumerate(target_fields):
            for field_b in target_fields[i+1:]:
                # Simple interaction measure based on shared elements
                shared_content = len(set(field_a.elements.keys()) & set(field_b.elements.keys()))
                if shared_content > 0:
                    interaction_complexity += shared_content
        
        # Determine complexity level
        if total_elements > 100 or total_attractors > 10 or interaction_complexity > 20:
            complexity = 'high'
        elif total_elements > 50 or total_attractors > 5 or interaction_complexity > 10:
            complexity = 'medium'
        else:
            complexity = 'low'
        
        # Determine interaction level
        if interaction_complexity > 15:
            interactions = 'high'
        elif interaction_complexity > 5:
            interactions = 'medium'
        else:
            interactions = 'minimal'
        
        # Calculate stability (based on field coherence)
        avg_coherence = sum(field.measure_field_coherence() for field in target_fields) / len(target_fields)
        if avg_coherence > 0.8:
            stability = 'high'
        elif avg_coherence > 0.6:
            stability = 'medium'
        else:
            stability = 'low'
        
        return {
            'complexity': complexity,
            'interactions': interactions,
            'stability': stability,
            'total_elements': total_elements,
            'total_attractors': total_attractors,
            'interaction_complexity': interaction_complexity,
            'avg_coherence': avg_coherence
        }
    
    def _analyze_composition(self, composition: ProtocolComposition) -> Dict[str, Any]:
        """Analyze composition characteristics."""
        total_protocols = len(composition.protocols)
        
        # Calculate dependency complexity
        total_dependencies = sum(len(p.dependencies) for p in composition.protocols)
        dependency_ratio = total_dependencies / total_protocols if total_protocols > 0 else 0
        
        # Analyze priority distribution
        priorities = [p.priority for p in composition.protocols]
        priority_variance = sum((p - sum(priorities)/len(priorities))**2 for p in priorities) / len(priorities) if priorities else 0
        
        # Calculate execution layers
        execution_layers = composition.get_execution_order()
        parallelization_potential = max(len(layer) for layer in execution_layers) if execution_layers else 1
        
        return {
            'total_protocols': total_protocols,
            'dependency_ratio': dependency_ratio,
            'priority_variance': priority_variance,
            'execution_layers': len(execution_layers),
            'parallelization_potential': parallelization_potential,
            'complexity': 'high' if total_protocols > 10 or dependency_ratio > 0.5 else 'medium' if total_protocols > 5 else 'low'
        }
    
    def _analyze_context_requirements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context requirements for execution."""
        urgency = context.get('urgency', 'normal')
        resource_availability = context.get('resource_availability', 'high')
        accuracy_requirements = context.get('accuracy_requirements', 'normal')
        
        return {
            'urgency': urgency,
            'resource_availability': resource_availability,
            'accuracy_requirements': accuracy_requirements,
            'parallel_friendly': resource_availability in ['high', 'unlimited'],
            'speed_priority': urgency in ['high', 'critical'],
            'accuracy_priority': accuracy_requirements in ['high', 'critical']
        }
    
    def _calculate_strategy_scores(self,
                                 field_analysis: Dict[str, Any],
                                 composition_analysis: Dict[str, Any],
                                 context_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate scores for each execution strategy."""
        scores = {
            'sequential': 0.0,
            'parallel': 0.0,
            'hierarchical': 0.0
        }
        
        # Sequential strategy scoring
        scores['sequential'] += 0.8 if field_analysis['stability'] == 'high' else 0.4
        scores['sequential'] += 0.7 if composition_analysis['dependency_ratio'] > 0.7 else 0.3
        scores['sequential'] += 0.8 if context_analysis['accuracy_priority'] else 0.4
        scores['sequential'] += 0.6 if composition_analysis['total_protocols'] <= 5 else 0.2
        
        # Parallel strategy scoring
        scores['parallel'] += 0.9 if context_analysis['parallel_friendly'] else 0.2
        scores['parallel'] += 0.8 if context_analysis['speed_priority'] else 0.4
        scores['parallel'] += 0.7 if composition_analysis['parallelization_potential'] > 2 else 0.3
        scores['parallel'] += 0.6 if composition_analysis['dependency_ratio'] < 0.3 else 0.2
        scores['parallel'] += 0.5 if field_analysis['interactions'] == 'minimal' else 0.2
        
        # Hierarchical strategy scoring
        scores['hierarchical'] += 0.8 if composition_analysis['priority_variance'] > 0.1 else 0.4
        scores['hierarchical'] += 0.7 if composition_analysis['total_protocols'] > 8 else 0.4
        scores['hierarchical'] += 0.6 if field_analysis['complexity'] == 'high' else 0.3
        scores['hierarchical'] += 0.5 if composition_analysis['dependency_ratio'] > 0.4 else 0.3
        
        # Normalize scores
        max_score = max(scores.values()) if scores.values() else 1.0
        return {strategy: score / max_score for strategy, score in scores.items()}
    
    def _record_strategy_selection(self,
                                 composition: ProtocolComposition,
                                 selected_strategy: str,
                                 strategy_scores: Dict[str, float],
                                 field_analysis: Dict[str, Any],
                                 composition_analysis: Dict[str, Any],
                                 context_analysis: Dict[str, Any]):
        """Record strategy selection for learning purposes."""
        record = {
            'composition_id': composition.composition_id,
            'selected_strategy': selected_strategy,
            'strategy_scores': strategy_scores,
            'field_analysis': field_analysis,
            'composition_analysis': composition_analysis,
            'context_analysis': context_analysis,
            'timestamp': time.time()
        }
        
        self.strategy_history.append(record)
    
    def analyze_execution_results(self,
                                composition: ProtocolComposition,
                                results: Dict[str, Any],
                                target_fields: List[ContextField]) -> 'AdaptationFeedback':
        """Analyze execution results to provide adaptation feedback."""
        
        # Calculate execution metrics
        successful_protocols = len([p for p in composition.protocols if p.status == ProtocolStatus.COMPLETED])
        total_protocols = len(composition.protocols)
        success_rate = successful_protocols / total_protocols if total_protocols > 0 else 0
        
        total_execution_time = sum(
            (p.end_time - p.start_time) for p in composition.protocols 
            if p.start_time and p.end_time
        )
        
        # Analyze field state changes
        field_coherence_change = self._measure_field_coherence_change(target_fields)
        
        # Determine if strategy change is recommended
        strategy_change_recommended = (
            success_rate < 0.8 or 
            total_execution_time > 60.0 or  # More than 1 minute
            field_coherence_change < -0.1  # Significant coherence decrease
        )
        
        return AdaptationFeedback(
            success_rate=success_rate,
            execution_time=total_execution_time,
            field_coherence_change=field_coherence_change,
            strategy_change_recommended=strategy_change_recommended,
            recommended_strategy=self._suggest_alternative_strategy(composition) if strategy_change_recommended else None
        )
    
    def _measure_field_coherence_change(self, target_fields: List[ContextField]) -> float:
        """Measure change in field coherence (simplified implementation)."""
        # This would need baseline measurements in a real implementation
        current_coherence = sum(field.measure_field_coherence() for field in target_fields) / len(target_fields) if target_fields else 0
        # For now, return a small positive change as default
        return 0.05
    
    def _suggest_alternative_strategy(self, composition: ProtocolComposition) -> ExecutionMode:
        """Suggest alternative strategy when current one underperforms."""
        current_strategy = composition.execution_strategy
        
        if current_strategy == ExecutionMode.SEQUENTIAL:
            return ExecutionMode.PARALLEL
        elif current_strategy == ExecutionMode.PARALLEL:
            return ExecutionMode.HIERARCHICAL
        else:
            return ExecutionMode.SEQUENTIAL


@dataclass
class AdaptationFeedback:
    """Feedback from execution analysis for adaptive strategy selection."""
    success_rate: float
    execution_time: float
    field_coherence_change: float
    strategy_change_recommended: bool
    recommended_strategy: Optional[ExecutionMode] = None
    
    def suggests_strategy_change(self) -> bool:
        """Check if feedback suggests strategy change."""
        return self.strategy_change_recommended


class CrossProtocolResonanceDetector:
    """Detects resonance patterns between different protocols and compositions."""
    
    def __init__(self):
        self.resonance_history: List[Dict[str, Any]] = []
        self.resonance_patterns: Dict[str, List[Dict[str, Any]]] = {}
    
    def detect_cross_protocol_resonance(self,
                                      compositions: List[ProtocolComposition],
                                      target_fields: List[ContextField]) -> Dict[str, Any]:
        """Detect resonance patterns across multiple protocol compositions."""
        
        resonance_matrix = self._calculate_resonance_matrix(compositions, target_fields)
        resonance_clusters = self._identify_resonance_clusters(resonance_matrix)
        emergence_indicators = self._detect_emergence_indicators(resonance_clusters, target_fields)
        
        resonance_analysis = {
            'resonance_matrix': resonance_matrix,
            'resonance_clusters': resonance_clusters,
            'emergence_indicators': emergence_indicators,
            'overall_resonance_strength': self._calculate_overall_resonance(resonance_matrix),
            'recommended_optimizations': self._suggest_resonance_optimizations(resonance_clusters)
        }
        
        # Record for pattern learning
        self._record_resonance_pattern(resonance_analysis, compositions, target_fields)
        
        return resonance_analysis
    
    def _calculate_resonance_matrix(self,
                                  compositions: List[ProtocolComposition],
                                  target_fields: List[ContextField]) -> Dict[str, Dict[str, float]]:
        """Calculate resonance between all composition pairs."""
        matrix = {}
        
        for i, comp_a in enumerate(compositions):
            matrix[comp_a.composition_id] = {}
            
            for j, comp_b in enumerate(compositions):
                if i == j:
                    matrix[comp_a.composition_id][comp_b.composition_id] = 1.0
                else:
                    resonance = self._calculate_composition_resonance(comp_a, comp_b, target_fields)
                    matrix[comp_a.composition_id][comp_b.composition_id] = resonance
        
        return matrix
    
    def _calculate_composition_resonance(self,
                                       comp_a: ProtocolComposition,
                                       comp_b: ProtocolComposition,
                                       target_fields: List[ContextField]) -> float:
        """Calculate resonance between two compositions."""
        
        # Protocol name similarity
        protocols_a = set(p.protocol_name for p in comp_a.protocols)
        protocols_b = set(p.protocol_name for p in comp_b.protocols)
        protocol_similarity = len(protocols_a & protocols_b) / len(protocols_a | protocols_b) if protocols_a | protocols_b else 0
        
        # Parameter similarity
        param_similarity = self._calculate_parameter_similarity(comp_a, comp_b)
        
        # Field impact similarity
        field_impact_similarity = self._calculate_field_impact_similarity(comp_a, comp_b, target_fields)
        
        # Weighted resonance calculation
        resonance = (
            protocol_similarity * 0.4 +
            param_similarity * 0.3 +
            field_impact_similarity * 0.3
        )
        
        return resonance
    
    def _calculate_parameter_similarity(self,
                                      comp_a: ProtocolComposition,
                                      comp_b: ProtocolComposition) -> float:
        """Calculate similarity in composition parameters."""
        # Simplified parameter similarity
        params_a = set(comp_a.composition_parameters.keys())
        params_b = set(comp_b.composition_parameters.keys())
        
        if not params_a and not params_b:
            return 1.0
        
        return len(params_a & params_b) / len(params_a | params_b) if params_a | params_b else 0
    
    def _calculate_field_impact_similarity(self,
                                         comp_a: ProtocolComposition,
                                         comp_b: ProtocolComposition,
                                         target_fields: List[ContextField]) -> float:
        """Calculate similarity in field impact patterns."""
        # This would analyze how similarly the compositions affect field states
        # Simplified implementation returns moderate similarity
        return 0.5
    
    def _identify_resonance_clusters(self, resonance_matrix: Dict[str, Dict[str, float]]) -> List[List[str]]:
        """Identify clusters of highly resonant compositions."""
        clusters = []
        processed = set()
        
        for comp_id in resonance_matrix:
            if comp_id in processed:
                continue
            
            cluster = [comp_id]
            processed.add(comp_id)
            
            # Find compositions with high resonance (> 0.7)
            for other_comp_id, resonance in resonance_matrix[comp_id].items():
                if other_comp_id != comp_id and resonance > 0.7 and other_comp_id not in processed:
                    cluster.append(other_comp_id)
                    processed.add(other_comp_id)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def _detect_emergence_indicators(self,
                                   resonance_clusters: List[List[str]],
                                   target_fields: List[ContextField]) -> Dict[str, Any]:
        """Detect indicators of emergent behavior from resonance patterns."""
        
        emergence_strength = 0.0
        emergence_types = []
        
        # Cluster-based emergence
        if resonance_clusters:
            emergence_strength += len(resonance_clusters) * 0.1
            emergence_types.append('cluster_formation')
        
        # Field coherence emergence
        if target_fields:
            avg_coherence = sum(field.measure_field_coherence() for field in target_fields) / len(target_fields)
            if avg_coherence > 0.8:
                emergence_strength += 0.3
                emergence_types.append('field_coherence')
        
        # Cross-field interaction emergence
        if len(target_fields) > 1:
            emergence_strength += 0.2
            emergence_types.append('cross_field_interaction')
        
        return {
            'emergence_strength': min(emergence_strength, 1.0),
            'emergence_types': emergence_types,
            'emergence_detected': emergence_strength > 0.5
        }
    
    def _calculate_overall_resonance(self, resonance_matrix: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall resonance strength across all compositions."""
        total_resonance = 0.0
        count = 0
        
        for comp_id in resonance_matrix:
            for other_comp_id, resonance in resonance_matrix[comp_id].items():
                if comp_id != other_comp_id:
                    total_resonance += resonance
                    count += 1
        
        return total_resonance / count if count > 0 else 0.0
    
    def _suggest_resonance_optimizations(self, resonance_clusters: List[List[str]]) -> List[str]:
        """Suggest optimizations based on resonance patterns."""
        optimizations = []
        
        if resonance_clusters:
            optimizations.append("Consider merging highly resonant compositions for efficiency")
            optimizations.append("Explore parallel execution of resonant composition clusters")
        
        if len(resonance_clusters) > 3:
            optimizations.append("High resonance diversity detected - consider hierarchical organization")
        
        return optimizations
    
    def _record_resonance_pattern(self,
                                resonance_analysis: Dict[str, Any],
                                compositions: List[ProtocolComposition],
                                target_fields: List[ContextField]):
        """Record resonance pattern for learning."""
        pattern_record = {
            'resonance_analysis': resonance_analysis,
            'composition_count': len(compositions),
            'field_count': len(target_fields),
            'timestamp': time.time()
        }
        
        self.resonance_history.append(pattern_record)
        
        # Update pattern database
        pattern_key = f"comp_{len(compositions)}_fields_{len(target_fields)}"
        if pattern_key not in self.resonance_patterns:
            self.resonance_patterns[pattern_key] = []
        
        self.resonance_patterns[pattern_key].append(pattern_record)