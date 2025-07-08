"""
Protocol Orchestrator
====================

Orchestrator for executing protocol shells on context fields.
Manages protocol execution, sequencing, and result integration.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..parsers.pareto_lang import ProtocolShell, ProtocolOperation


class ExecutionStatus(Enum):
    """Status of protocol execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Result of protocol execution."""
    protocol_name: str
    status: ExecutionStatus
    start_time: float
    end_time: Optional[float] = None
    results: Dict[str, Any] = field(default_factory=dict)
    field_state: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_log: List[str] = field(default_factory=list)


class ProtocolOrchestrator:
    """
    Orchestrator for protocol shell execution on context fields.
    
    Manages the execution of protocol shells, handles operation sequencing,
    and integrates results back into the field state.
    """
    
    def __init__(self):
        """Initialize protocol orchestrator."""
        self.operation_handlers: Dict[str, Callable] = {}
        self.execution_history: List[ExecutionResult] = []
        self.active_executions: Dict[str, ExecutionResult] = {}
        
        # Register default operation handlers
        self._register_default_handlers()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def register_operation_handler(self, 
                                  namespace: str, 
                                  operation: str, 
                                  handler: Callable) -> None:
        """
        Register a handler for a specific operation.
        
        Args:
            namespace: Operation namespace (e.g., 'attractor', 'field')
            operation: Operation name (e.g., 'scan', 'audit')
            handler: Function to handle the operation
        """
        key = f"{namespace}.{operation}"
        self.operation_handlers[key] = handler
        self.logger.info(f"Registered handler for operation: {key}")
    
    def execute_protocol(self, 
                        protocol: ProtocolShell,
                        context_field,
                        input_data: Dict[str, Any] = None,
                        execution_id: Optional[str] = None) -> ExecutionResult:
        """
        Execute a protocol shell on a context field.
        
        Args:
            protocol: Protocol shell to execute
            context_field: Context field to operate on
            input_data: Input data for the protocol
            execution_id: Optional execution ID for tracking
            
        Returns:
            ExecutionResult with execution details and results
        """
        if execution_id is None:
            execution_id = f"{protocol.name}_{int(time.time() * 1000)}"
        
        # Create execution result tracker
        result = ExecutionResult(
            protocol_name=protocol.name,
            status=ExecutionStatus.PENDING,
            start_time=time.time()
        )
        
        self.active_executions[execution_id] = result
        
        try:
            result.status = ExecutionStatus.RUNNING
            result.execution_log.append(f"Starting execution of protocol: {protocol.name}")
            
            # Validate input data
            if not self._validate_input_data(protocol, input_data or {}):
                raise ValueError(f"Invalid input data for protocol {protocol.name}")
            
            # Execute protocol operations
            operation_results = self._execute_operations(
                protocol.process_operations, 
                context_field, 
                input_data or {},
                result
            )
            
            # Process results according to output specification
            output_data = self._process_output(
                protocol.output_spec,
                operation_results,
                context_field
            )
            
            # Update execution result
            result.results = operation_results
            result.field_state = context_field.get_field_state()
            result.status = ExecutionStatus.COMPLETED
            result.end_time = time.time()
            result.execution_log.append(f"Completed execution successfully")
            
            self.logger.info(f"Protocol {protocol.name} executed successfully")
            
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.end_time = time.time()
            result.error_message = str(e)
            result.execution_log.append(f"Execution failed: {str(e)}")
            
            self.logger.error(f"Protocol {protocol.name} execution failed: {str(e)}")
        
        finally:
            # Move to history
            self.execution_history.append(result)
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
        
        return result
    
    def execute_protocol_sequence(self,
                                 protocols: List[ProtocolShell],
                                 context_field,
                                 input_data: Dict[str, Any] = None) -> List[ExecutionResult]:
        """
        Execute a sequence of protocols on a context field.
        
        Args:
            protocols: List of protocol shells to execute
            context_field: Context field to operate on
            input_data: Initial input data
            
        Returns:
            List of ExecutionResult objects
        """
        results = []
        current_input = input_data or {}
        
        for i, protocol in enumerate(protocols):
            self.logger.info(f"Executing protocol {i+1}/{len(protocols)}: {protocol.name}")
            
            # Execute protocol
            result = self.execute_protocol(
                protocol, 
                context_field, 
                current_input,
                f"sequence_{int(time.time() * 1000)}_{i}"
            )
            
            results.append(result)
            
            # If execution failed, stop the sequence
            if result.status == ExecutionStatus.FAILED:
                self.logger.error(f"Protocol sequence stopped at {protocol.name} due to failure")
                break
            
            # Pass results to next protocol
            current_input = {
                "previous_results": result.results,
                "field_state": result.field_state,
                **current_input
            }
        
        return results
    
    def _validate_input_data(self, protocol: ProtocolShell, input_data: Dict[str, Any]) -> bool:
        """Validate input data against protocol specification."""
        for required_input in protocol.input_spec.keys():
            if required_input not in input_data:
                self.logger.warning(f"Missing required input: {required_input}")
                return False
        return True
    
    def _execute_operations(self,
                          operations: List[ProtocolOperation],
                          context_field,
                          input_data: Dict[str, Any],
                          result: ExecutionResult) -> Dict[str, Any]:
        """Execute all operations in a protocol."""
        operation_results = {}
        
        for i, operation in enumerate(operations):
            operation_key = f"operation_{i+1}"
            
            try:
                result.execution_log.append(f"Executing operation: {operation.full_path}")
                
                # Execute operation
                op_result = self._execute_single_operation(
                    operation, 
                    context_field, 
                    input_data,
                    operation_results
                )
                
                operation_results[operation_key] = {
                    "operation": operation.full_path,
                    "parameters": operation.parameters,
                    "result": op_result,
                    "timestamp": time.time()
                }
                
                result.execution_log.append(f"Operation completed: {operation.full_path}")
                
            except Exception as e:
                error_msg = f"Operation {operation.full_path} failed: {str(e)}"
                result.execution_log.append(error_msg)
                operation_results[operation_key] = {
                    "operation": operation.full_path,
                    "parameters": operation.parameters,
                    "error": str(e),
                    "timestamp": time.time()
                }
                # Continue with other operations rather than failing completely
                self.logger.warning(error_msg)
        
        return operation_results
    
    def _execute_single_operation(self,
                                operation: ProtocolOperation,
                                context_field,
                                input_data: Dict[str, Any],
                                previous_results: Dict[str, Any]) -> Any:
        """Execute a single protocol operation."""
        handler_key = f"{operation.namespace}.{operation.operation}"
        
        if handler_key in self.operation_handlers:
            # Use registered handler
            handler = self.operation_handlers[handler_key]
            return handler(
                context_field=context_field,
                parameters=operation.parameters,
                input_data=input_data,
                previous_results=previous_results
            )
        else:
            # Try to find a default implementation
            return self._execute_default_operation(
                operation,
                context_field,
                input_data,
                previous_results
            )
    
    def _execute_default_operation(self,
                                 operation: ProtocolOperation,
                                 context_field,
                                 input_data: Dict[str, Any],
                                 previous_results: Dict[str, Any]) -> Any:
        """Execute operation using default implementations."""
        namespace = operation.namespace
        op_name = operation.operation
        params = operation.parameters
        
        if namespace == "attractor":
            return self._handle_attractor_operation(op_name, params, context_field)
        elif namespace == "field":
            return self._handle_field_operation(op_name, params, context_field)
        elif namespace == "residue":
            return self._handle_residue_operation(op_name, params, context_field)
        elif namespace == "boundary":
            return self._handle_boundary_operation(op_name, params, context_field)
        elif namespace == "agency":
            return self._handle_agency_operation(op_name, params, context_field)
        elif namespace == "resonance":
            return self._handle_resonance_operation(op_name, params, context_field)
        else:
            raise NotImplementedError(f"No handler found for operation: {operation.full_path}")
    
    def _handle_attractor_operation(self, operation: str, params: Dict[str, Any], context_field) -> Any:
        """Handle attractor namespace operations."""
        if operation == "scan":
            # Scan for attractors with optional filtering
            attractors = context_field.get_attractors()
            filter_by = params.get('filter_by', 'strength')
            
            if filter_by == 'strength':
                min_strength = params.get('min_strength', 0.0)
                return [a for a in attractors if a.strength >= min_strength]
            else:
                return attractors
        
        elif operation == "strengthen":
            # Strengthen existing attractors
            strength_factor = params.get('factor', 1.1)
            for attractor in context_field.attractors.values():
                attractor.strength *= strength_factor
            return {"strengthened": len(context_field.attractors)}
        
        elif operation == "create":
            # Create new attractor at specified position
            position = params.get('position', (0.5, 0.5))
            strength = params.get('strength', 1.0)
            name = params.get('name', f"Attractor_{len(context_field.attractors)}")
            
            from ..core.field import Attractor
            attractor = Attractor(
                id=f"manual_{int(time.time() * 1000)}",
                name=name,
                center=position,
                strength=strength,
                radius=params.get('radius', 0.1)
            )
            
            context_field.attractors[attractor.id] = attractor
            return {"created_attractor": attractor.id}
        
        else:
            raise NotImplementedError(f"Attractor operation not implemented: {operation}")
    
    def _handle_field_operation(self, operation: str, params: Dict[str, Any], context_field) -> Any:
        """Handle field namespace operations."""
        if operation == "audit":
            # Audit field state and return analysis
            field_state = context_field.get_field_state()
            analysis = {
                "total_elements": len(context_field.elements),
                "total_attractors": len(context_field.attractors),
                "field_coherence": context_field.measure_field_coherence(),
                "resonance_patterns": len(context_field.resonance_patterns),
                "field_age": time.time() - context_field.creation_time
            }
            
            surface_new = params.get('surface_new')
            if surface_new == 'attractor_basins':
                # Identify potential new attractor formation areas
                analysis['potential_attractors'] = self._identify_potential_attractors(context_field)
            
            return analysis
        
        elif operation == "decay":
            # Apply field decay
            context_field.decay()
            return {"decay_applied": True, "timestamp": time.time()}
        
        elif operation == "snapshot":
            # Take field snapshot
            return context_field.get_field_state()
        
        else:
            raise NotImplementedError(f"Field operation not implemented: {operation}")
    
    def _handle_residue_operation(self, operation: str, params: Dict[str, Any], context_field) -> Any:
        """Handle residue namespace operations."""
        if operation == "surface":
            # Surface symbolic residue in the field
            mode = params.get('mode', 'basic')
            
            # Simplified residue detection
            residue_elements = []
            for element in context_field.elements.values():
                if element.strength < 0.5 and element.element_type.value == 'residue':
                    residue_elements.append(element.id)
            
            return {"surfaced_residue": residue_elements, "count": len(residue_elements)}
        
        elif operation == "compress":
            # Compress residue patterns
            return {"compressed": True, "residue_count": 0}
        
        else:
            raise NotImplementedError(f"Residue operation not implemented: {operation}")
    
    def _handle_boundary_operation(self, operation: str, params: Dict[str, Any], context_field) -> Any:
        """Handle boundary namespace operations."""
        if operation == "collapse":
            # Collapse field boundaries
            context_field.boundary_permeability = min(1.0, context_field.boundary_permeability * 1.2)
            return {"boundary_permeability": context_field.boundary_permeability}
        
        elif operation == "adapt":
            # Adapt boundary properties
            new_permeability = params.get('permeability', context_field.boundary_permeability)
            context_field.boundary_permeability = new_permeability
            return {"adapted_permeability": new_permeability}
        
        else:
            raise NotImplementedError(f"Boundary operation not implemented: {operation}")
    
    def _handle_agency_operation(self, operation: str, params: Dict[str, Any], context_field) -> Any:
        """Handle agency namespace operations."""
        if operation == "activate":
            # Activate autonomous agency
            return {"agency_activated": True, "timestamp": time.time()}
        
        elif operation == "self-prompt":
            # Generate self-prompting behavior
            trigger_condition = params.get('trigger_condition', 'manual')
            return {"self_prompt_triggered": True, "condition": trigger_condition}
        
        else:
            raise NotImplementedError(f"Agency operation not implemented: {operation}")
    
    def _handle_resonance_operation(self, operation: str, params: Dict[str, Any], context_field) -> Any:
        """Handle resonance namespace operations."""
        if operation == "measure":
            # Measure resonance patterns
            patterns = context_field.get_resonance_patterns()
            return {
                "resonance_count": len(patterns),
                "average_coherence": sum(p.coherence_score for p in patterns) / len(patterns) if patterns else 0,
                "patterns": [p.id for p in patterns]
            }
        
        elif operation == "amplify":
            # Amplify resonance patterns
            amplification_factor = params.get('factor', 1.1)
            for pattern in context_field.resonance_patterns.values():
                pattern.amplitude *= amplification_factor
            return {"amplified_patterns": len(context_field.resonance_patterns)}
        
        else:
            raise NotImplementedError(f"Resonance operation not implemented: {operation}")
    
    def _identify_potential_attractors(self, context_field) -> List[Dict[str, Any]]:
        """Identify potential areas for new attractor formation."""
        potential_attractors = []
        
        # Simple heuristic: find areas with high element density
        import numpy as np
        
        if len(context_field.elements) < 3:
            return potential_attractors
        
        positions = np.array([elem.position for elem in context_field.elements.values()])
        
        # Find clusters using simple grid-based approach
        grid_size = 10
        grid_counts = np.zeros((grid_size, grid_size))
        
        for pos in positions:
            grid_x = int(pos[0] * (grid_size - 1))
            grid_y = int(pos[1] * (grid_size - 1))
            grid_counts[grid_x, grid_y] += 1
        
        # Find high-density areas
        threshold = np.mean(grid_counts) + np.std(grid_counts)
        high_density_areas = np.where(grid_counts > threshold)
        
        for i in range(len(high_density_areas[0])):
            grid_x, grid_y = high_density_areas[0][i], high_density_areas[1][i]
            potential_attractors.append({
                "position": (grid_x / (grid_size - 1), grid_y / (grid_size - 1)),
                "density": float(grid_counts[grid_x, grid_y]),
                "confidence": float(grid_counts[grid_x, grid_y] / np.max(grid_counts))
            })
        
        return potential_attractors
    
    def _process_output(self,
                       output_spec: Dict[str, Any],
                       operation_results: Dict[str, Any],
                       context_field) -> Dict[str, Any]:
        """Process operation results according to output specification."""
        output_data = {}
        
        for output_name, output_type in output_spec.items():
            if output_name == "updated_field_state":
                output_data[output_name] = context_field.get_field_state()
            elif output_name == "operation_results":
                output_data[output_name] = operation_results
            else:
                # Try to map from operation results
                output_data[output_name] = operation_results.get(output_name, None)
        
        return output_data
    
    def _register_default_handlers(self) -> None:
        """Register default operation handlers."""
        # Default handlers are implemented in _execute_default_operation
        pass
    
    def get_execution_history(self) -> List[ExecutionResult]:
        """Get execution history."""
        return self.execution_history.copy()
    
    def get_active_executions(self) -> Dict[str, ExecutionResult]:
        """Get currently active executions."""
        return self.active_executions.copy()
    
    def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an active execution.
        
        Args:
            execution_id: ID of execution to cancel
            
        Returns:
            True if cancellation was successful
        """
        if execution_id in self.active_executions:
            result = self.active_executions[execution_id]
            result.status = ExecutionStatus.CANCELLED
            result.end_time = time.time()
            result.execution_log.append("Execution cancelled")
            
            # Move to history
            self.execution_history.append(result)
            del self.active_executions[execution_id]
            
            return True
        
        return False