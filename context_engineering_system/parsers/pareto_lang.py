"""
Pareto-lang Parser
==================

Parser for Pareto-lang protocol shell definitions used in field operations.
Supports parsing structured protocol definitions into executable objects.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class OperationType(Enum):
    """Types of operations in protocol shells."""
    ATTRACTOR = "attractor"
    RESIDUE = "residue"
    BOUNDARY = "boundary"
    FIELD = "field"
    AGENCY = "agency"
    RESONANCE = "resonance"
    MEMORY = "memory"


@dataclass
class ProtocolOperation:
    """A single operation within a protocol shell."""
    namespace: str
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    full_path: str = ""


@dataclass
class ProtocolShell:
    """A parsed protocol shell with structured operations."""
    name: str
    intent: str
    input_spec: Dict[str, Any] = field(default_factory=dict)
    process_operations: List[ProtocolOperation] = field(default_factory=list)
    output_spec: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    raw_content: str = ""


class ParetoLangParser:
    """
    Parser for Pareto-lang protocol shell definitions.
    
    Pareto-lang is a concise syntax for defining field operations:
    
    /protocol_name {
      intent: "Clear statement of protocol purpose",
      
      input: {
        input_field_1: <type>,
        input_field_2: <type>
      },
      
      process: [
        "/operation.name{param='value'}",
        "/operation.name{param='value'}"
      ],
      
      output: {
        output_field_1: <type>,
        output_field_2: <type>
      },
      
      meta: {
        version: "x.y.z",
        timestamp: "<now>"
      }
    }
    """
    
    def __init__(self):
        # Regex patterns for parsing
        self.protocol_pattern = r'/(\w+(?:\.\w+)*)\s*\{'
        self.intent_pattern = r'intent\s*[=:]\s*["\']([^"\']*)["\']'
        self.section_pattern = r'(\w+)\s*[=:]\s*\{'
        self.operation_pattern = r'["\']?(/[\w\.]+\{[^}]*\})["\']?'
        self.param_pattern = r'(\w+)\s*=\s*["\']?([^,}\'"]*)["\']?'
        
    def parse_file(self, filepath: str) -> ProtocolShell:
        """Parse a protocol shell from a file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return self.parse_content(content)
    
    def parse_content(self, content: str) -> ProtocolShell:
        """Parse a protocol shell from content string."""
        # Clean content
        content = self._clean_content(content)
        
        # Extract protocol name
        protocol_name = self._extract_protocol_name(content)
        
        # Extract intent
        intent = self._extract_intent(content)
        
        # Extract sections
        input_spec = self._extract_section(content, "input")
        process_operations = self._extract_process_section(content)
        output_spec = self._extract_section(content, "output")
        meta = self._extract_section(content, "meta")
        
        return ProtocolShell(
            name=protocol_name,
            intent=intent,
            input_spec=input_spec,
            process_operations=process_operations,
            output_spec=output_spec,
            meta=meta,
            raw_content=content
        )
    
    def validate_protocol(self, protocol: ProtocolShell) -> Tuple[bool, List[str]]:
        """
        Validate a parsed protocol shell.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not protocol.name:
            errors.append("Protocol must have a name")
        
        if not protocol.intent:
            errors.append("Protocol must have an intent statement")
        
        if not protocol.process_operations:
            errors.append("Protocol must have at least one process operation")
        
        # Validate operations
        for op in protocol.process_operations:
            if not op.namespace or not op.operation:
                errors.append(f"Invalid operation format: {op.full_path}")
        
        return len(errors) == 0, errors
    
    def _clean_content(self, content: str) -> str:
        """Clean content by removing comments and normalizing whitespace."""
        # Remove single-line comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()
    
    def _extract_protocol_name(self, content: str) -> str:
        """Extract protocol name from content."""
        match = re.search(self.protocol_pattern, content)
        if match:
            return match.group(1)
        return ""
    
    def _extract_intent(self, content: str) -> str:
        """Extract intent statement from content."""
        match = re.search(self.intent_pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)
        return ""
    
    def _extract_section(self, content: str, section_name: str) -> Dict[str, Any]:
        """Extract a structured section from content."""
        # Find section start
        pattern = rf'{section_name}\s*[=:]\s*\{{'
        match = re.search(pattern, content, re.IGNORECASE)
        
        if not match:
            return {}
        
        # Find matching closing brace
        start_pos = match.end() - 1  # Position of opening brace
        brace_count = 1
        pos = start_pos + 1
        
        while pos < len(content) and brace_count > 0:
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
            pos += 1
        
        if brace_count == 0:
            section_content = content[start_pos + 1:pos - 1].strip()
            return self._parse_json_like(section_content)
        
        return {}
    
    def _extract_process_section(self, content: str) -> List[ProtocolOperation]:
        """Extract process operations from content."""
        # Find process section
        pattern = r'process\s*[=:]\s*\['
        match = re.search(pattern, content, re.IGNORECASE)
        
        if not match:
            return []
        
        # Find matching closing bracket
        start_pos = match.end() - 1  # Position of opening bracket
        bracket_count = 1
        pos = start_pos + 1
        
        while pos < len(content) and bracket_count > 0:
            if content[pos] == '[':
                bracket_count += 1
            elif content[pos] == ']':
                bracket_count -= 1
            pos += 1
        
        if bracket_count == 0:
            process_content = content[start_pos + 1:pos - 1].strip()
            return self._parse_operations(process_content)
        
        return []
    
    def _parse_operations(self, process_content: str) -> List[ProtocolOperation]:
        """Parse operation strings into ProtocolOperation objects."""
        operations = []
        
        # Find all operation strings
        operation_matches = re.findall(self.operation_pattern, process_content)
        
        for op_string in operation_matches:
            operation = self._parse_single_operation(op_string)
            if operation:
                operations.append(operation)
        
        return operations
    
    def _parse_single_operation(self, op_string: str) -> Optional[ProtocolOperation]:
        """Parse a single operation string."""
        # Extract operation path and parameters
        # Format: /namespace.operation{param1='value1', param2='value2'}
        
        # Match operation path
        path_match = re.match(r'/([^{]+)', op_string)
        if not path_match:
            return None
        
        full_path = path_match.group(1)
        path_parts = full_path.split('.')
        
        if len(path_parts) < 2:
            return None
        
        namespace = path_parts[0]
        operation = '.'.join(path_parts[1:])
        
        # Extract parameters
        params_match = re.search(r'\{([^}]*)\}', op_string)
        parameters = {}
        
        if params_match:
            params_content = params_match.group(1)
            param_matches = re.findall(self.param_pattern, params_content)
            
            for param_name, param_value in param_matches:
                # Try to parse as JSON value
                try:
                    if param_value.lower() in ['true', 'false']:
                        parameters[param_name] = param_value.lower() == 'true'
                    elif param_value.isdigit():
                        parameters[param_name] = int(param_value)
                    elif '.' in param_value and param_value.replace('.', '').isdigit():
                        parameters[param_name] = float(param_value)
                    else:
                        parameters[param_name] = param_value
                except:
                    parameters[param_name] = param_value
        
        return ProtocolOperation(
            namespace=namespace,
            operation=operation,
            parameters=parameters,
            full_path=full_path
        )
    
    def _parse_json_like(self, content: str) -> Dict[str, Any]:
        """Parse JSON-like content with flexible syntax."""
        result = {}
        
        # Simple key-value parsing
        # This is a simplified parser - in production, you'd want more robust parsing
        lines = content.split(',')
        
        for line in lines:
            line = line.strip()
            if ':' in line or '=' in line:
                # Try both : and = as separators
                if ':' in line:
                    key, value = line.split(':', 1)
                else:
                    key, value = line.split('=', 1)
                
                key = key.strip().strip('"\'')
                value = value.strip().strip('"\'')
                
                # Try to convert value to appropriate type
                try:
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    elif value.isdigit():
                        value = int(value)
                    elif '.' in value and value.replace('.', '').isdigit():
                        value = float(value)
                    elif value.startswith('[') and value.endswith(']'):
                        # Simple list parsing
                        list_content = value[1:-1].strip()
                        if list_content:
                            value = [item.strip().strip('"\'') for item in list_content.split(',')]
                        else:
                            value = []
                    elif value.startswith('{') and value.endswith('}'):
                        # Nested object - recursively parse
                        value = self._parse_json_like(value[1:-1])
                except:
                    pass  # Keep as string if parsing fails
                
                result[key] = value
        
        return result
    
    def to_executable_format(self, protocol: ProtocolShell) -> Dict[str, Any]:
        """Convert protocol shell to executable format."""
        return {
            "name": protocol.name,
            "intent": protocol.intent,
            "input_spec": protocol.input_spec,
            "operations": [
                {
                    "namespace": op.namespace,
                    "operation": op.operation,
                    "parameters": op.parameters,
                    "full_path": op.full_path
                }
                for op in protocol.process_operations
            ],
            "output_spec": protocol.output_spec,
            "meta": protocol.meta
        }
    
    def generate_python_code(self, protocol: ProtocolShell) -> str:
        """Generate Python code template for protocol implementation."""
        code_lines = [
            f'"""',
            f'Protocol: {protocol.name}',
            f'Intent: {protocol.intent}',
            f'"""',
            f'',
            f'def execute_{protocol.name.replace(".", "_")}(context_field, **kwargs):',
            f'    """Execute {protocol.name} protocol."""',
            f'    # Input validation',
            f'    required_inputs = {list(protocol.input_spec.keys())}',
            f'    for input_name in required_inputs:',
            f'        if input_name not in kwargs:',
            f'            raise ValueError(f"Missing required input: {{input_name}}")',
            f'    ',
            f'    # Initialize results',
            f'    results = {{}}'
        ]
        
        # Add operation implementations
        for i, op in enumerate(protocol.process_operations):
            code_lines.extend([
                f'    ',
                f'    # Operation {i+1}: {op.full_path}',
                f'    operation_{i+1}_result = context_field.{op.namespace}_{op.operation}(**{op.parameters})',
                f'    results["operation_{i+1}"] = operation_{i+1}_result'
            ])
        
        code_lines.extend([
            f'    ',
            f'    # Return structured results',
            f'    return {{',
            f'        "protocol_name": "{protocol.name}",',
            f'        "execution_results": results,',
            f'        "field_state": context_field.get_field_state(),',
            f'        "output": {{',
        ])
        
        # Add output mappings
        for output_name in protocol.output_spec.keys():
            code_lines.append(f'            "{output_name}": results.get("relevant_operation_result"),')
        
        code_lines.extend([
            f'        }}',
            f'    }}'
        ])
        
        return '\n'.join(code_lines)


# Utility functions for working with protocol shells

def load_protocol_shell(filepath: str) -> ProtocolShell:
    """Load and parse a protocol shell from file."""
    parser = ParetoLangParser()
    return parser.parse_file(filepath)


def validate_protocol_shell(filepath: str) -> Tuple[bool, List[str]]:
    """Validate a protocol shell file."""
    parser = ParetoLangParser()
    protocol = parser.parse_file(filepath)
    return parser.validate_protocol(protocol)


def generate_protocol_template(name: str, intent: str) -> str:
    """Generate a template for a new protocol shell."""
    template = f"""/{name} {{
  intent="{intent}",
  
  input={{
    current_field_state="<field_state>",
    // Add more input parameters as needed
  }},
  
  process=[
    "/attractor.scan{{detect='attractors', filter_by='strength'}}",
    "/field.audit{{surface_new='attractor_basins'}}",
    // Add more operations as needed
  ],
  
  output={{
    updated_field_state="<new_state>",
    // Add more output parameters as needed
  }},
  
  meta={{
    version="1.0.0",
    timestamp="<now>"
  }}
}}"""
    return template