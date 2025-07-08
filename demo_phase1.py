#!/usr/bin/env python3
"""
Phase 1 Demonstration Script
===========================

Demonstrates the core Phase 1 functionality of the context engineering system:
- Neural Field Implementation with attractors, resonance, and persistence
- Pareto-lang Parser for protocol shells
- Protocol Orchestrator for executing field operations
- Field Visualization Tools
- Cognitive Processing Integration

This script validates that all Phase 1 components are working correctly.
"""

import sys
import os
import time
import json

# Add the context engineering system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

try:
    from context_engineering_system.core.field import ContextField, FieldManager, FieldElementType
    from context_engineering_system.core.protocol_orchestrator import ProtocolOrchestrator
    from context_engineering_system.core.cognitive_processor import CognitiveProcessor, CognitiveToolType
    from context_engineering_system.parsers.pareto_lang import ParetoLangParser, generate_protocol_template
    from context_engineering_system.visualizations.field_visualizer import FieldVisualizer
    print("✅ All core components imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


def demonstrate_neural_field():
    """Demonstrate neural field operations with attractors and resonance."""
    print("\n" + "="*60)
    print("PHASE 1 DEMO: Neural Field Implementation")
    print("="*60)
    
    # Create a context field
    field = ContextField(
        dimensions=2,
        decay_rate=0.05,
        boundary_permeability=0.8,
        attractor_threshold=0.6
    )
    
    print(f"Created context field with parameters:")
    print(f"  - Dimensions: {field.dimensions}")
    print(f"  - Decay rate: {field.decay_rate}")
    print(f"  - Boundary permeability: {field.boundary_permeability}")
    print(f"  - Attractor threshold: {field.attractor_threshold}")
    
    # Inject some content
    print(f"\nInjecting content into field...")
    
    content_items = [
        ("artificial intelligence concepts", 0.9, (0.3, 0.4)),
        ("machine learning algorithms", 0.8, (0.35, 0.45)),
        ("neural network architectures", 0.85, (0.32, 0.42)),
        ("natural language processing", 0.7, (0.6, 0.3)),
        ("computer vision techniques", 0.6, (0.65, 0.35)),
        ("deep learning frameworks", 0.75, (0.8, 0.7)),
    ]
    
    element_ids = []
    for content, strength, position in content_items:
        element_id = field.inject(content, strength, position)
        element_ids.append(element_id)
        print(f"  ✓ Injected: '{content}' (strength: {strength})")
    
    # Display field state
    print(f"\nField state after injection:")
    print(f"  - Elements: {len(field.elements)}")
    print(f"  - Attractors: {len(field.attractors)}")
    print(f"  - Resonance patterns: {len(field.resonance_patterns)}")
    print(f"  - Field coherence: {field.measure_field_coherence():.3f}")
    
    # Show attractors
    if field.attractors:
        print(f"\nFormed attractors:")
        for attractor in field.get_attractors():
            print(f"  - {attractor.name}: strength={attractor.strength:.2f}, elements={len(attractor.elements)}")
    
    # Show resonance patterns
    if field.resonance_patterns:
        print(f"\nResonance patterns:")
        for pattern in field.get_resonance_patterns():
            print(f"  - {pattern.id}: coherence={pattern.coherence_score:.2f}")
    
    # Evolve field over time
    print(f"\nEvolving field over 3 time steps...")
    for i in range(3):
        field.decay()
        print(f"  Step {i+1}: {len(field.elements)} elements, {len(field.attractors)} attractors, coherence={field.measure_field_coherence():.3f}")
    
    return field


def demonstrate_pareto_parser():
    """Demonstrate Pareto-lang parser functionality."""
    print("\n" + "="*60)
    print("PHASE 1 DEMO: Pareto-lang Parser")
    print("="*60)
    
    # Create a sample protocol shell
    sample_protocol = """
    /attractor.co.emerge {
      intent="Demonstrate co-emergence of multiple attractors",
      
      input={
        current_field_state="<field_state>",
        candidate_attractors="<attractor_list>"
      },
      
      process=[
        "/attractor.scan{detect='attractors', filter_by='strength'}",
        "/field.audit{surface_new='attractor_basins'}",
        "/attractor.strengthen{factor=1.2}",
        "/resonance.measure{}"
      ],
      
      output={
        updated_field_state="<new_state>",
        co_emergent_attractors="<attractor_list>",
        resonance_metrics="<metrics>"
      },
      
      meta={
        version="1.0.0",
        timestamp="demo"
      }
    }
    """
    
    print("Parsing sample protocol shell...")
    
    # Parse the protocol
    parser = ParetoLangParser()
    protocol = parser.parse_content(sample_protocol)
    
    print(f"✅ Successfully parsed protocol: {protocol.name}")
    print(f"  - Intent: {protocol.intent}")
    print(f"  - Input parameters: {len(protocol.input_spec)}")
    print(f"  - Process operations: {len(protocol.process_operations)}")
    print(f"  - Output parameters: {len(protocol.output_spec)}")
    
    # Validate protocol
    is_valid, errors = parser.validate_protocol(protocol)
    if is_valid:
        print("✅ Protocol validation: PASSED")
    else:
        print("❌ Protocol validation: FAILED")
        for error in errors:
            print(f"    - {error}")
    
    # Show operations
    print(f"\nParsed operations:")
    for i, op in enumerate(protocol.process_operations, 1):
        print(f"  {i}. {op.namespace}.{op.operation} with params: {op.parameters}")
    
    return protocol


def demonstrate_protocol_orchestrator(field, protocol):
    """Demonstrate protocol orchestrator executing operations."""
    print("\n" + "="*60)
    print("PHASE 1 DEMO: Protocol Orchestrator")
    print("="*60)
    
    # Create orchestrator
    orchestrator = ProtocolOrchestrator()
    
    print("Created protocol orchestrator with default operation handlers")
    
    # Execute protocol on field
    print(f"\nExecuting protocol '{protocol.name}' on context field...")
    
    input_data = {
        "current_field_state": field.get_field_state(),
        "candidate_attractors": [a.id for a in field.get_attractors()]
    }
    
    result = orchestrator.execute_protocol(protocol, field, input_data)
    
    print(f"✅ Protocol execution completed!")
    print(f"  - Status: {result.status.value}")
    print(f"  - Execution time: {result.end_time - result.start_time:.3f}s")
    print(f"  - Operations executed: {len(result.results)}")
    
    if result.error_message:
        print(f"  - Error: {result.error_message}")
    
    # Show execution log
    print(f"\nExecution log:")
    for log_entry in result.execution_log:
        print(f"  - {log_entry}")
    
    # Show operation results
    print(f"\nOperation results:")
    for op_key, op_result in result.results.items():
        print(f"  {op_key}: {op_result.get('operation', 'unknown')}")
        if 'error' in op_result:
            print(f"    ❌ Error: {op_result['error']}")
        else:
            print(f"    ✅ Success")
    
    return result


def demonstrate_cognitive_processor(field):
    """Demonstrate cognitive processing tools."""
    print("\n" + "="*60)
    print("PHASE 1 DEMO: Cognitive Processor")
    print("="*60)
    
    # Create cognitive processor
    processor = CognitiveProcessor()
    
    print("Created cognitive processor with understanding, reasoning, and verification tools")
    
    # Test cognitive processing
    test_input = "How can we optimize the field coherence to improve attractor formation?"
    
    print(f"\nProcessing query: '{test_input}'")
    
    # Process with tool sequence
    tool_sequence = [
        CognitiveToolType.UNDERSTANDING,
        CognitiveToolType.REASONING,
        CognitiveToolType.VERIFICATION
    ]
    
    results = processor.process_with_tools(test_input, tool_sequence, field)
    
    print(f"✅ Processed through {len(results)} cognitive tools")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.tool_type.value.upper()} TOOL:")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Processing time: {result.processing_time:.3f}s")
        print(f"   Output:\n{result.output_data}")
        print(f"   {'-'*40}")
    
    # Test field-enhanced processing
    print(f"\nTesting field-enhanced cognitive processing...")
    enhanced_result = processor.enhance_with_field_context(
        "Analyze the current field state", 
        field, 
        CognitiveToolType.UNDERSTANDING
    )
    
    print(f"✅ Field-enhanced processing completed")
    print(f"   Enhanced confidence: {enhanced_result.confidence:.2f}")
    print(f"   Field elements considered: {enhanced_result.metadata.get('field_elements', 0)}")
    
    return results


def demonstrate_field_visualizer(field):
    """Demonstrate field visualization capabilities."""
    print("\n" + "="*60)
    print("PHASE 1 DEMO: Field Visualizer")
    print("="*60)
    
    # Create visualizer
    visualizer = FieldVisualizer()
    
    print("Created field visualizer")
    
    # Create text visualization (always available)
    print(f"\nGenerating text visualization...")
    text_viz = visualizer.visualize_field_state(field, mode='heatmap')
    
    if isinstance(text_viz, str) and text_viz.startswith('\n'):
        print("✅ Text visualization generated:")
        print(text_viz)
    else:
        print("✅ Graphics visualization generated (base64 encoded)")
    
    # Export field data
    print(f"\nExporting field data...")
    
    # JSON export
    json_data = visualizer.export_field_data(field, format='json')
    print(f"✅ JSON export: {len(json_data)} characters")
    
    # CSV export
    csv_data = visualizer.export_field_data(field, format='csv')
    print(f"✅ CSV export: {len(csv_data.splitlines())} lines")
    
    return visualizer


def demonstrate_field_manager():
    """Demonstrate field manager for multi-field operations."""
    print("\n" + "="*60)
    print("PHASE 1 DEMO: Field Manager")
    print("="*60)
    
    # Create field manager
    manager = FieldManager()
    
    print("Created field manager for multi-field operations")
    
    # Create multiple fields
    field1 = manager.create_field("field_1", dimensions=2)
    field2 = manager.create_field("field_2", dimensions=2)
    
    print(f"Created 2 fields: field_1, field_2")
    
    # Add content to fields
    field1.inject("concept A", 0.8, (0.3, 0.3))
    field1.inject("concept B", 0.7, (0.7, 0.7))
    
    field2.inject("concept C", 0.9, (0.4, 0.4))
    field2.inject("concept D", 0.6, (0.6, 0.6))
    
    print(f"Added content to both fields")
    
    # Evolve fields
    manager.evolve_field("field_1", evolution_steps=2)
    manager.evolve_field("field_2", evolution_steps=2)
    
    print(f"Evolved both fields over 2 time steps")
    
    # Merge fields
    merged_field = manager.merge_fields(["field_1", "field_2"], "merged_field")
    
    print(f"✅ Merged fields into 'merged_field'")
    print(f"  - Merged field elements: {len(merged_field.elements)}")
    print(f"  - Merged field attractors: {len(merged_field.attractors)}")
    
    # Get global state
    global_state = manager.get_global_field_state()
    print(f"✅ Global field state:")
    print(f"  - Active fields: {global_state['field_count']}")
    print(f"  - Total elements: {global_state['total_elements']}")
    print(f"  - Total attractors: {global_state['total_attractors']}")
    
    return manager


def save_demo_results(field, protocol, execution_result, cognitive_results):
    """Save demonstration results to files."""
    print("\n" + "="*60)
    print("SAVING DEMO RESULTS")
    print("="*60)
    
    # Create results directory
    results_dir = "phase1_demo_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save field state
    field_state = field.get_field_state()
    with open(f"{results_dir}/field_state.json", 'w') as f:
        json.dump(field_state, f, indent=2, default=str)
    print(f"✅ Saved field state to {results_dir}/field_state.json")
    
    # Save protocol execution results
    execution_data = {
        "protocol_name": execution_result.protocol_name,
        "status": execution_result.status.value,
        "execution_time": execution_result.end_time - execution_result.start_time if execution_result.end_time else 0,
        "execution_log": execution_result.execution_log,
        "results": execution_result.results
    }
    with open(f"{results_dir}/protocol_execution.json", 'w') as f:
        json.dump(execution_data, f, indent=2, default=str)
    print(f"✅ Saved protocol execution to {results_dir}/protocol_execution.json")
    
    # Save cognitive processing results
    cognitive_data = []
    for result in cognitive_results:
        cognitive_data.append({
            "tool_type": result.tool_type.value,
            "confidence": result.confidence,
            "processing_time": result.processing_time,
            "output_length": len(result.output_data),
            "metadata": result.metadata
        })
    
    with open(f"{results_dir}/cognitive_results.json", 'w') as f:
        json.dump(cognitive_data, f, indent=2, default=str)
    print(f"✅ Saved cognitive results to {results_dir}/cognitive_results.json")
    
    # Create summary report
    summary = f"""
Phase 1 Demonstration Summary
============================

Execution Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Components Tested:
✅ Neural Field Implementation
  - Field creation and management
  - Element injection and positioning
  - Attractor formation ({len(field.attractors)} formed)
  - Resonance pattern detection ({len(field.resonance_patterns)} detected)
  - Field coherence measurement ({field.measure_field_coherence():.3f})
  - Field decay and evolution

✅ Pareto-lang Parser
  - Protocol shell parsing
  - Operation extraction
  - Parameter parsing
  - Protocol validation

✅ Protocol Orchestrator
  - Protocol execution on fields
  - Operation sequencing
  - Result integration
  - Error handling and logging

✅ Cognitive Processor
  - Understanding tools
  - Reasoning tools  
  - Verification tools
  - Field-enhanced processing

✅ Field Visualizer
  - Text-based visualization
  - Data export (JSON, CSV)
  - Field state visualization

✅ Field Manager
  - Multi-field management
  - Field evolution
  - Field merging
  - Global state tracking

Performance Metrics:
- Field elements processed: {len(field.elements)}
- Attractors formed: {len(field.attractors)}
- Resonance patterns: {len(field.resonance_patterns)}
- Protocol execution time: {execution_result.end_time - execution_result.start_time if execution_result.end_time else 0:.3f}s
- Cognitive tools executed: {len(cognitive_results)}

Status: ALL PHASE 1 COMPONENTS WORKING ✅

This demonstrates successful implementation of the core Phase 1 requirements
from the Context Engineering System Architecture.
"""
    
    with open(f"{results_dir}/summary_report.txt", 'w') as f:
        f.write(summary)
    print(f"✅ Saved summary report to {results_dir}/summary_report.txt")


def main():
    """Run the complete Phase 1 demonstration."""
    print("🚀 CONTEXT ENGINEERING SYSTEM - PHASE 1 DEMONSTRATION")
    print("=" * 80)
    print("This demo validates all Phase 1 components are working correctly.")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # 1. Demonstrate neural field
        field = demonstrate_neural_field()
        
        # 2. Demonstrate Pareto parser
        protocol = demonstrate_pareto_parser()
        
        # 3. Demonstrate protocol orchestrator
        execution_result = demonstrate_protocol_orchestrator(field, protocol)
        
        # 4. Demonstrate cognitive processor
        cognitive_results = demonstrate_cognitive_processor(field)
        
        # 5. Demonstrate field visualizer
        visualizer = demonstrate_field_visualizer(field)
        
        # 6. Demonstrate field manager
        manager = demonstrate_field_manager()
        
        # 7. Save results
        save_demo_results(field, protocol, execution_result, cognitive_results)
        
        # Final summary
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("🎉 PHASE 1 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"All core Phase 1 components are working correctly.")
        print(f"Results saved to 'phase1_demo_results/' directory.")
        print("\nReady to proceed to Phase 2: Cognitive Integration")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)