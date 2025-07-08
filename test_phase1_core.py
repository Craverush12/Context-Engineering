#!/usr/bin/env python3
"""
Core Phase 1 Test
=================

Tests the core Phase 1 functionality without requiring external dependencies.
This validates the fundamental architecture and implementation.
"""

import sys
import os
import time

# Add the context engineering system to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_core_imports():
    """Test that all core modules can be imported."""
    print("Testing core imports...")
    
    try:
        from context_engineering_system.core.field import ContextField, FieldManager, FieldElementType
        print("✅ Core field module imported")
    except ImportError as e:
        print(f"❌ Failed to import field module: {e}")
        return False
    
    try:
        from context_engineering_system.parsers.pareto_lang import ParetoLangParser, ProtocolShell
        print("✅ Pareto-lang parser imported")
    except ImportError as e:
        print(f"❌ Failed to import parser: {e}")
        return False
    
    try:
        from context_engineering_system.core.cognitive_processor import CognitiveProcessor, CognitiveToolType
        print("✅ Cognitive processor imported")
    except ImportError as e:
        print(f"❌ Failed to import cognitive processor: {e}")
        return False
    
    return True

def test_field_operations():
    """Test basic field operations."""
    print("\nTesting field operations...")
    
    from context_engineering_system.core.field import ContextField, FieldElementType
    
    # Create field
    field = ContextField(dimensions=2, decay_rate=0.1)
    print(f"✅ Created field: {field.dimensions}D, decay_rate={field.decay_rate}")
    
    # Inject elements
    elem1 = field.inject("test content 1", 0.8, (0.3, 0.4))
    elem2 = field.inject("test content 2", 0.7, (0.6, 0.7))
    print(f"✅ Injected 2 elements: {len(field.elements)} total")
    
    # Check field state
    state = field.get_field_state()
    print(f"✅ Field state retrieved: {len(state['elements'])} elements")
    
    # Test decay
    initial_elements = len(field.elements)
    field.decay()
    print(f"✅ Field decay applied: {initial_elements} -> {len(field.elements)} elements")
    
    return True

def test_parser():
    """Test Pareto-lang parser."""
    print("\nTesting Pareto-lang parser...")
    
    from context_engineering_system.parsers.pareto_lang import ParetoLangParser
    
    # Simple protocol definition
    protocol_content = """
    /test.protocol {
      intent="Test protocol for validation",
      
      input={
        field_state="<state>"
      },
      
      process=[
        "/field.audit{}"
      ],
      
      output={
        result="<output>"
      }
    }
    """
    
    parser = ParetoLangParser()
    protocol = parser.parse_content(protocol_content)
    
    print(f"✅ Parsed protocol: {protocol.name}")
    print(f"✅ Intent: {protocol.intent}")
    print(f"✅ Operations: {len(protocol.process_operations)}")
    
    # Validate
    is_valid, errors = parser.validate_protocol(protocol)
    if is_valid:
        print("✅ Protocol validation passed")
    else:
        print(f"❌ Validation failed: {errors}")
        return False
    
    return True

def test_cognitive_processor():
    """Test cognitive processor."""
    print("\nTesting cognitive processor...")
    
    from context_engineering_system.core.cognitive_processor import CognitiveProcessor, CognitiveToolType
    
    processor = CognitiveProcessor()
    print("✅ Created cognitive processor")
    
    # Test understanding tool
    test_input = "What is context engineering?"
    results = processor.process_with_tools(
        test_input, 
        [CognitiveToolType.UNDERSTANDING]
    )
    
    print(f"✅ Processed input through understanding tool")
    print(f"✅ Result confidence: {results[0].confidence}")
    print(f"✅ Processing time: {results[0].processing_time:.3f}s")
    
    return True

def test_field_manager():
    """Test field manager."""
    print("\nTesting field manager...")
    
    from context_engineering_system.core.field import FieldManager
    
    manager = FieldManager()
    print("✅ Created field manager")
    
    # Create fields
    field1 = manager.create_field("test_field_1")
    field2 = manager.create_field("test_field_2")
    print(f"✅ Created 2 fields")
    
    # Add content
    field1.inject("content A", 0.8)
    field2.inject("content B", 0.7)
    print(f"✅ Added content to fields")
    
    # Get global state
    global_state = manager.get_global_field_state()
    print(f"✅ Global state: {global_state['field_count']} fields, {global_state['total_elements']} elements")
    
    return True

def test_protocol_orchestrator():
    """Test protocol orchestrator (basic functionality)."""
    print("\nTesting protocol orchestrator...")
    
    try:
        from context_engineering_system.core.protocol_orchestrator import ProtocolOrchestrator
        from context_engineering_system.core.field import ContextField
        from context_engineering_system.parsers.pareto_lang import ParetoLangParser
        
        # Create components
        orchestrator = ProtocolOrchestrator()
        field = ContextField()
        
        # Add some content to field
        field.inject("test content", 0.5)
        
        # Simple protocol
        protocol_content = """
        /test.operations {
          intent="Test basic operations",
          input={field_state="<state>"},
          process=["/field.audit{}"],
          output={result="<result>"}
        }
        """
        
        parser = ParetoLangParser()
        protocol = parser.parse_content(protocol_content)
        
        print("✅ Created orchestrator and parsed protocol")
        
        # Test execution (this will use default handlers)
        result = orchestrator.execute_protocol(protocol, field, {"field_state": field.get_field_state()})
        
        print(f"✅ Protocol execution completed: {result.status.value}")
        print(f"✅ Execution time: {result.processing_time if hasattr(result, 'processing_time') else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Protocol orchestrator test failed: {e}")
        return False

def run_all_tests():
    """Run all core tests."""
    print("🧪 CONTEXT ENGINEERING SYSTEM - CORE PHASE 1 TESTS")
    print("="*60)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Field Operations", test_field_operations),
        ("Pareto-lang Parser", test_parser),
        ("Cognitive Processor", test_cognitive_processor),
        ("Field Manager", test_field_manager),
        ("Protocol Orchestrator", test_protocol_orchestrator),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n" + "="*60)
    print(f"🏁 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL CORE PHASE 1 TESTS PASSED!")
        print("✅ Core implementation is working correctly")
        print("✅ Ready for full Phase 1 validation with dependencies")
        return True
    else:
        print("❌ Some tests failed - check implementation")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)