#!/usr/bin/env python3
"""
Phase 4 Integration Tests
========================

Tests the integration of all Phase 4 meta-recursive components.
"""

import unittest
import time
from typing import Dict, Any

from context_engineering_system.core.field import FieldManager
from context_engineering_system.core.protocol_orchestrator import ProtocolOrchestrator
from context_engineering_system.core.cognitive_processor import CognitiveProcessor
from context_engineering_system.unified.unified_orchestrator import UnifiedContextOrchestrator

from context_engineering_system.meta_recursive.meta_recursive_orchestrator import (
    MetaRecursiveOrchestrator, SelfImprovementRequest, RequestType,
    CollaborationMode, MetaCognitiveState
)
from context_engineering_system.meta_recursive.self_reflection import (
    ReflectionDepth, SelfReflectionEngine
)
from context_engineering_system.meta_recursive.interpretability import (
    InterpretabilityScaffold, ExplanationType
)
from context_engineering_system.meta_recursive.collaborative_evolution import (
    HumanAIPartnershipFramework
)
from context_engineering_system.meta_recursive.recursive_improvement import (
    RecursiveImprovementEngine
)


class TestPhase4Integration(unittest.TestCase):
    """Integration tests for Phase 4 meta-recursive capabilities."""
    
    def setUp(self):
        """Set up test environment."""
        # Create base components
        self.field_manager = FieldManager()
        self.protocol_orchestrator = ProtocolOrchestrator()
        self.cognitive_processor = CognitiveProcessor(
            self.field_manager, self.protocol_orchestrator
        )
        
        # Create unified orchestrator
        self.unified_orchestrator = UnifiedContextOrchestrator(
            self.field_manager,
            self.protocol_orchestrator,
            self.cognitive_processor
        )
        
        # Create meta-recursive orchestrator
        self.meta_orchestrator = MetaRecursiveOrchestrator(self.unified_orchestrator)
    
    def test_self_reflection_integration(self):
        """Test self-reflection capabilities integration."""
        # Test behavioral reflection
        request = SelfImprovementRequest(
            request_id="test_reflection_1",
            request_type=RequestType.SELF_REFLECTION,
            parameters={"depth": "behavioral"}
        )
        
        result = self.meta_orchestrator.process_request(request)
        
        self.assertIn("reflection", result)
        self.assertIn("performance_trends", result)
        self.assertIn("recommendations", result)
        
        # Test philosophical reflection
        request = SelfImprovementRequest(
            request_id="test_reflection_2",
            request_type=RequestType.SELF_REFLECTION,
            parameters={"depth": "philosophical"}
        )
        
        result = self.meta_orchestrator.process_request(request)
        
        self.assertIn("introspection", result)
        introspection = result.get("introspection", {})
        self.assertIn("self_awareness", introspection)
        self.assertIn("philosophical_insights", introspection)
    
    def test_interpretability_integration(self):
        """Test interpretability scaffolding integration."""
        # Generate some activity first
        self.unified_orchestrator.process_request({
            "type": "cognitive",
            "content": "test input",
            "require_understanding": True
        })
        
        # Request explanation
        request = SelfImprovementRequest(
            request_id="test_explain_1",
            request_type=RequestType.EXPLANATION_REQUEST,
            parameters={
                "target": "cognitive_processing",
                "type": "process",
                "context": {"operation": "understanding"}
            }
        )
        
        result = self.meta_orchestrator.process_request(request)
        
        self.assertIn("report", result)
        self.assertIn("summary", result)
        
        report = result["report"]
        self.assertIsNotNone(report.explanation)
        self.assertIsNotNone(report.confidence_scores)
        self.assertIsInstance(report.visualizations, list)
    
    def test_collaborative_evolution_integration(self):
        """Test human-AI collaboration integration."""
        # Start collaboration session
        request = SelfImprovementRequest(
            request_id="test_collab_1",
            request_type=RequestType.COLLABORATION_START,
            parameters={
                "mode": "partnership",
                "objectives": ["Test collaboration"],
                "enable_evolution": True
            }
        )
        
        result = self.meta_orchestrator.process_request(request)
        
        self.assertIn("session_id", result)
        self.assertIn("session_details", result)
        self.assertIn("interaction_protocol", result)
        
        session_id = result["session_id"]
        
        # Test interaction
        if session_id:
            interaction_result = self.meta_orchestrator.partnership_framework.interact(
                session_id,
                "feedback",
                {
                    "participant": "human",
                    "feedback": {"type": "positive", "content": "Good progress"}
                }
            )
            
            self.assertIn("status", interaction_result)
            self.assertEqual(interaction_result["status"], "feedback_received")
    
    def test_recursive_improvement_integration(self):
        """Test recursive self-improvement integration."""
        # Generate some performance data
        for i in range(5):
            self.meta_orchestrator.reflection_engine.record_performance({
                "operation": f"test_op_{i}",
                "success": i % 2 == 0,
                "duration": 0.1 + i * 0.01,
                "resource_usage": {"cpu": 0.1, "memory": 100}
            })
        
        # Request improvement cycle
        request = SelfImprovementRequest(
            request_id="test_improve_1",
            request_type=RequestType.IMPROVEMENT_CYCLE,
            parameters={}
        )
        
        result = self.meta_orchestrator.process_request(request)
        
        self.assertIn("cycle_result", result)
        self.assertIn("evolution_status", result)
        
        cycle = result["cycle_result"]
        self.assertIn("status", cycle)
        
        # Check evolution status
        evolution = result["evolution_status"]
        self.assertIn("current_generation", evolution)
        self.assertIn("fitness_trend", evolution)
    
    def test_meta_cognitive_awareness_integration(self):
        """Test meta-cognitive awareness integration."""
        # Query meta-cognitive state
        request = SelfImprovementRequest(
            request_id="test_meta_1",
            request_type=RequestType.META_COGNITIVE_QUERY,
            parameters={"query_type": "state"}
        )
        
        result = self.meta_orchestrator.process_request(request)
        
        self.assertIn("meta_cognitive_state", result)
        self.assertIn("awareness_level", result)
        
        state = result["meta_cognitive_state"]
        self.assertIsInstance(state, MetaCognitiveState)
        self.assertGreater(state.awareness_level, 0)
        self.assertIsInstance(state.active_thoughts, list)
        
        # Query philosophical stance
        request = SelfImprovementRequest(
            request_id="test_meta_2",
            request_type=RequestType.META_COGNITIVE_QUERY,
            parameters={"query_type": "philosophical"}
        )
        
        result = self.meta_orchestrator.process_request(request)
        
        self.assertIn("philosophical_stance", result)
        self.assertIn("consciousness_exploration", result)
    
    def test_full_session_orchestration(self):
        """Test complete meta-recursive session orchestration."""
        session_result = self.meta_orchestrator.orchestrate_meta_recursive_session(
            goals=["Test session"],
            duration_minutes=0.1  # Very short for testing
        )
        
        self.assertIn("session_id", session_result)
        self.assertIn("phases_completed", session_result)
        self.assertIn("final_synthesis", session_result)
        
        # Check that at least some phases completed
        self.assertGreater(len(session_result["phases_completed"]), 0)
        
        # Check final synthesis
        synthesis = session_result["final_synthesis"]
        self.assertIn("key_achievements", synthesis)
        self.assertIn("recommended_next_steps", synthesis)
    
    def test_system_status_integration(self):
        """Test system status reporting."""
        status = self.meta_orchestrator.get_system_status()
        
        self.assertIn("mode", status)
        self.assertIn("current_generation", status)
        self.assertIn("awareness_level", status)
        self.assertIn("performance_health", status)
        self.assertIn("capabilities", status)
        
        # Check capabilities list
        capabilities = status["capabilities"]
        self.assertIsInstance(capabilities, list)
        self.assertIn("self_reflection", capabilities)
        self.assertIn("meta_cognition", capabilities)
    
    def test_component_interactions(self):
        """Test interactions between Phase 4 components."""
        # Reflection should feed into improvement identification
        reflection_result = self.meta_orchestrator.reflection_engine.reflect(
            ReflectionDepth.BEHAVIORAL
        )
        
        opportunities = reflection_result.get("synthesis", {}).get(
            "improvement_opportunities", []
        )
        
        # If opportunities found, they should be processable
        if opportunities:
            opportunity = opportunities[0]
            result = self.meta_orchestrator.improvement_engine.attempt_improvement(
                opportunity
            )
            self.assertIn("success", result)
        
        # Interpretability should work with all components
        explanation = self.meta_orchestrator.interpretability_scaffold.explain(
            "self_reflection",
            ExplanationType.PROCESS
        )
        
        self.assertIsNotNone(explanation)
        self.assertGreater(explanation.confidence, 0)
    
    def test_safety_mechanisms(self):
        """Test safety mechanisms in self-improvement."""
        engine = self.meta_orchestrator.improvement_engine
        
        # Check safety parameters
        self.assertLessEqual(engine.max_modification_magnitude, 0.5)
        self.assertTrue(engine.safety_mode)
        
        # Validator should have safety constraints
        validator = engine.validator
        constraints = validator.safety_constraints
        
        self.assertIn("max_performance_degradation", constraints)
        self.assertIn("prohibited_components", constraints)
        self.assertIsInstance(constraints["prohibited_components"], list)
    
    def test_evolutionary_progression(self):
        """Test evolutionary tracking and progression."""
        tracker = self.meta_orchestrator.improvement_engine.evolution_tracker
        
        # Check initial generation
        self.assertEqual(len(tracker.generations), 1)
        self.assertEqual(tracker.generations[0].generation, 1)
        
        # Get evolution summary
        summary = tracker.get_evolution_summary()
        
        self.assertIn("current_generation", summary)
        self.assertIn("fitness_trend", summary)
        self.assertIn("current_capabilities", summary)
        
        # Test prediction
        prediction = tracker.predict_next_generation()
        
        if prediction.get("status") != "insufficient_history":
            self.assertIn("predicted_fitness", prediction)
            self.assertIn("recommended_focus", prediction)


class TestPhase4Performance(unittest.TestCase):
    """Performance tests for Phase 4 components."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.field_manager = FieldManager()
        self.protocol_orchestrator = ProtocolOrchestrator()
        self.cognitive_processor = CognitiveProcessor(
            self.field_manager, self.protocol_orchestrator
        )
        
        self.unified_orchestrator = UnifiedContextOrchestrator(
            self.field_manager,
            self.protocol_orchestrator,
            self.cognitive_processor
        )
        
        self.meta_orchestrator = MetaRecursiveOrchestrator(self.unified_orchestrator)
    
    def test_reflection_performance(self):
        """Test performance of reflection operations."""
        start_time = time.time()
        
        # Perform multiple reflections
        for depth in ["behavioral", "cognitive", "philosophical"]:
            request = SelfImprovementRequest(
                request_id=f"perf_reflect_{depth}",
                request_type=RequestType.SELF_REFLECTION,
                parameters={"depth": depth}
            )
            
            self.meta_orchestrator.process_request(request)
        
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time
        self.assertLess(elapsed, 5.0, "Reflection taking too long")
    
    def test_improvement_cycle_performance(self):
        """Test performance of improvement cycles."""
        # Generate performance data
        for i in range(20):
            self.meta_orchestrator.reflection_engine.record_performance({
                "operation": f"perf_op_{i}",
                "success": True,
                "duration": 0.05,
                "resource_usage": {"cpu": 0.1, "memory": 100}
            })
        
        start_time = time.time()
        
        # Run improvement cycle
        request = SelfImprovementRequest(
            request_id="perf_improve",
            request_type=RequestType.IMPROVEMENT_CYCLE,
            parameters={}
        )
        
        result = self.meta_orchestrator.process_request(request)
        
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time
        self.assertLess(elapsed, 10.0, "Improvement cycle taking too long")
    
    def test_explanation_generation_performance(self):
        """Test performance of explanation generation."""
        start_time = time.time()
        
        # Generate multiple explanations
        for i in range(5):
            request = SelfImprovementRequest(
                request_id=f"perf_explain_{i}",
                request_type=RequestType.EXPLANATION_REQUEST,
                parameters={
                    "target": f"operation_{i}",
                    "type": "decision"
                }
            )
            
            self.meta_orchestrator.process_request(request)
        
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time
        self.assertLess(elapsed, 3.0, "Explanation generation taking too long")


def run_integration_tests():
    """Run all Phase 4 integration tests."""
    print("\n🧪 Running Phase 4 Integration Tests...\n")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add integration tests
    suite.addTest(unittest.makeSuite(TestPhase4Integration))
    suite.addTest(unittest.makeSuite(TestPhase4Performance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    print(f"{'='*60}\n")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)