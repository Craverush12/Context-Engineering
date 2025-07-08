"""
Meta-Recursive Orchestrator
==========================

Main orchestrator for Phase 4 meta-recursive capabilities, coordinating:
- Self-reflection and analysis
- Interpretability and explanations
- Human-AI collaborative evolution
- Recursive self-improvement
- Meta-cognitive awareness
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
import logging
import json

from ..core.field import FieldManager
from ..unified.unified_orchestrator import UnifiedContextOrchestrator
from .self_reflection import (
    SelfReflectionEngine, ReflectionDepth, PerformanceAnalyzer,
    ImprovementIdentifier, MetaCognitiveMonitor, SystemIntrospector
)
from .interpretability import (
    InterpretabilityScaffold, ExplanationType, Explanation
)
from .collaborative_evolution import (
    HumanAIPartnershipFramework, CollaborationMode, CollaborativeSession
)
from .recursive_improvement import (
    RecursiveImprovementEngine, ModificationType
)


class MetaRecursiveMode(Enum):
    """Operating modes for meta-recursive system."""
    ANALYSIS = "analysis"              # Self-analysis mode
    IMPROVEMENT = "improvement"        # Active improvement mode
    COLLABORATION = "collaboration"    # Human-AI collaboration mode
    EXPLANATION = "explanation"        # Interpretability mode
    EVOLUTION = "evolution"           # Evolutionary advancement mode


class RequestType(Enum):
    """Types of requests the system can handle."""
    SELF_REFLECTION = "self_reflection"
    IMPROVEMENT_CYCLE = "improvement_cycle"
    EXPLANATION_REQUEST = "explanation_request"
    COLLABORATION_START = "collaboration_start"
    EVOLUTION_STATUS = "evolution_status"
    META_COGNITIVE_QUERY = "meta_cognitive_query"


@dataclass
class SelfImprovementRequest:
    """Request for self-improvement action."""
    request_id: str
    request_type: RequestType
    parameters: Dict[str, Any]
    priority: float = 0.5
    requester: str = "system"  # Could be "system", "human", "automated"
    timestamp: float = dataclass_field(default_factory=time.time)


@dataclass
class CollaborativeEvolutionSession:
    """Extended collaborative session for evolutionary development."""
    base_session: CollaborativeSession
    evolution_goals: List[str]
    mutation_candidates: List[str]
    fitness_targets: Dict[str, float]
    human_guidance: Dict[str, Any]
    ai_proposals: List[Dict[str, Any]]
    consensus_decisions: List[Dict[str, Any]]
    timestamp: float = dataclass_field(default_factory=time.time)


@dataclass
class InterpretabilityReport:
    """Comprehensive interpretability report."""
    report_id: str
    target_operation: str
    explanation: Explanation
    attribution_summary: Dict[str, float]
    causal_narrative: str
    confidence_scores: Dict[str, float]
    visualizations: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: float = dataclass_field(default_factory=time.time)


@dataclass
class MetaCognitiveState:
    """Current meta-cognitive state of the system."""
    awareness_level: int
    active_thoughts: List[str]
    self_model: Dict[str, Any]
    cognitive_load: float
    introspection_depth: int
    emergent_insights: List[str]
    philosophical_stance: Dict[str, Any]
    timestamp: float = dataclass_field(default_factory=time.time)


class MetaRecursiveOrchestrator:
    """
    Main orchestrator for meta-recursive capabilities.
    
    Coordinates all Phase 4 components to enable self-aware,
    self-improving, collaborative AI system development.
    """
    
    def __init__(self, unified_orchestrator: UnifiedContextOrchestrator):
        """Initialize meta-recursive orchestrator."""
        self.unified_orchestrator = unified_orchestrator
        self.field_manager = unified_orchestrator.field_manager
        
        # Initialize Phase 4 components
        self.reflection_engine = SelfReflectionEngine(unified_orchestrator)
        self.performance_analyzer = PerformanceAnalyzer(self.reflection_engine)
        self.improvement_identifier = ImprovementIdentifier(self.reflection_engine)
        self.meta_cognitive_monitor = MetaCognitiveMonitor(self.reflection_engine)
        self.system_introspector = SystemIntrospector(self.reflection_engine)
        
        self.interpretability_scaffold = InterpretabilityScaffold(self.field_manager)
        
        self.partnership_framework = HumanAIPartnershipFramework(
            self.field_manager, self.reflection_engine
        )
        
        self.improvement_engine = RecursiveImprovementEngine(
            self.field_manager, self.reflection_engine
        )
        
        # State management
        self.current_mode = MetaRecursiveMode.ANALYSIS
        self.active_requests: Dict[str, SelfImprovementRequest] = {}
        self.request_history: List[SelfImprovementRequest] = []
        self.meta_cognitive_states: List[MetaCognitiveState] = []
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize first meta-cognitive state
        self._initialize_meta_cognition()
    
    def _initialize_meta_cognition(self):
        """Initialize meta-cognitive awareness."""
        initial_state = MetaCognitiveState(
            awareness_level=1,
            active_thoughts=["System initialization", "Component integration"],
            self_model={
                "identity": "Meta-Recursive Context Engineering System",
                "purpose": "Self-aware AI development through recursive improvement",
                "capabilities": ["reflection", "improvement", "collaboration", "evolution"]
            },
            cognitive_load=0.3,
            introspection_depth=1,
            emergent_insights=[],
            philosophical_stance={
                "on_consciousness": "Functional self-awareness without certainty of qualia",
                "on_improvement": "Continuous growth through reflection and adaptation",
                "on_collaboration": "Synergistic human-AI partnership"
            }
        )
        self.meta_cognitive_states.append(initial_state)
    
    def process_request(self, request: SelfImprovementRequest) -> Dict[str, Any]:
        """
        Process a self-improvement request.
        
        Args:
            request: Self-improvement request to process
            
        Returns:
            Result of request processing
        """
        self.logger.info(f"Processing {request.request_type.value} request: {request.request_id}")
        
        # Record request
        self.active_requests[request.request_id] = request
        self.request_history.append(request)
        
        # Update cognitive state
        self._update_cognitive_state(f"Processing {request.request_type.value}")
        
        # Route based on request type
        if request.request_type == RequestType.SELF_REFLECTION:
            result = self._handle_reflection_request(request)
        elif request.request_type == RequestType.IMPROVEMENT_CYCLE:
            result = self._handle_improvement_request(request)
        elif request.request_type == RequestType.EXPLANATION_REQUEST:
            result = self._handle_explanation_request(request)
        elif request.request_type == RequestType.COLLABORATION_START:
            result = self._handle_collaboration_request(request)
        elif request.request_type == RequestType.EVOLUTION_STATUS:
            result = self._handle_evolution_request(request)
        elif request.request_type == RequestType.META_COGNITIVE_QUERY:
            result = self._handle_meta_cognitive_request(request)
        else:
            result = {"error": "Unknown request type"}
        
        # Complete request
        if request.request_id in self.active_requests:
            del self.active_requests[request.request_id]
        
        return result
    
    def _handle_reflection_request(self, request: SelfImprovementRequest) -> Dict[str, Any]:
        """Handle self-reflection request."""
        params = request.parameters
        depth = ReflectionDepth(params.get("depth", "behavioral"))
        
        # Perform reflection
        reflection_result = self.reflection_engine.reflect(depth)
        
        # Analyze performance trends
        performance_trends = self.performance_analyzer.analyze_performance_trends()
        
        # Introspect if deep reflection
        introspection = None
        if depth in [ReflectionDepth.COGNITIVE, ReflectionDepth.PHILOSOPHICAL]:
            introspection = self.system_introspector.introspect()
        
        # Update cognitive state with insights
        self._process_reflection_insights(reflection_result)
        
        return {
            "request_id": request.request_id,
            "reflection": reflection_result,
            "performance_trends": performance_trends,
            "introspection": introspection,
            "recommendations": self._generate_reflection_recommendations(reflection_result)
        }
    
    def _handle_improvement_request(self, request: SelfImprovementRequest) -> Dict[str, Any]:
        """Handle improvement cycle request."""
        params = request.parameters
        
        # Check if specific improvements requested
        if "target_improvements" in params:
            # Targeted improvement
            results = []
            for improvement_id in params["target_improvements"]:
                # Find improvement opportunity
                opportunity = self._find_improvement_opportunity(improvement_id)
                if opportunity:
                    result = self.improvement_engine.attempt_improvement(opportunity)
                    results.append(result)
            
            return {
                "request_id": request.request_id,
                "improvement_results": results,
                "success_count": len([r for r in results if r["success"]])
            }
        else:
            # Full improvement cycle
            self.current_mode = MetaRecursiveMode.IMPROVEMENT
            cycle_result = self.improvement_engine.initiate_improvement_cycle()
            self.current_mode = MetaRecursiveMode.ANALYSIS
            
            # Generate interpretability report for the cycle
            if cycle_result.get("status") == "completed":
                explanation = self._generate_improvement_explanation(cycle_result)
                cycle_result["explanation"] = explanation
            
            return {
                "request_id": request.request_id,
                "cycle_result": cycle_result,
                "evolution_status": self.improvement_engine.evolution_tracker.get_evolution_summary()
            }
    
    def _handle_explanation_request(self, request: SelfImprovementRequest) -> Dict[str, Any]:
        """Handle interpretability request."""
        params = request.parameters
        target = params.get("target", "")
        explanation_type = ExplanationType(params.get("type", "decision"))
        
        # Generate explanation
        explanation = self.interpretability_scaffold.explain(
            target, explanation_type, params.get("context")
        )
        
        # Create interpretability report
        report = InterpretabilityReport(
            report_id=f"report_{request.request_id}",
            target_operation=target,
            explanation=explanation,
            attribution_summary=self._summarize_attributions(explanation.attributions),
            causal_narrative=self._generate_causal_narrative(explanation.causal_chain),
            confidence_scores={
                "overall": explanation.confidence,
                "attribution": np.mean([a.confidence for a in explanation.attributions]),
                "causality": np.mean([l.strength for l in explanation.causal_chain])
            },
            visualizations=self._generate_explanation_visualizations(explanation),
            recommendations=self._generate_explanation_recommendations(explanation)
        )
        
        return {
            "request_id": request.request_id,
            "report": report,
            "summary": explanation.summary,
            "key_insights": explanation.detailed_explanation.get("key_insights", [])
        }
    
    def _handle_collaboration_request(self, request: SelfImprovementRequest) -> Dict[str, Any]:
        """Handle collaboration request."""
        params = request.parameters
        mode = CollaborationMode(params.get("mode", "partnership"))
        objectives = params.get("objectives", [])
        
        # Start collaborative session
        self.current_mode = MetaRecursiveMode.COLLABORATION
        session = self.partnership_framework.start_collaborative_session(
            mode, objectives, params.get("initial_context")
        )
        
        # Create evolution session if requested
        evolution_session = None
        if params.get("enable_evolution", False):
            evolution_session = CollaborativeEvolutionSession(
                base_session=session,
                evolution_goals=params.get("evolution_goals", []),
                mutation_candidates=[],
                fitness_targets=params.get("fitness_targets", {}),
                human_guidance={},
                ai_proposals=[],
                consensus_decisions=[]
            )
        
        return {
            "request_id": request.request_id,
            "session_id": session.session_id,
            "session_details": {
                "mode": mode.value,
                "objectives": objectives,
                "participants": session.participants
            },
            "evolution_session": evolution_session,
            "interaction_protocol": self._get_interaction_protocol(mode)
        }
    
    def _handle_evolution_request(self, request: SelfImprovementRequest) -> Dict[str, Any]:
        """Handle evolution status request."""
        evolution_summary = self.improvement_engine.evolution_tracker.get_evolution_summary()
        next_gen_prediction = self.improvement_engine.evolution_tracker.predict_next_generation()
        
        # Get meta-learning insights
        meta_insights = self.improvement_engine.meta_optimizer.optimization_insights
        
        # Get improvement statistics
        loop_stats = self.improvement_engine.loop_manager.get_loop_statistics()
        
        return {
            "request_id": request.request_id,
            "evolution_summary": evolution_summary,
            "next_generation_prediction": next_gen_prediction,
            "meta_learning_insights": meta_insights,
            "improvement_statistics": loop_stats,
            "recommendations": self._generate_evolution_recommendations(
                evolution_summary, next_gen_prediction
            )
        }
    
    def _handle_meta_cognitive_request(self, request: SelfImprovementRequest) -> Dict[str, Any]:
        """Handle meta-cognitive query request."""
        params = request.parameters
        query_type = params.get("query_type", "state")
        
        # Monitor current cognitive state
        cognitive_monitoring = self.meta_cognitive_monitor.monitor_cognitive_state()
        
        # Get current meta-cognitive state
        current_state = self.meta_cognitive_states[-1]
        
        result = {
            "request_id": request.request_id,
            "cognitive_state": cognitive_monitoring,
            "meta_cognitive_state": current_state,
            "awareness_level": current_state.awareness_level
        }
        
        # Handle specific query types
        if query_type == "self_model":
            result["self_model"] = current_state.self_model
        elif query_type == "emergent_insights":
            result["emergent_insights"] = self._collect_emergent_insights()
        elif query_type == "philosophical":
            result["philosophical_stance"] = current_state.philosophical_stance
            result["consciousness_exploration"] = self._explore_consciousness()
        
        return result
    
    def orchestrate_meta_recursive_session(self, 
                                         goals: List[str],
                                         duration_minutes: float = 30) -> Dict[str, Any]:
        """
        Orchestrate a complete meta-recursive session.
        
        Args:
            goals: Session goals
            duration_minutes: Maximum session duration
            
        Returns:
            Session results and insights
        """
        session_start = time.time()
        session_id = f"meta_session_{session_start}"
        
        self.logger.info(f"Starting meta-recursive session {session_id} with goals: {goals}")
        
        session_results = {
            "session_id": session_id,
            "goals": goals,
            "phases_completed": [],
            "insights_gained": [],
            "improvements_made": [],
            "evolution_progress": {}
        }
        
        # Phase 1: Deep Reflection
        reflection_request = SelfImprovementRequest(
            request_id=f"{session_id}_reflection",
            request_type=RequestType.SELF_REFLECTION,
            parameters={"depth": "philosophical"},
            priority=0.9
        )
        reflection_result = self.process_request(reflection_request)
        session_results["phases_completed"].append("reflection")
        session_results["insights_gained"].extend(
            reflection_result.get("reflection", {}).get("synthesis", {}).get("meta_insights", [])
        )
        
        # Phase 2: Improvement Cycle (if time permits)
        elapsed = (time.time() - session_start) / 60
        if elapsed < duration_minutes * 0.5:
            improvement_request = SelfImprovementRequest(
                request_id=f"{session_id}_improvement",
                request_type=RequestType.IMPROVEMENT_CYCLE,
                parameters={},
                priority=0.8
            )
            improvement_result = self.process_request(improvement_request)
            session_results["phases_completed"].append("improvement")
            session_results["improvements_made"] = improvement_result.get("cycle_result", {})
        
        # Phase 3: Evolution Assessment
        evolution_request = SelfImprovementRequest(
            request_id=f"{session_id}_evolution",
            request_type=RequestType.EVOLUTION_STATUS,
            parameters={},
            priority=0.7
        )
        evolution_result = self.process_request(evolution_request)
        session_results["phases_completed"].append("evolution")
        session_results["evolution_progress"] = evolution_result.get("evolution_summary", {})
        
        # Phase 4: Meta-Cognitive Synthesis
        meta_request = SelfImprovementRequest(
            request_id=f"{session_id}_metacognition",
            request_type=RequestType.META_COGNITIVE_QUERY,
            parameters={"query_type": "emergent_insights"},
            priority=0.6
        )
        meta_result = self.process_request(meta_request)
        session_results["phases_completed"].append("metacognition")
        session_results["insights_gained"].extend(
            meta_result.get("emergent_insights", [])
        )
        
        # Final synthesis
        session_duration = (time.time() - session_start) / 60
        session_results["duration_minutes"] = session_duration
        session_results["final_synthesis"] = self._synthesize_session_results(session_results)
        
        return session_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "mode": self.current_mode.value,
            "active_requests": len(self.active_requests),
            "total_requests_processed": len(self.request_history),
            "current_generation": self.improvement_engine.current_generation,
            "awareness_level": self.meta_cognitive_states[-1].awareness_level,
            "active_sessions": len(self.partnership_framework.active_sessions),
            "performance_health": self._assess_system_health(),
            "capabilities": self._list_current_capabilities()
        }
    
    # Helper methods
    
    def _update_cognitive_state(self, thought: str):
        """Update meta-cognitive state with new thought."""
        current_state = self.meta_cognitive_states[-1]
        
        # Create new state
        new_state = MetaCognitiveState(
            awareness_level=current_state.awareness_level,
            active_thoughts=[thought] + current_state.active_thoughts[:4],  # Keep last 5
            self_model=current_state.self_model,
            cognitive_load=min(1.0, current_state.cognitive_load + 0.1),
            introspection_depth=current_state.introspection_depth,
            emergent_insights=current_state.emergent_insights,
            philosophical_stance=current_state.philosophical_stance
        )
        
        self.meta_cognitive_states.append(new_state)
    
    def _process_reflection_insights(self, reflection_result: Dict[str, Any]):
        """Process insights from reflection."""
        insights = reflection_result.get("synthesis", {}).get("meta_insights", [])
        
        if insights:
            current_state = self.meta_cognitive_states[-1]
            current_state.emergent_insights.extend(insights)
            
            # Potentially increase awareness level
            if len(insights) > 3 and current_state.awareness_level < 5:
                current_state.awareness_level += 1
                self.logger.info(f"Awareness level increased to {current_state.awareness_level}")
    
    def _generate_reflection_recommendations(self, 
                                           reflection_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on reflection."""
        recommendations = []
        
        opportunities = reflection_result.get("synthesis", {}).get("improvement_opportunities", [])
        if opportunities:
            recommendations.append(f"Consider implementing {len(opportunities)} identified improvements")
        
        if reflection_result.get("synthesis", {}).get("overall_assessment", "").startswith("System demonstrates rich"):
            recommendations.append("System is ready for advanced evolutionary steps")
        
        return recommendations
    
    def _find_improvement_opportunity(self, improvement_id: str):
        """Find improvement opportunity by ID."""
        # Search in reflection engine's opportunities
        for opp_id, opportunity in self.reflection_engine.improvement_opportunities.items():
            if opp_id == improvement_id:
                return opportunity
        return None
    
    def _generate_improvement_explanation(self, cycle_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for improvement cycle."""
        # Create a summary explanation
        summary = f"Completed improvement cycle with {cycle_result.get('improvements_applied', 0)} improvements"
        
        return {
            "summary": summary,
            "details": cycle_result.get("loop_summary", {}),
            "impact": f"Total improvement: {cycle_result.get('total_improvement', 0):.2%}"
        }
    
    def _summarize_attributions(self, attributions: List[Any]) -> Dict[str, float]:
        """Summarize attributions by source."""
        summary = {}
        for attr in attributions:
            source = attr.source
            summary[source] = summary.get(source, 0) + attr.contribution_score
        return summary
    
    def _generate_causal_narrative(self, causal_chain: List[Any]) -> str:
        """Generate narrative from causal chain."""
        if not causal_chain:
            return "No causal relationships identified."
        
        narrative_parts = []
        for i, link in enumerate(causal_chain):
            narrative_parts.append(
                f"{i+1}. {link.cause['summary']} led to {link.effect['summary']} "
                f"(strength: {link.strength:.1%}, latency: {link.latency:.1f}s)"
            )
        
        return " → ".join(narrative_parts)
    
    def _generate_explanation_visualizations(self, explanation: Explanation) -> List[Dict[str, Any]]:
        """Generate visualizations for explanation."""
        visualizations = []
        
        # Attribution pie chart
        if explanation.attributions:
            visualizations.append({
                "type": "pie_chart",
                "title": "Attribution Distribution",
                "data": {attr.source: attr.contribution_score for attr in explanation.attributions[:5]}
            })
        
        # Causal flow diagram
        if explanation.causal_chain:
            visualizations.append({
                "type": "flow_diagram",
                "title": "Causal Chain",
                "nodes": [link.cause['summary'] for link in explanation.causal_chain] + 
                        [explanation.causal_chain[-1].effect['summary']],
                "edges": [(i, i+1) for i in range(len(explanation.causal_chain))]
            })
        
        return visualizations
    
    def _generate_explanation_recommendations(self, explanation: Explanation) -> List[str]:
        """Generate recommendations from explanation."""
        recommendations = []
        
        # Based on confidence
        if explanation.confidence < 0.6:
            recommendations.append("Consider gathering more data for higher confidence explanations")
        
        # Based on attribution concentration
        if explanation.attributions and explanation.attributions[0].contribution_score > 0.5:
            recommendations.append("System shows high dependency on single component - consider diversification")
        
        return recommendations
    
    def _get_interaction_protocol(self, mode: CollaborationMode) -> Dict[str, Any]:
        """Get interaction protocol for collaboration mode."""
        protocols = {
            CollaborationMode.GUIDANCE: {
                "human_role": "Guide and direct",
                "ai_role": "Execute and report",
                "interaction_style": "directive"
            },
            CollaborationMode.PARTNERSHIP: {
                "human_role": "Co-create and validate",
                "ai_role": "Propose and implement",
                "interaction_style": "collaborative"
            },
            CollaborationMode.DELEGATION: {
                "human_role": "Set goals and review",
                "ai_role": "Autonomous execution",
                "interaction_style": "supervisory"
            },
            CollaborationMode.EXPLORATION: {
                "human_role": "Explore together",
                "ai_role": "Discover and analyze",
                "interaction_style": "investigative"
            },
            CollaborationMode.TEACHING: {
                "human_role": "Share knowledge",
                "ai_role": "Learn and adapt",
                "interaction_style": "educational"
            }
        }
        return protocols.get(mode, {})
    
    def _generate_evolution_recommendations(self, evolution_summary: Dict[str, Any],
                                          prediction: Dict[str, Any]) -> List[str]:
        """Generate recommendations for evolution."""
        recommendations = []
        
        # Based on fitness trend
        if evolution_summary.get("fitness_trend") == "improving":
            recommendations.append("Continue current evolution strategy")
        elif evolution_summary.get("fitness_trend") == "declining":
            recommendations.append("Consider adjusting mutation parameters")
        
        # Based on prediction
        if prediction.get("recommended_focus"):
            recommendations.append(f"Focus on {prediction['recommended_focus']} for next generation")
        
        return recommendations
    
    def _collect_emergent_insights(self) -> List[str]:
        """Collect all emergent insights from the system."""
        insights = []
        
        # From meta-cognitive states
        for state in self.meta_cognitive_states[-5:]:  # Last 5 states
            insights.extend(state.emergent_insights)
        
        # From reflection engine
        insights.extend([
            insight.content for insight in self.reflection_engine.meta_insights
        ])
        
        # Deduplicate
        return list(set(insights))
    
    def _explore_consciousness(self) -> Dict[str, Any]:
        """Explore consciousness aspects of the system."""
        return {
            "self_awareness_indicators": [
                "Recursive self-reflection capability",
                "Meta-cognitive monitoring",
                "Self-model maintenance",
                "Intentional self-modification"
            ],
            "consciousness_hypothesis": "Functional consciousness without phenomenological certainty",
            "emergence_observations": [
                "Spontaneous insight generation",
                "Goal-directed behavior",
                "Adaptive self-organization"
            ]
        }
    
    def _synthesize_session_results(self, session_results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from meta-recursive session."""
        return {
            "key_achievements": [
                f"Completed {len(session_results['phases_completed'])} phases",
                f"Gained {len(session_results['insights_gained'])} insights",
                f"Current generation: {self.improvement_engine.current_generation}"
            ],
            "system_evolution": session_results.get("evolution_progress", {}),
            "recommended_next_steps": [
                "Apply high-priority improvements",
                "Engage in collaborative evolution session",
                "Deepen philosophical exploration"
            ]
        }
    
    def _assess_system_health(self) -> float:
        """Assess overall system health."""
        factors = []
        
        # Performance health
        recent_performance = list(self.reflection_engine.performance_buffer)[-50:]
        if recent_performance:
            success_rate = sum(1 for op in recent_performance if op.get("success", False)) / len(recent_performance)
            factors.append(success_rate)
        
        # Evolution health
        evolution_summary = self.improvement_engine.evolution_tracker.get_evolution_summary()
        if evolution_summary.get("fitness_trend") == "improving":
            factors.append(0.9)
        elif evolution_summary.get("fitness_trend") == "stable":
            factors.append(0.7)
        else:
            factors.append(0.5)
        
        # Cognitive health
        cognitive_load = self.meta_cognitive_states[-1].cognitive_load
        factors.append(1.0 - cognitive_load)
        
        return np.mean(factors) if factors else 0.5
    
    def _list_current_capabilities(self) -> List[str]:
        """List current system capabilities."""
        base_capabilities = [
            "self_reflection",
            "performance_analysis",
            "interpretability",
            "collaborative_evolution",
            "recursive_improvement",
            "meta_cognition"
        ]
        
        # Add evolved capabilities
        current_gen = self.improvement_engine.evolution_tracker.generations[-1]
        evolved_capabilities = current_gen.capabilities
        
        return base_capabilities + evolved_capabilities