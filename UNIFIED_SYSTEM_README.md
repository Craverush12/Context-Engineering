# Unified Context Engineering System

## 🌟 Overview

This unified implementation progressively demonstrates all four phases of the Context Engineering System:

1. **Phase 1**: Neural Fields & Protocol Foundation
2. **Phase 2**: Cognitive Integration with LLM Enhancement (16.6% reasoning improvement)
3. **Phase 3**: Unified Field Operations with System Intelligence
4. **Phase 4**: Meta-Recursive Capabilities with Self-Improvement

The system integrates with OpenAI, Google Gemini, and Groq for enhanced AI capabilities.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
python setup_and_run.py
```

This interactive script will:
- Check Python version (3.8+ required)
- Install core dependencies
- Ask which LLM providers to install
- Help configure API keys
- Run the unified system demonstration

### Option 2: Manual Setup

1. **Install Core Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install LLM Providers** (optional, as needed)
   ```bash
   # For OpenAI
   pip install openai>=1.0.0
   
   # For Google Gemini
   pip install google-generativeai>=0.3.0
   
   # For Groq
   pip install groq>=0.4.0
   ```

3. **Configure API Keys**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your API keys
   nano .env  # or use your preferred editor
   ```

4. **Run the System**
   ```bash
   python unified_system_demo.py
   ```

## 🔑 API Key Configuration

Create a `.env` file with your API keys:

```env
# OpenAI API Key (for GPT models)
OPENAI_API_KEY=your_openai_api_key_here

# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Groq API Key (for fast inference)
GROQ_API_KEY=your_groq_api_key_here

# Default LLM provider (openai, gemini, or groq)
DEFAULT_LLM_PROVIDER=openai
```

## 📊 System Architecture

### Phase 1: Neural Fields Foundation
- **ContextField**: Multi-dimensional semantic space with attractors
- **FieldManager**: Multi-field orchestration and management
- **ProtocolOrchestrator**: Executes operations on fields
- **CognitiveProcessor**: Basic reasoning capabilities
- **ParetoLangParser**: Parses protocol shells

### Phase 2: Cognitive Integration
- **EnhancedCognitiveToolEngine**: Advanced reasoning with 16.6% improvement
- **SymbolicMechanismEngine**: 3-stage symbolic processing
- **QuantumSemanticsEngine**: Observer-dependent interpretation
- **LLMInterface**: Integration with OpenAI, Gemini, and Groq

### Phase 3: Unified Operations
- **UnifiedContextOrchestrator**: Master coordination system
- **MultiProtocolOrchestrator**: 4 execution strategies
- **AdvancedFieldOperationsEngine**: Attractor scanning, boundary manipulation
- **SystemLevelPropertiesEngine**: Emergence detection and analysis

### Phase 4: Meta-Recursive Capabilities
- **MetaRecursiveOrchestrator**: Self-aware coordination
- **SelfReflectionEngine**: Multi-level introspection
- **InterpretabilityScaffold**: Decision tracing and explanation
- **HumanAIPartnershipFramework**: Collaborative evolution
- **RecursiveImprovementEngine**: Safe self-modification

## 🎯 Features Demonstrated

### Progressive Enhancement
The system shows how each phase builds upon the previous:

1. **Basic Field Operations** → **Enhanced Cognition** → **Unified System** → **Self-Improvement**
2. **Simple Protocols** → **LLM Enhancement** → **Multi-Strategy Execution** → **Meta-Cognition**
3. **Static Fields** → **Dynamic Adaptation** → **System Intelligence** → **Recursive Evolution**

### LLM Integration
- Enhances understanding and reasoning capabilities
- Supports multiple providers for flexibility
- Fallback mechanisms ensure robustness
- Context-aware responses based on field states

### Key Capabilities
- ✅ Neural fields with attractors and resonance patterns
- ✅ 16.6% mathematical reasoning improvement (IBM research-backed)
- ✅ Quantum semantics with observer-dependent meaning
- ✅ System-level emergence and intelligence detection
- ✅ Self-reflection and interpretability
- ✅ Human-AI collaborative evolution
- ✅ Safe recursive self-improvement

## 📖 Usage Examples

### Basic Usage
```python
from unified_system_demo import UnifiedContextSystem, LLMConfig

# Configure LLM providers
config = LLMConfig(
    openai_api_key="your_key",
    gemini_api_key="your_key",
    groq_api_key="your_key"
)

# Create and initialize system
system = UnifiedContextSystem(config)

# Initialize phases progressively
system.initialize_phase_1()  # Neural fields
system.initialize_phase_2()  # Cognitive enhancement
system.initialize_phase_3()  # Unified operations
system.initialize_phase_4()  # Meta-recursive capabilities

# Demonstrate all capabilities
system.demonstrate_progressive_enhancement()
```

### Advanced Usage
```python
# Access specific phase components
field_manager = system.phase_components['phase_1']['field_manager']
quantum_engine = system.phase_components['phase_2']['quantum_semantics']
meta_orchestrator = system.phase_components['phase_4']['meta_orchestrator']

# Create custom fields
custom_field = field_manager.create_field("custom", dimensions=2)
custom_field.inject("your concept", strength=0.8, position=(0.5, 0.5))

# Use quantum semantics
superposition = quantum_engine.create_semantic_superposition("ambiguous term")
collapsed = quantum_engine.collapse_meaning(superposition, "technical context")

# Enable self-reflection
reflection = meta_orchestrator.request_operation({
    "operation": "self_reflection",
    "depth": "philosophical",
    "target": "system purpose"
})
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Install LLM providers as needed

2. **API Key Errors**
   - Check your `.env` file has valid API keys
   - Ensure environment variables are loaded

3. **Module Not Found**
   - Run from the project root directory
   - Check that all phase demos have been run previously

4. **LLM Provider Failures**
   - System will fallback to available providers
   - Can run without LLMs with reduced functionality

## 🚀 Next Steps

1. **Explore Individual Phases**
   - Run `demo_phase1.py` through `demo_phase4.py` for detailed demonstrations
   - Read phase completion summaries for deep dives

2. **Customize the System**
   - Create custom protocol shells
   - Define new cognitive tools
   - Implement domain-specific enhancements

3. **Build Applications**
   - Adaptive learning platforms
   - Strategic decision support systems
   - Research discovery tools
   - Organizational intelligence systems

## 📄 License

This implementation follows the licensing of the Context Engineering System project.

## 🤝 Contributing

Contributions are welcome! The system is designed to be extensible:
- Add new LLM providers
- Create custom cognitive tools
- Implement new field operations
- Enhance meta-recursive capabilities

---

**Ready to experience the future of context-aware AI? Run `python setup_and_run.py` to begin!** 🚀