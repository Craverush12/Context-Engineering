#!/usr/bin/env python3
"""
Setup and Run Script for Unified Context Engineering System
==========================================================

This script helps you:
1. Install required dependencies including LLM providers
2. Configure API keys
3. Run the unified system demonstration
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Ensure Python 3.8+ is being used"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version}")


def install_core_dependencies():
    """Install core dependencies"""
    print("\n📦 Installing core dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Core dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install core dependencies")
        return False


def ask_llm_providers():
    """Ask which LLM providers to install"""
    print("\n🤖 Which LLM providers would you like to use?")
    providers = []
    
    if input("   Install OpenAI? (y/n): ").lower() == 'y':
        providers.append("openai>=1.0.0")
        
    if input("   Install Google Gemini? (y/n): ").lower() == 'y':
        providers.append("google-generativeai>=0.3.0")
        
    if input("   Install Groq? (y/n): ").lower() == 'y':
        providers.append("groq>=0.4.0")
        
    return providers


def install_llm_providers(providers):
    """Install selected LLM providers"""
    if not providers:
        print("⚠️  No LLM providers selected. The system will run with limited functionality.")
        return
        
    print(f"\n📦 Installing LLM providers: {', '.join(providers)}")
    for provider in providers:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", provider])
            print(f"✅ {provider} installed successfully!")
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {provider}")


def setup_env_file():
    """Setup .env file with API keys"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("\n🔑 Setting up API keys...")
        print("   Creating .env file from .env.example")
        
        # Copy example to .env
        env_content = env_example.read_text()
        
        # Ask for API keys
        print("\n   Enter your API keys (press Enter to skip):")
        
        openai_key = input("   OpenAI API key: ").strip()
        if openai_key:
            env_content = env_content.replace("your_openai_api_key_here", openai_key)
            
        gemini_key = input("   Gemini API key: ").strip()
        if gemini_key:
            env_content = env_content.replace("your_gemini_api_key_here", gemini_key)
            
        groq_key = input("   Groq API key: ").strip()
        if groq_key:
            env_content = env_content.replace("your_groq_api_key_here", groq_key)
            
        # Write .env file
        env_file.write_text(env_content)
        print("✅ .env file created successfully!")
        
    elif env_file.exists():
        print("\n✅ .env file already exists")
    else:
        print("\n⚠️  No .env.example file found")


def load_env_variables():
    """Load environment variables from .env file"""
    env_file = Path(".env")
    if env_file.exists():
        print("\n🔧 Loading environment variables...")
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
        print("✅ Environment variables loaded")
        
        # Show available providers
        available = []
        if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here":
            available.append("OpenAI")
        if os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_KEY") != "your_gemini_api_key_here":
            available.append("Gemini")
        if os.getenv("GROQ_API_KEY") and os.getenv("GROQ_API_KEY") != "your_groq_api_key_here":
            available.append("Groq")
            
        if available:
            print(f"   Available LLM providers: {', '.join(available)}")
        else:
            print("   ⚠️  No LLM API keys configured")


def run_unified_system():
    """Run the unified system demonstration"""
    print("\n🚀 Running Unified Context Engineering System...")
    print("="*60)
    
    try:
        subprocess.check_call([sys.executable, "unified_system_demo.py"])
    except subprocess.CalledProcessError:
        print("\n❌ System execution failed")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Execution interrupted by user")
        return False
        
    return True


def main():
    """Main setup and run function"""
    print("🌟 CONTEXT ENGINEERING SYSTEM - SETUP & RUN")
    print("="*60)
    
    # Check Python version
    check_python_version()
    
    # Install core dependencies
    if not install_core_dependencies():
        print("\n❌ Setup failed. Please check error messages above.")
        sys.exit(1)
    
    # Ask about LLM providers
    llm_providers = ask_llm_providers()
    if llm_providers:
        install_llm_providers(llm_providers)
    
    # Setup environment file
    setup_env_file()
    
    # Load environment variables
    load_env_variables()
    
    # Ask if user wants to run the system
    print("\n" + "="*60)
    if input("🎯 Ready to run the unified system? (y/n): ").lower() == 'y':
        run_unified_system()
    else:
        print("\n✅ Setup complete! Run 'python unified_system_demo.py' when ready.")
        print("   Make sure to configure your API keys in .env file first.")


if __name__ == "__main__":
    main()