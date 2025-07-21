#!/usr/bin/env python3
"""Setup script for the agentic trading system."""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def setup_directories():
    """Create necessary directories."""
    print("📁 Setting up directories...")
    directories = ["logs", "data", "backups"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created {directory}/ directory")

def check_env_file():
    """Check if .env file exists and is configured."""
    print("🔧 Checking configuration...")
    
    if not Path(".env").exists():
        print("⚠️  .env file not found. Copying from .env.example...")
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("✓ .env file created from template")
            print("📝 Please edit .env file with your API keys before running the system")
        else:
            print("❌ .env.example not found")
            return False
    else:
        print("✓ .env file exists")
    
    # Check if basic keys are set
    with open(".env", "r") as f:
        content = f.read()
        
    required_keys = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
    missing_keys = []
    
    for key in required_keys:
        if f"{key}=your_" in content or f"{key}=" in content.replace(f"{key}=\n", f"{key}="):
            missing_keys.append(key)
    
    if missing_keys:
        print(f"⚠️  Please configure these API keys in .env: {', '.join(missing_keys)}")
        return False
    
    print("✓ Basic configuration appears complete")
    return True

def run_tests():
    """Run basic system tests."""
    print("🧪 Running basic tests...")
    try:
        # Import test to check basic functionality
        sys.path.append(os.getcwd())
        
        # Test imports
        from config.settings import settings
        from tools.alpaca_client import AlpacaClient
        from agents.state import create_initial_state
        from tools.risk_controls import RiskMonitor
        
        print("✓ Core modules import successfully")
        
        # Test state creation
        state = create_initial_state("test")
        assert "portfolio" in state
        print("✓ State management working")
        
        # Test risk monitor
        monitor = RiskMonitor()
        assert monitor is not None
        print("✓ Risk controls initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Agentic Trading System")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Setup directories
    setup_directories()
    
    # Install dependencies
    install_dependencies()
    
    # Check configuration
    env_configured = check_env_file()
    
    # Run tests
    tests_passed = run_tests()
    
    print("\n" + "=" * 50)
    print("📋 Setup Summary")
    print("=" * 50)
    
    if env_configured and tests_passed:
        print("✅ Setup completed successfully!")
        print("\n🎯 Next steps:")
        print("1. Verify your API keys in .env file")
        print("2. Run: python main.py")
        print("3. Use 'cycle' command to test a trading cycle")
        print("4. Use 'auto' command for continuous trading")
        
        print("\n⚠️  Important Reminders:")
        print("- System is configured for PAPER TRADING only")
        print("- Monitor the system actively during operation")
        print("- Review risk limits in .env before starting")
        
    else:
        print("⚠️  Setup completed with warnings")
        if not env_configured:
            print("- Configure API keys in .env file")
        if not tests_passed:
            print("- Fix import/dependency issues")
    
    print("\n📚 Documentation: See README.md for detailed usage")
    print("🛡️  Safety: Always use paper trading mode")

if __name__ == "__main__":
    main()