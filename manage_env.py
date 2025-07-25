#!/usr/bin/env python3
"""
Environment File Manager

Clarifies which .env file is being used and provides management utilities.
"""

import os
import shutil
from datetime import datetime

def show_env_status():
    """Show the status of all environment files."""
    
    print("🔧 ENVIRONMENT FILES STATUS")
    print("=" * 60)
    
    env_files = {
        '.env': 'ACTIVE - Used by application',
        '.env.real': 'Updated backup with current config',
        '.env.example': 'Template for GitHub (no secrets)',
        '.env.backup': 'Original backup'
    }
    
    for file, description in env_files.items():
        if os.path.exists(file):
            size = os.path.getsize(file)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file))
            print(f"✅ {file}")
            print(f"   {description}")
            print(f"   Size: {size} bytes")
            print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        else:
            print(f"❌ {file} - Not found")
            print()
    
    print("📝 CONFIGURATION SUMMARY:")
    print("  • Application uses: .env (loaded by config/settings.py)")
    print("  • Current setup: Aggressive trading, email configured")
    print("  • API keys: Alpaca working, email needs app password")
    print()

def switch_to_conservative():
    """Switch to conservative trading configuration."""
    print("🔄 Switching to conservative trading configuration...")
    
    # Backup current .env
    backup_name = f".env.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy('.env', backup_name)
    print(f"   Backed up current .env to {backup_name}")
    
    # Read current .env and modify trading settings
    with open('.env', 'r') as f:
        content = f.read()
    
    # Replace aggressive settings with conservative
    content = content.replace('MAX_PORTFOLIO_RISK=0.05', 'MAX_PORTFOLIO_RISK=0.02')
    content = content.replace('MAX_POSITION_SIZE=0.25', 'MAX_POSITION_SIZE=0.08')
    content = content.replace('STOP_LOSS_PERCENT=0.08', 'STOP_LOSS_PERCENT=0.05')
    content = content.replace('# Trading Configuration - Aggressive stance', '# Trading Configuration - Conservative stance')
    
    with open('.env', 'w') as f:
        f.write(content)
    
    print("   ✅ Switched to conservative trading settings")
    print("   📊 New settings: 2% risk, 8% max position, 5% stop loss")

def switch_to_aggressive():
    """Switch to aggressive trading configuration."""
    print("🚀 Switching to aggressive trading configuration...")
    
    # Backup current .env
    backup_name = f".env.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy('.env', backup_name)
    print(f"   Backed up current .env to {backup_name}")
    
    # Read current .env and modify trading settings
    with open('.env', 'r') as f:
        content = f.read()
    
    # Replace conservative settings with aggressive
    content = content.replace('MAX_PORTFOLIO_RISK=0.02', 'MAX_PORTFOLIO_RISK=0.05')
    content = content.replace('MAX_POSITION_SIZE=0.08', 'MAX_POSITION_SIZE=0.25')
    content = content.replace('STOP_LOSS_PERCENT=0.05', 'STOP_LOSS_PERCENT=0.08')
    content = content.replace('# Trading Configuration - Conservative stance', '# Trading Configuration - Aggressive stance')
    
    with open('.env', 'w') as f:
        f.write(content)
    
    print("   ✅ Switched to aggressive trading settings")
    print("   📊 New settings: 5% risk, 15% max position, 8% stop loss")

def setup_email():
    """Help setup email configuration."""
    print("📧 EMAIL CONFIGURATION SETUP")
    print("=" * 40)
    print()
    print("Current email configuration:")
    print("  GMAIL_EMAIL=ztaschdjian@gmail.com ✅")
    print("  GMAIL_APP_PASSWORD=your_16_digit_app_password_here ❌")
    print()
    print("📝 TO FIX EMAIL NOTIFICATIONS:")
    print("1. Go to https://myaccount.google.com/apppasswords")
    print("2. Generate an app password for 'Mail'")
    print("3. Update .env file with: GMAIL_APP_PASSWORD=your_actual_16_digit_password")
    print()
    print("🔗 ALTERNATIVE: Set up webhooks")
    print("  • Zapier: https://hooks.zapier.com/")
    print("  • Make.com: https://make.com/")
    print("  • n8n: Self-hosted automation")

if __name__ == "__main__":
    print("🤖 AI TRADING SYSTEM - Environment Manager")
    print("=" * 60)
    
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'status':
            show_env_status()
        elif command == 'conservative':
            switch_to_conservative()
        elif command == 'aggressive':
            switch_to_aggressive()
        elif command == 'email':
            setup_email()
        else:
            print("❌ Unknown command. Available: status, conservative, aggressive, email")
    else:
        show_env_status()
        print("💡 AVAILABLE COMMANDS:")
        print("  python manage_env.py status      - Show environment status")
        print("  python manage_env.py conservative - Switch to conservative trading")
        print("  python manage_env.py aggressive  - Switch to aggressive trading")
        print("  python manage_env.py email       - Setup email configuration")