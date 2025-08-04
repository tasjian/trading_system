#!/usr/bin/env python3
"""
Continuous Rebalancer Management Script
Control and manage the continuous rebalancing system.
"""

import argparse
import json
import subprocess
import signal
import os
import sys
from pathlib import Path
from datetime import datetime

def get_status():
    """Get current system status."""
    state_file = Path("data/continuous_rebalancer_state.json")
    
    if not state_file.exists():
        print("❌ Rebalancer state file not found")
        print("System is likely not running")
        return
    
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        health = state.get("health", {})
        
        print("📊 CONTINUOUS REBALANCER STATUS")
        print("=" * 50)
        print(f"Total Runs: {health.get('total_runs', 0)}")
        print(f"Successful: {health.get('successful_runs', 0)}")
        print(f"Failed: {health.get('failed_runs', 0)}")
        
        if health.get('total_runs', 0) > 0:
            success_rate = (health.get('successful_runs', 0) / health.get('total_runs', 1)) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        print(f"Uptime: {health.get('uptime_hours', 0):.1f} hours")
        print(f"Consecutive Failures: {health.get('consecutive_failures', 0)}")
        
        last_success = health.get('last_success')
        if last_success:
            print(f"Last Success: {last_success}")
        
        last_failure = health.get('last_failure')
        if last_failure:
            print(f"Last Failure: {last_failure}")
        
        last_run = state.get('last_run_time')
        if last_run:
            print(f"Last Run: {last_run}")
        
    except Exception as e:
        print(f"❌ Error reading status: {e}")

def start_rebalancer(detached=False):
    """Start the continuous rebalancer."""
    print("🚀 Starting Continuous Rebalancer...")
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    if detached:
        # Run in background
        print("Starting in detached mode...")
        log_file = Path("logs/continuous_rebalancer.log")
        
        with open(log_file, 'a') as f:
            f.write(f"\n=== STARTED AT {datetime.now()} ===\n")
        
        # Start process in background
        process = subprocess.Popen([
            sys.executable, "continuous_rebalancer.py"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        # Save PID for later management
        pid_file = Path("data/rebalancer.pid")
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))
        
        print(f"✅ Started with PID {process.pid}")
        print(f"📝 Logs: {log_file}")
        print("Use 'python manage_rebalancer.py stop' to stop")
        print("Use 'python manage_rebalancer.py monitor' to monitor")
        
    else:
        # Run in foreground
        try:
            subprocess.run([sys.executable, "continuous_rebalancer.py"])
        except KeyboardInterrupt:
            print("\n👋 Stopped by user")

def stop_rebalancer():
    """Stop the continuous rebalancer."""
    pid_file = Path("data/rebalancer.pid")
    
    if not pid_file.exists():
        print("❌ No PID file found")
        print("Rebalancer may not be running or was started manually")
        return
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        print(f"🛑 Stopping rebalancer (PID {pid})...")
        
        # Send SIGTERM for graceful shutdown
        os.kill(pid, signal.SIGTERM)
        
        # Wait a moment
        import time
        time.sleep(2)
        
        # Check if still running
        try:
            os.kill(pid, 0)  # Check if process exists
            print("⏳ Waiting for graceful shutdown...")
            time.sleep(5)
            
            # Force kill if still running
            try:
                os.kill(pid, signal.SIGKILL)
                print("🔥 Force killed")
            except ProcessLookupError:
                pass
                
        except ProcessLookupError:
            print("✅ Stopped successfully")
        
        # Remove PID file
        pid_file.unlink()
        
    except Exception as e:
        print(f"❌ Error stopping rebalancer: {e}")

def monitor_rebalancer():
    """Start the monitoring dashboard."""
    print("🔍 Starting Monitor Dashboard...")
    try:
        subprocess.run([sys.executable, "monitor_rebalancer.py"])
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

def show_logs(lines=50):
    """Show recent log entries."""
    log_file = Path("logs/continuous_rebalancer.log")
    
    if not log_file.exists():
        print("❌ Log file not found")
        return
    
    print(f"📝 RECENT LOGS (last {lines} lines)")
    print("=" * 50)
    
    try:
        # Use tail equivalent
        with open(log_file, 'r') as f:
            file_lines = f.readlines()
            recent_lines = file_lines[-lines:] if len(file_lines) > lines else file_lines
            
            for line in recent_lines:
                print(line.rstrip())
                
    except Exception as e:
        print(f"❌ Error reading logs: {e}")

def reset_system():
    """Reset system state (careful!)."""
    confirm = input("⚠️  This will reset all system state. Continue? (yes/no): ")
    if confirm.lower() != 'yes':
        print("❌ Reset cancelled")
        return
    
    # Stop first
    stop_rebalancer()
    
    # Remove state files
    files_to_remove = [
        "data/continuous_rebalancer_state.json",
        "data/rebalancer.pid"
    ]
    
    for file_path in files_to_remove:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            print(f"🗑️  Removed {file_path}")
    
    print("✅ System state reset")
    print("You can now start fresh with 'python manage_rebalancer.py start'")

def main():
    parser = argparse.ArgumentParser(description="Manage Continuous Rebalancer")
    parser.add_argument("command", choices=[
        "start", "stop", "status", "monitor", "logs", "reset"
    ], help="Command to execute")
    parser.add_argument("-d", "--detached", action="store_true", 
                       help="Start in detached/background mode")
    parser.add_argument("-n", "--lines", type=int, default=50,
                       help="Number of log lines to show")
    
    args = parser.parse_args()
    
    if args.command == "start":
        start_rebalancer(detached=args.detached)
    elif args.command == "stop":
        stop_rebalancer()
    elif args.command == "status":
        get_status()
    elif args.command == "monitor":
        monitor_rebalancer()
    elif args.command == "logs":
        show_logs(args.lines)
    elif args.command == "reset":
        reset_system()

if __name__ == "__main__":
    main()