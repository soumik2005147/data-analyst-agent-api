#!/usr/bin/env python3
"""
Simple server startup script that avoids OneDrive sync issues
"""

import os
import sys
import subprocess
import time

def start_server():
    """Start the server with proper error handling"""
    
    print("🔧 Data Analyst Agent Server Startup")
    print("=" * 40)
    
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Check if required files exist
    if not os.path.exists('app.py'):
        print("❌ app.py not found!")
        return False
    
    print("✅ app.py found")
    
    # Try different approaches to start the server
    methods = [
        {
            "name": "Direct Python execution",
            "cmd": [sys.executable, "app.py"]
        },
        {
            "name": "Uvicorn module",
            "cmd": [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
        },
        {
            "name": "Uvicorn direct",
            "cmd": ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
        }
    ]
    
    for method in methods:
        print(f"\n🚀 Trying: {method['name']}")
        try:
            # Start the process
            process = subprocess.Popen(
                method['cmd'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=script_dir
            )
            
            # Wait a moment to see if it starts successfully
            time.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                print(f"✅ Server started successfully with {method['name']}!")
                print(f"🌐 API available at: http://localhost:8000")
                print(f"📊 Health check: http://localhost:8000/health")
                print(f"🔧 API endpoint: http://localhost:8000/api/")
                print("\n🛑 Press Ctrl+C to stop the server")
                
                try:
                    # Keep the server running
                    process.wait()
                except KeyboardInterrupt:
                    print("\n🛑 Shutting down server...")
                    process.terminate()
                    process.wait()
                    print("✅ Server stopped")
                
                return True
            else:
                # Process died, get error
                stdout, stderr = process.communicate()
                print(f"❌ Failed: {stderr.strip() if stderr else stdout.strip()}")
                
        except FileNotFoundError:
            print(f"❌ Command not found: {method['cmd'][0]}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n❌ All startup methods failed!")
    print("💡 Try installing uvicorn: pip install uvicorn")
    return False

if __name__ == "__main__":
    start_server()
