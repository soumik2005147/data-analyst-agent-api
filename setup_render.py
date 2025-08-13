#!/usr/bin/env python3
"""
Quick setup script for Render deployment
"""

import os
import subprocess
import sys

def check_git_repo():
    """Check if this is a git repository"""
    try:
        subprocess.run(['git', 'status'], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def initialize_git():
    """Initialize git repository and add files"""
    print("🔧 Initializing git repository...")
    
    commands = [
        ['git', 'init'],
        ['git', 'add', '.'],
        ['git', 'commit', '-m', 'Initial commit: Data Analyst Agent API for Render deployment']
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ {' '.join(cmd)}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running {' '.join(cmd)}: {e}")
            print(f"   Output: {e.stdout}")
            print(f"   Error: {e.stderr}")
            return False
    
    return True

def print_deployment_instructions():
    """Print next steps for deployment"""
    print("\n🚀 Ready for Render deployment!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Push this repository to GitHub:")
    print("   git remote add origin https://github.com/yourusername/your-repo.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    print("\n2. Deploy on Render:")
    print("   - Go to https://dashboard.render.com/")
    print("   - Click 'New +' → 'Web Service'")
    print("   - Connect your GitHub repository")
    print("   - Use these settings:")
    print("     * Build Command: pip install -r requirements.txt")
    print("     * Start Command: python production_ready_api.py")
    print("\n3. Set Environment Variables in Render:")
    print("   - OPENAI_API_KEY: your_openai_api_key_here")
    print("\n4. Test your deployment:")
    print("   python test_render_deployment.py https://your-app.onrender.com")

def main():
    print("🛠️  Data Analyst Agent API - Render Setup")
    print("=" * 50)
    
    # Check if git is installed
    try:
        subprocess.run(['git', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git is not installed or not in PATH")
        print("   Please install Git: https://git-scm.com/downloads")
        sys.exit(1)
    
    # Check if already a git repo
    if check_git_repo():
        print("✅ Git repository already exists")
    else:
        if initialize_git():
            print("✅ Git repository initialized")
        else:
            print("❌ Failed to initialize git repository")
            sys.exit(1)
    
    # Check important files exist
    required_files = [
        'production_ready_api.py',
        'requirements.txt',
        'render.yaml',
        'Procfile',
        'README.md',
        'LICENSE'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing files: {', '.join(missing_files)}")
        print("   Please ensure all files are present before deployment")
    else:
        print("✅ All required files present")
    
    print_deployment_instructions()

if __name__ == "__main__":
    main()
