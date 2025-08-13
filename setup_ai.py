#!/usr/bin/env python3
"""
Setup script for AI-Enhanced Data Analyst Agent
"""

import os
import sys

def setup_openai_key():
    """Setup OpenAI API key"""
    print("🔑 OpenAI API Key Setup")
    print("=" * 30)
    
    # Check if key already exists
    existing_key = os.getenv('OPENAI_API_KEY')
    if existing_key:
        print(f"✅ OpenAI API key already set: {existing_key[:8]}...")
        return True
    
    print("Please enter your OpenAI API key:")
    print("(It should start with 'sk-')")
    
    api_key = input("API Key: ").strip()
    
    if not api_key.startswith('sk-'):
        print("❌ Invalid API key format. It should start with 'sk-'")
        return False
    
    # Set environment variable for current session
    os.environ['OPENAI_API_KEY'] = api_key
    
    # Create a .env file for persistence
    try:
        with open('.env', 'w') as f:
            f.write(f'OPENAI_API_KEY={api_key}\n')
        print("✅ API key saved to .env file")
    except Exception as e:
        print(f"⚠️  Could not save to .env file: {e}")
    
    # For Windows PowerShell - set for current session
    print(f"\n💡 To set permanently in PowerShell, run:")
    print(f'   $env:OPENAI_API_KEY="{api_key}"')
    
    return True

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing AI dependencies...")
    
    packages = [
        'openai>=1.0.0',
        'python-dotenv'
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            os.system(f'pip install {package}')
        except Exception as e:
            print(f"❌ Failed to install {package}: {e}")

def test_openai_connection():
    """Test OpenAI API connection"""
    print("\n🧪 Testing OpenAI connection...")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Simple test
        response = client.models.list()
        print("✅ OpenAI connection successful!")
        print(f"   Available models: {len(response.data)} models found")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 AI-Enhanced Data Analyst Agent Setup")
    print("=" * 50)
    
    # Install dependencies
    install_dependencies()
    
    # Setup API key
    if setup_openai_key():
        # Load .env file if it exists
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ Environment loaded")
        except:
            pass
        
        # Test connection
        if test_openai_connection():
            print("\n🎉 Setup complete! You can now run:")
            print("   python ai_enhanced_api.py")
        else:
            print("\n⚠️  Setup completed with warnings")
            print("   Check your API key and try again")
    else:
        print("\n❌ Setup failed. Please check your API key")
