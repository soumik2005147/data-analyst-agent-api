#!/usr/bin/env python3
"""
Quick API Test - Simple test that doesn't interfere with server process
"""

import requests
import json

def quick_test():
    """Quick test of the running API"""
    print("🧪 Quick API Test")
    print("=" * 30)
    
    try:
        # Test health
        print("1️⃣ Health Check...")
        health = requests.get("http://localhost:8000/health", timeout=5)
        print(f"   Status: {health.status_code}")
        
        if health.status_code == 200:
            print("   ✅ Server is healthy!")
            
            # Test API with simple data
            print("\n2️⃣ API Test...")
            
            # Create questions content
            questions_data = "Wikipedia analysis questions:\n1. How many $2bn movies before 2000?\n2. Earliest film over $1.5bn?\n3. Correlation between Rank and Peak?\n4. Scatterplot visualization?"
            
            # Prepare files for upload
            files = {
                'questions.txt': ('questions.txt', questions_data, 'text/plain')
            }
            
            # Send request
            response = requests.post(
                "http://localhost:8000/api/", 
                files=files, 
                timeout=20
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ SUCCESS! Got response!")
                print(f"   Type: {type(result)}")
                print(f"   Length: {len(result) if isinstance(result, list) else 'not list'}")
                
                if isinstance(result, list) and len(result) == 4:
                    print(f"\n🎯 FORMAT CHECK:")
                    print(f"   [0]: {result[0]} ({type(result[0]).__name__})")
                    print(f"   [1]: {result[1]} ({type(result[1]).__name__})")
                    print(f"   [2]: {result[2]} ({type(result[2]).__name__})")
                    print(f"   [3]: {result[3][:40]}... ({len(result[3])} chars)")
                    
                    # Check exact format
                    if (isinstance(result[0], int) and 
                        isinstance(result[1], str) and
                        isinstance(result[2], (int, float)) and
                        isinstance(result[3], str) and result[3].startswith('data:image')):
                        print(f"\n🎉 PERFECT! Exact format achieved!")
                        print(f"   [{result[0]}, \"{result[1]}\", {result[2]}, \"data:image/...\"]")
                        return True
                    else:
                        print(f"\n⚠️  Format issues detected")
                else:
                    print(f"\n❌ Expected 4-element array")
            else:
                print(f"   ❌ API Error: {response.text}")
        else:
            print(f"   ❌ Health check failed")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    return False

if __name__ == "__main__":
    success = quick_test()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")
