#!/usr/bin/env python3
"""
Simple API Format Tester
Tests the API and shows the exact response format
"""

import requests
import time
import sys

def test_api_format():
    """Test the API and verify the exact format"""
    
    print("🧪 Data Analyst Agent API Format Test")
    print("=" * 50)
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    try:
        # Test health endpoint first
        print("\n1️⃣ Testing Health Endpoint")
        health_response = requests.get("http://localhost:8000/health", timeout=10)
        print(f"   Status: {health_response.status_code}")
        print(f"   Response: {health_response.json()}")
        
        if health_response.status_code != 200:
            print("❌ Health check failed - server may not be running")
            return
        
        # Test main API endpoint
        print("\n2️⃣ Testing Analysis Endpoint")
        
        # Create sample questions content
        questions_content = """Wikipedia Films Analysis Questions:

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak with a regression line."""
        
        # Send the request
        files = {'files': ('questions.txt', questions_content, 'text/plain')}
        
        print("   Sending analysis request...")
        api_response = requests.post(
            "http://localhost:8000/api/", 
            files=files, 
            timeout=30
        )
        
        print(f"   Status: {api_response.status_code}")
        
        if api_response.status_code == 200:
            result = api_response.json()
            
            print(f"\n✅ SUCCESS! API Response Received")
            print(f"   Response Type: {type(result)}")
            print(f"   Response Length: {len(result) if isinstance(result, list) else 'not a list'}")
            
            if isinstance(result, list) and len(result) == 4:
                print(f"\n📋 Response Format Analysis:")
                print(f"   [0]: {result[0]} ({type(result[0]).__name__})")
                print(f"   [1]: {result[1]} ({type(result[1]).__name__})")
                print(f"   [2]: {result[2]} ({type(result[2]).__name__})")
                print(f"   [3]: {result[3][:50]}... ({type(result[3]).__name__}, {len(result[3])} chars)")
                
                # Format validation
                is_valid = (
                    isinstance(result[0], int) and
                    isinstance(result[1], str) and
                    isinstance(result[2], (int, float)) and
                    isinstance(result[3], str) and result[3].startswith('data:image')
                )
                
                if is_valid:
                    print(f"\n🎉 PERFECT FORMAT MATCH!")
                    print(f"   Expected: [int, str, float, data_uri]")
                    print(f"   Received: [int, str, float, data_uri] ✅")
                    print(f"\n🏆 FINAL RESPONSE FORMAT:")
                    print(f"   [{result[0]}, \"{result[1]}\", {result[2]}, \"data:image/png;base64,...\"]")
                    
                    return True
                else:
                    print(f"\n⚠️  Format validation failed")
                    return False
            else:
                print(f"\n❌ Expected 4-element array, got: {result}")
                return False
        else:
            print(f"❌ API Error ({api_response.status_code}): {api_response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server at http://localhost:8000")
        print(f"   Make sure the server is running first!")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_api_format()
    sys.exit(0 if success else 1)
