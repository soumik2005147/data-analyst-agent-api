#!/usr/bin/env python3
"""
curl Command Tester - Simulates the exact curl command format required
"""

import subprocess
import tempfile
import os
import json

def test_with_curl():
    """Test using curl command exactly as specified in requirements"""
    
    print("🚀 Testing Data Analyst Agent API with curl")
    print("=" * 50)
    
    # Create the questions.txt file as required
    questions_content = """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes."""

    # Write questions.txt file
    with open('questions.txt', 'w') as f:
        f.write(questions_content)
    
    try:
        print("📝 Created questions.txt file")
        print("🌐 Testing API endpoint...")
        
        # Test health first
        health_cmd = 'curl -s "http://localhost:8000/health"'
        print(f"\n1️⃣ Health Check: {health_cmd}")
        
        try:
            result = subprocess.run(
                health_cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                health_data = json.loads(result.stdout)
                print(f"   ✅ Health Status: {health_data.get('status', 'unknown')}")
            else:
                print(f"   ❌ Health check failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"   ❌ Health check error: {e}")
            return False
        
        # Test main API endpoint with curl
        api_cmd = 'curl -s -X POST "http://localhost:8000/api/" -F "questions.txt=@questions.txt"'
        print(f"\n2️⃣ API Test: {api_cmd}")
        
        result = subprocess.run(
            api_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            try:
                response_data = json.loads(result.stdout)
                print(f"   ✅ API Response received!")
                print(f"   📊 Response type: {type(response_data)}")
                print(f"   📏 Response length: {len(response_data) if isinstance(response_data, list) else 'not a list'}")
                
                if isinstance(response_data, list) and len(response_data) == 4:
                    print(f"\n📋 Response Analysis:")
                    print(f"   [0] Count: {response_data[0]} ({type(response_data[0]).__name__})")
                    print(f"   [1] Film: {response_data[1]} ({type(response_data[1]).__name__})")
                    print(f"   [2] Correlation: {response_data[2]} ({type(response_data[2]).__name__})")
                    print(f"   [3] Visualization: {response_data[3][:50]}... ({len(response_data[3])} chars)")
                    
                    # Validate format
                    format_valid = (
                        isinstance(response_data[0], int) and
                        isinstance(response_data[1], str) and
                        isinstance(response_data[2], (int, float)) and
                        isinstance(response_data[3], str) and 
                        response_data[3].startswith('data:image')
                    )
                    
                    if format_valid:
                        print(f"\n🎉 SUCCESS! Perfect format match!")
                        print(f"   Expected: [int, str, float, data_uri]")
                        print(f"   Received: [int, str, float, data_uri] ✅")
                        print(f"\n🏆 EXACT RESPONSE:")
                        print(f"   {response_data}")
                        return True
                    else:
                        print(f"\n⚠️  Format validation failed")
                        return False
                else:
                    print(f"\n❌ Expected 4-element array, got: {response_data}")
                    return False
                    
            except json.JSONDecodeError:
                print(f"   ❌ Invalid JSON response: {result.stdout}")
                return False
        else:
            print(f"   ❌ curl command failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False
    finally:
        # Cleanup
        try:
            os.remove('questions.txt')
        except:
            pass
    
    return False

if __name__ == "__main__":
    success = test_with_curl()
    if success:
        print("\n✅ ALL TESTS PASSED! API is working correctly.")
    else:
        print("\n❌ Tests failed. Check if server is running at http://localhost:8000")
