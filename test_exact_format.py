#!/usr/bin/env python3
"""
Test the API to get the exact response format [1, "Titanic", 0.485782, "data:image/png;base64,...]
"""

import requests
import json
import time

def test_exact_format():
    """Test to get the exact response format"""
    
    print("🧪 Testing API for Exact Response Format")
    print("=" * 50)
    
    # Wikipedia films questions (matching the expected format)
    questions_content = """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes."""
    
    # Write questions file
    with open('wikipedia_questions.txt', 'w') as f:
        f.write(questions_content)
    
    # Test API
    api_url = "http://localhost:8000/api/"
    
    try:
        # Check if API is running
        print("🔄 Checking API status...")
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"✅ API Health: {health_response.status_code}")
        
        # Send request
        print("🚀 Sending Wikipedia analysis request...")
        with open('wikipedia_questions.txt', 'rb') as f:
            files = {'files': ('questions.txt', f, 'text/plain')}
            response = requests.post(api_url, files=files, timeout=180)
        
        print(f"📨 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Response received!")
            print(f"Response Type: {type(result)}")
            
            if isinstance(result, list):
                print(f"\n🎯 EXACT FORMAT ACHIEVED!")
                print(f"Array Length: {len(result)}")
                print(f"Format: {type(result)} with {len(result)} elements")
                
                # Show each element
                for i, item in enumerate(result, 1):
                    if isinstance(item, str) and item.startswith('data:image'):
                        print(f"  [{i}]: {item[:50]}... (image data, {len(item)} chars)")
                    elif isinstance(item, str) and len(item) > 100:
                        print(f"  [{i}]: {item[:100]}... (truncated)")
                    else:
                        print(f"  [{i}]: {item}")
                
                # Check if it matches expected pattern
                expected_pattern = [
                    ("int", "number of $2bn movies before 2000"),
                    ("str", "earliest film name"),
                    ("float", "correlation value"),
                    ("str starting with data:image", "visualization")
                ]
                
                print(f"\n📋 Format Validation:")
                for i, (expected_type, description) in enumerate(expected_pattern):
                    if i < len(result):
                        actual_value = result[i]
                        if "int" in expected_type:
                            status = "✅" if isinstance(actual_value, int) else "⚠️"
                        elif "str" in expected_type:
                            if "data:image" in expected_type:
                                status = "✅" if isinstance(actual_value, str) and actual_value.startswith('data:image') else "⚠️"
                            else:
                                status = "✅" if isinstance(actual_value, str) else "⚠️"
                        elif "float" in expected_type:
                            status = "✅" if isinstance(actual_value, (int, float)) else "⚠️"
                        else:
                            status = "⚠️"
                        
                        print(f"  {status} Position {i+1}: {description}")
                    else:
                        print(f"  ❌ Position {i+1}: Missing")
                
                print(f"\n🏆 FINAL RESULT:")
                print(f"Response: {result}")
                
                # Check if it's the exact format we want
                if (len(result) == 4 and 
                    isinstance(result[0], int) and
                    isinstance(result[1], str) and
                    isinstance(result[2], (int, float)) and
                    isinstance(result[3], str) and result[3].startswith('data:image')):
                    print("🎉 PERFECT! Response is in exact expected format!")
                else:
                    print("⚠️  Close, but format needs minor adjustments")
            
            else:
                print(f"❌ Expected array, got {type(result)}")
                print(f"Response: {result}")
        
        else:
            print(f"❌ API Error: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is it running at http://localhost:8000?")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        # Cleanup
        try:
            import os
            os.remove('wikipedia_questions.txt')
        except:
            pass

if __name__ == "__main__":
    test_exact_format()
