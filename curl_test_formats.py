#!/usr/bin/env python3
"""
Test API with curl-like requests to see what format it expects
"""

import requests
import os

def test_api_formats():
    """Test different API request formats to find what works"""
    print("🧪 Testing API Request Formats")
    print("=" * 50)
    
    # File paths
    weather_csv = r"C:\Users\soumi\Downloads\sample-weather.csv"
    questions_txt = r"C:\Users\soumi\Downloads\questions (1).txt"
    
    # Check files exist
    if not os.path.exists(weather_csv):
        print(f"❌ Weather CSV not found: {weather_csv}")
        return False
        
    if not os.path.exists(questions_txt):
        print(f"❌ Questions file not found: {questions_txt}")
        return False
    
    print(f"✅ Both files found")
    
    try:
        # Test 1: Health check
        print(f"\n1️⃣ Health Check")
        health = requests.get("http://localhost:8000/health")
        print(f"   Status: {health.status_code}")
        print(f"   Response: {health.json()}")
        
        # Test 2: Check what endpoints exist
        print(f"\n2️⃣ Root Endpoint")
        root = requests.get("http://localhost:8000/")
        print(f"   Status: {root.status_code}")
        if root.status_code == 200:
            print(f"   Response: {root.json()}")
        
        # Test 3: Try different file upload formats
        print(f"\n3️⃣ Testing File Upload Formats")
        
        # Format 1: Standard multipart
        print(f"   🔍 Format 1: Standard multipart files")
        with open(weather_csv, 'rb') as f1, open(questions_txt, 'rb') as f2:
            files = {
                'files': [
                    ('sample-weather.csv', f1.read(), 'text/csv'),
                    ('questions (1).txt', f2.read(), 'text/plain')
                ]
            }
            response1 = requests.post("http://localhost:8000/api/", files=files)
            print(f"      Status: {response1.status_code}")
            if response1.status_code != 200:
                print(f"      Error: {response1.text[:200]}")
        
        # Format 2: Single file field
        print(f"   🔍 Format 2: Single 'file' field")
        with open(weather_csv, 'rb') as f:
            files = {'file': ('sample-weather.csv', f.read(), 'text/csv')}
            response2 = requests.post("http://localhost:8000/api/", files=files)
            print(f"      Status: {response2.status_code}")
            if response2.status_code == 200:
                result = response2.json()
                print(f"      ✅ SUCCESS! Got result: {result}")
                return True
            else:
                print(f"      Error: {response2.text[:200]}")
        
        # Format 3: Named fields
        print(f"   🔍 Format 3: Named fields (weather_data, questions)")
        with open(weather_csv, 'rb') as f1, open(questions_txt, 'rb') as f2:
            files = {
                'weather_data': ('sample-weather.csv', f1.read(), 'text/csv'),
                'questions': ('questions.txt', f2.read(), 'text/plain')
            }
            response3 = requests.post("http://localhost:8000/api/", files=files)
            print(f"      Status: {response3.status_code}")
            if response3.status_code == 200:
                result = response3.json()
                print(f"      ✅ SUCCESS! Got result: {result}")
                return True
            else:
                print(f"      Error: {response3.text[:200]}")
        
        # Format 4: Try JSON body
        print(f"   🔍 Format 4: JSON body")
        with open(weather_csv, 'r') as f1, open(questions_txt, 'r') as f2:
            data = {
                'weather_data': f1.read(),
                'questions': f2.read()
            }
            response4 = requests.post(
                "http://localhost:8000/api/", 
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            print(f"      Status: {response4.status_code}")
            if response4.status_code == 200:
                result = response4.json()
                print(f"      ✅ SUCCESS! Got result: {result}")
                return True
            else:
                print(f"      Error: {response4.text[:200]}")
        
        # Format 5: Try /analyze endpoint
        print(f"   🔍 Format 5: /analyze endpoint")
        with open(weather_csv, 'rb') as f1, open(questions_txt, 'rb') as f2:
            files = {
                'file': ('sample-weather.csv', f1.read(), 'text/csv'),
                'questions': ('questions.txt', f2.read(), 'text/plain')
            }
            response5 = requests.post("http://localhost:8000/analyze", files=files)
            print(f"      Status: {response5.status_code}")
            if response5.status_code == 200:
                result = response5.json()
                print(f"      ✅ SUCCESS! Got result: {result}")
                return True
            else:
                print(f"      Error: {response5.text[:200]}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    return False

if __name__ == "__main__":
    success = test_api_formats()
    if success:
        print(f"\n🎉 Found working format!")
    else:
        print(f"\n❌ No working format found - API may need different approach")
