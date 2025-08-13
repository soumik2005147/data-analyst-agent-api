#!/usr/bin/env python3
"""
Test the API with the correct format based on OpenAPI spec
"""

import requests
import os

def test_correct_format():
    """Test with the correct 'files' parameter format"""
    print("🧪 Testing API with Correct Format")
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
        
    print(f"✅ Found both files")
    
    try:
        # Read files
        with open(weather_csv, 'rb') as f:
            csv_content = f.read()
        with open(questions_txt, 'rb') as f:
            txt_content = f.read()
        
        print(f"📄 CSV: {len(csv_content)} bytes")
        print(f"📄 TXT: {len(txt_content)} bytes")
        
        # Test with correct format - 'files' parameter as array
        print(f"\n🚀 Testing with 'files' parameter...")
        
        files_data = [
            ('files', ('sample-weather.csv', csv_content, 'text/csv')),
            ('files', ('questions.txt', txt_content, 'text/plain'))
        ]
        
        response = requests.post(
            "http://localhost:8000/api/",
            files=files_data,
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS!")
            print(f"Response: {result}")
            
            if isinstance(result, list) and len(result) == 4:
                print(f"\n🎉 PERFECT FORMAT!")
                print(f"[{result[0]}, '{result[1]}', {result[2]}, 'visualization']")
                print(f"Types: [{type(result[0])}, {type(result[1])}, {type(result[2])}, {type(result[3])}]")
                return True
            else:
                print(f"❌ Unexpected format: {result}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        
    return False

if __name__ == "__main__":
    success = test_correct_format()
    if success:
        print(f"\n🎉 SUCCESS! Got the answers to your questions!")
    else:
        print(f"\n❌ Test failed")
