#!/usr/bin/env python3
"""
Quick test to see the actual results with real data
"""

import requests
import os

def check_real_results():
    """Check what the API returns with real data"""
    print("🔍 Checking Real Results")
    print("=" * 40)
    
    # File paths
    weather_csv = r"C:\Users\soumi\Downloads\sample-weather.csv"
    questions_txt = r"C:\Users\soumi\Downloads\questions (1).txt"
    
    # Check files exist
    if not os.path.exists(weather_csv) or not os.path.exists(questions_txt):
        print("❌ Files not found")
        return
    
    try:
        # Read files
        with open(weather_csv, 'rb') as f:
            csv_content = f.read()
        with open(questions_txt, 'rb') as f:
            txt_content = f.read()
        
        print(f"📊 CSV: {len(csv_content)} bytes")
        print(f"📋 Questions: {len(txt_content)} bytes")
        
        # Also show first few lines of each file
        with open(weather_csv, 'r') as f:
            csv_preview = f.read()[:200]
        with open(questions_txt, 'r') as f:
            questions_preview = f.read()
            
        print(f"\n📊 Weather CSV Preview:")
        print(f"   {csv_preview}...")
        
        print(f"\n📋 Questions:")
        print(f"   {questions_preview}")
        
        # Test API
        files_data = [
            ('files', ('sample-weather.csv', csv_content, 'text/csv')),
            ('files', ('questions.txt', txt_content, 'text/plain'))
        ]
        
        print(f"\n🚀 Sending to API...")
        response = requests.post(
            "http://localhost:8000/api/",
            files=files_data,
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS! API Response:")
            print(f"   Type: {type(result)}")
            print(f"   Length: {len(result) if isinstance(result, list) else 'N/A'}")
            print(f"   Full Response: {result}")
            
            if isinstance(result, list) and len(result) == 4:
                print(f"\n🎯 ANSWERS TO YOUR QUESTIONS:")
                print(f"   Answer 1 ({type(result[0]).__name__}): {result[0]}")
                print(f"   Answer 2 ({type(result[1]).__name__}): {result[1]}")
                print(f"   Answer 3 ({type(result[2]).__name__}): {result[2]}")
                print(f"   Answer 4 ({type(result[3]).__name__}): {result[3][:100]}...")
                
                print(f"\n🏆 FINAL FORMAT: [{result[0]}, '{result[1]}', {result[2]}, 'data:...']")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_real_results()
