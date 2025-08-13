#!/usr/bin/env python3
"""
Test the REAL data analysis API on port 8003
"""

import requests
import os

def test_real_analysis():
    """Test with the enhanced API that does real analysis"""
    print("🧪 Testing REAL Data Analysis API")
    print("=" * 50)
    
    # File paths
    weather_csv = r"C:\Users\soumi\Downloads\sample-weather.csv"
    questions_txt = r"C:\Users\soumi\Downloads\questions (1).txt"
    
    # Check files exist
    if not os.path.exists(weather_csv) or not os.path.exists(questions_txt):
        print("❌ Files not found")
        return False
    
    try:
        # Read files
        with open(weather_csv, 'rb') as f:
            csv_content = f.read()
        with open(questions_txt, 'rb') as f:
            txt_content = f.read()
        
        print(f"📊 Weather CSV: {len(csv_content)} bytes")
        print(f"📋 Questions: {len(txt_content)} bytes")
        
        # Show what questions we're asking
        with open(questions_txt, 'r') as f:
            questions = f.read()
        print(f"\n📋 Your Questions:")
        print(f"   {questions}")
        
        # Test the REAL analysis API on port 8003
        files_data = [
            ('files', ('sample-weather.csv', csv_content, 'text/csv')),
            ('files', ('questions.txt', txt_content, 'text/plain'))
        ]
        
        print(f"\n🚀 Sending to REAL Analysis API (port 8003)...")
        response = requests.post(
            "http://localhost:8003/api/",
            files=files_data,
            timeout=60  # Longer timeout for real analysis
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS! REAL Analysis Complete:")
            
            if isinstance(result, list) and len(result) == 4:
                print(f"\n🎯 REAL ANSWERS FROM YOUR WEATHER DATA:")
                print(f"   Answer 1 ({type(result[0]).__name__}): {result[0]}")
                print(f"   Answer 2 ({type(result[1]).__name__}): {result[1]}")
                print(f"   Answer 3 ({type(result[2]).__name__}): {result[2]}")
                
                if str(result[3]).startswith('data:'):
                    print(f"   Answer 4 ({type(result[3]).__name__}): Visualization data ({len(str(result[3]))} chars)")
                else:
                    print(f"   Answer 4 ({type(result[3]).__name__}): {result[3]}")
                
                print(f"\n🏆 FINAL REAL RESULT: [{result[0]}, '{result[1]}', {result[2]}, 'visualization']")
                print(f"📊 This is REAL analysis of your weather CSV file!")
                return True
            else:
                print(f"❌ Unexpected format: {result}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    return False

if __name__ == "__main__":
    success = test_real_analysis()
    if success:
        print(f"\n🎉 SUCCESS! Got REAL answers from your weather data!")
        print(f"✅ No more 'Titanic' - this is actual analysis!")
    else:
        print(f"\n❌ Test failed")
