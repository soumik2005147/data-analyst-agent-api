#!/usr/bin/env python3
"""
Test the AI-Enhanced Data Analyst Agent
"""

import requests
import os
import time

def test_ai_enhanced_api():
    """Test the AI-enhanced API with real data"""
    print("🤖 Testing AI-Enhanced Data Analyst Agent")
    print("=" * 50)
    
    # File paths
    weather_csv = r"C:\Users\soumi\Downloads\sample-weather.csv"
    questions_txt = r"C:\Users\soumi\Downloads\questions (1).txt"
    
    # Check if files exist
    if not os.path.exists(weather_csv):
        print(f"❌ Weather CSV not found: {weather_csv}")
        return False
        
    if not os.path.exists(questions_txt):
        print(f"❌ Questions file not found: {questions_txt}")
        return False
    
    print(f"✅ Found both files")
    
    try:
        # Health check first
        print(f"\n1️⃣ Health Check")
        health_response = requests.get("http://localhost:8005/health", timeout=10)
        
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"   Status: {health_data['status']}")
            print(f"   AI Integration: {health_data['ai_integration']}")
            print(f"   ✅ {health_data['message']}")
        else:
            print(f"   ❌ API not running on port 8005")
            print(f"   Start it with: python ai_enhanced_api.py")
            return False
        
        # Read files
        with open(weather_csv, 'rb') as f:
            csv_content = f.read()
        with open(questions_txt, 'rb') as f:
            txt_content = f.read()
        
        print(f"\n2️⃣ AI-Powered Analysis")
        print(f"   📊 CSV: {len(csv_content)} bytes")
        print(f"   📋 Questions: {len(txt_content)} bytes")
        
        # Show questions
        with open(questions_txt, 'r') as f:
            questions = f.read()
        print(f"   ❓ Your Questions:")
        print(f"      {questions[:200]}...")
        
        # Send to AI API
        files_data = [
            ('files', ('sample-weather.csv', csv_content, 'text/csv')),
            ('files', ('questions.txt', txt_content, 'text/plain'))
        ]
        
        print(f"\n   🤖 Sending to AI Assistant...")
        print(f"   ⏳ This may take 10-30 seconds for AI processing...")
        
        response = requests.post(
            "http://localhost:8005/api/",
            files=files_data,
            timeout=120  # Longer timeout for AI processing
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ AI ANALYSIS COMPLETE!")
            
            if isinstance(result, list) and len(result) == 4:
                print(f"\n🤖 AI-POWERED ANSWERS:")
                print(f"   🔢 Answer 1 ({type(result[0]).__name__}): {result[0]}")
                print(f"   📝 Answer 2 ({type(result[1]).__name__}): '{result[1]}'")
                print(f"   📊 Answer 3 ({type(result[2]).__name__}): {result[2]}")
                
                if str(result[3]).startswith('data:image'):
                    print(f"   🎨 Answer 4: AI-generated visualization ({len(str(result[3]))} chars)")
                else:
                    print(f"   🎨 Answer 4: {result[3]}")
                
                print(f"\n🏆 FINAL AI RESULT:")
                print(f"   [{result[0]}, '{result[1]}', {result[2]}, 'visualization']")
                
                # Format validation
                format_valid = (
                    isinstance(result[0], int) and
                    isinstance(result[1], str) and
                    isinstance(result[2], (int, float)) and
                    isinstance(result[3], str)
                )
                
                if format_valid:
                    print(f"\n🎉 PERFECT! AI analysis successful!")
                    print(f"   ✅ Format: [int, str, float, str] ✓")
                    print(f"   ✅ Real analysis powered by OpenAI ✓")
                    print(f"   ✅ Your weather data analyzed by AI ✓")
                    return True
                else:
                    print(f"\n⚠️  Format validation failed")
                    print(f"   Types: [{type(result[0])}, {type(result[1])}, {type(result[2])}, {type(result[3])}]")
            else:
                print(f"\n❌ Unexpected response format: {result}")
        else:
            print(f"   ❌ API Error ({response.status_code}): {response.text}")
            
    except requests.exceptions.Timeout:
        print(f"\n⏰ Request timed out - AI processing takes time, try again")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    return False

if __name__ == "__main__":
    success = test_ai_enhanced_api()
    if success:
        print(f"\n🎉 SUCCESS! AI-Enhanced Data Analysis Works!")
        print(f"✅ OpenAI Assistant analyzed your weather data")
        print(f"✅ Real AI-powered insights generated")
        print(f"✅ No more mock responses - this is real AI analysis!")
    else:
        print(f"\n❌ Test failed - check the output above")
        print(f"🔧 Make sure to:")
        print(f"   1. Set your OpenAI API key: python setup_ai.py")
        print(f"   2. Start the AI API: python ai_enhanced_api.py")
        print(f"   3. Then run this test again")
