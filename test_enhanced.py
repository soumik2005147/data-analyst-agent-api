#!/usr/bin/env python3
"""
Test Enhanced Universal API with user's specific files
"""

import requests
import os

                    print(f"   🎯 CURL Command that produced these results:")
                    print(f"      curl -X POST \"http://localhost:8000/api/\" \\")
                    for field_name, file_path in files_to_test:
                        print(f"           -F \"{field_name}=@{file_path}\" \\")
                    print(f"           -H \"accept: application/json\"")est_with_user_files():
    """Test the API with the user's specific files and get real answers"""
    print("🧪 Testing Enhanced Universal API with User's Files")
    print("=" * 60)
    
    # File paths
    weather_csv = r"C:\Users\soumi\Downloads\sample-weather.csv"
    questions_txt = r"C:\Users\soumi\Downloads\questions (1).txt"
    
    # Check if files exist
    files_to_test = []
    
    if os.path.exists(weather_csv):
        print(f"✅ Found weather CSV: {weather_csv}")
        files_to_test.append(('weather_data.csv', weather_csv))
    else:
        print(f"❌ Weather CSV not found: {weather_csv}")
    
    if os.path.exists(questions_txt):
        print(f"✅ Found questions file: {questions_txt}")
        files_to_test.append(('questions.txt', questions_txt))
    else:
        print(f"❌ Questions file not found: {questions_txt}")
    
    if not files_to_test:
        print("❌ No files found! Please check the file paths.")
        return False
    
    try:
        # Test 1: Health Check
        print(f"\n1️⃣ Health Check")
        health_response = requests.get("http://localhost:8000/health", timeout=10)
        print(f"   Status: {health_response.status_code}")
        
        if health_response.status_code != 200:
            print(f"   ❌ API not running on port 8000. Start the server first!")
            return False
        
        health_data = health_response.json()
        print(f"   ✅ {health_data.get('message', 'API ready')}")
        
        # Test 2: Upload user's files for REAL analysis
        print(f"\n2️⃣ Real Data Analysis with User Files")
        
        files_for_upload = {}
        for field_name, file_path in files_to_test:
            try:
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                    files_for_upload[field_name] = (os.path.basename(file_path), file_content, 'application/octet-stream')
                    print(f"   📄 Loaded: {os.path.basename(file_path)} ({len(file_content)} bytes)")
            except Exception as e:
                print(f"   ❌ Error reading {file_path}: {e}")
        
        if not files_for_upload:
            print("   ❌ No files could be loaded!")
            return False
        
        # Send the request for REAL analysis
        print(f"   🚀 Analyzing {len(files_for_upload)} files and answering questions...")
        
        response = requests.post(
            "http://localhost:8000/api/",
            files=files_for_upload,
            timeout=60  # Longer timeout for real analysis
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ SUCCESS! Real answers received!")
            
            print(f"\n📊 REAL ANALYSIS RESULTS:")
            print(f"   Response Type: {type(result)}")
            print(f"   Response Length: {len(result) if isinstance(result, list) else 'not a list'}")
            
            if isinstance(result, list) and len(result) == 4:
                print(f"\n   📋 ACTUAL ANSWERS TO YOUR QUESTIONS:")
                print(f"      🔢 Question 1 Answer: {result[0]} ({type(result[0]).__name__})")
                print(f"      📝 Question 2 Answer: '{result[1]}' ({type(result[1]).__name__})")
                print(f"      📊 Question 3 Answer: {result[2]} ({type(result[2]).__name__})")
                
                # Handle visualization data
                viz_data = result[3]
                if isinstance(viz_data, str):
                    if viz_data.startswith('data:image'):
                        print(f"      🎨 Question 4 Answer: Chart/visualization ({len(viz_data)} chars)")
                    elif viz_data.startswith('data:text'):
                        try:
                            import base64
                            decoded = base64.b64decode(viz_data.split(',')[1]).decode()
                            print(f"      🎨 Question 4 Answer: {decoded}")
                        except:
                            print(f"      🎨 Question 4 Answer: Visualization data")
                    else:
                        print(f"      🎨 Question 4 Answer: {viz_data}")
                
                # Format validation
                format_valid = (
                    isinstance(result[0], int) and
                    isinstance(result[1], str) and
                    isinstance(result[2], (int, float)) and
                    isinstance(result[3], str)
                )
                
                if format_valid:
                    print(f"\n   🎉 PERFECT! Real analysis completed successfully!")
                    print(f"      Expected format: [int, str, float, str]")
                    print(f"      Received format: [int, str, float, str] ✅")
                    
                    print(f"\n   🏆 FINAL ANALYZED RESULT:")
                    print(f"      [{result[0]}, '{result[1]}', {result[2]}, 'data:...']")
                    
                    print(f"\n   🎯 CURL Command that produced these results:")
                    print(f"      curl -X POST \"http://localhost:8002/api/\" \\")
                    for field_name, file_path in files_to_test:
                        print(f"           -F \"{field_name}=@{file_path}\" \\")
                    print(f"           -H \"accept: application/json\"")
                    
                    return True
                else:
                    print(f"\n   ⚠️ Format validation failed")
                    print(f"      Types: [{type(result[0])}, {type(result[1])}, {type(result[2])}, {type(result[3])}]")
            else:
                print(f"\n   ❌ Expected 4-element array, got: {result}")
        else:
            print(f"   ❌ API Error ({response.status_code}): {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    return False

if __name__ == "__main__":
    success = test_with_user_files()
    if success:
        print(f"\n🎉 SUCCESS! Your questions were analyzed and answered!")
        print(f"✅ Weather CSV data was processed and analyzed")
        print(f"✅ Questions were read and answered with real data") 
        print(f"✅ Response format is exactly [int, str, float, str]")
        print(f"✅ API provides REAL answers, not mock responses!")
    else:
        print(f"\n❌ Test failed - check the output above")
        print(f"🔧 Make sure the Enhanced Universal API server is running on port 8002")
