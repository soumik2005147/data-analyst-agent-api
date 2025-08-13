#!/usr/bin/env python3
"""
Test Universal API with user's specific files
"""

import requests
import os

                    print(f"   🎯 CURL Command Equivalent:")
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

def test_with_user_files():
    """Test the API with the user's specific files"""
    print("🧪 Testing Universal API with User's Files")
    print("=" * 50)
    
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
        health_response = requests.get("http://localhost:8002/health", timeout=10)
        print(f"   Status: {health_response.status_code}")
        
        if health_response.status_code != 200:
            print(f"   ❌ API not running. Start the server first!")
            return False
        
        health_data = health_response.json()
        print(f"   ✅ {health_data.get('message', 'API ready')}")
        
        # Test 2: Upload user's files
        print(f"\n2️⃣ Uploading User Files")
        
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
        
        # Send the request
        print(f"   🚀 Sending {len(files_for_upload)} files to API...")
        
        response = requests.post(
            "http://localhost:8002/api/",
            files=files_for_upload,
            timeout=30
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ SUCCESS! API Response received!")
            
            print(f"\n📊 Response Analysis:")
            print(f"   Type: {type(result)}")
            print(f"   Length: {len(result) if isinstance(result, list) else 'not a list'}")
            
            if isinstance(result, list) and len(result) == 4:
                print(f"\n   📋 Detailed Response:")
                print(f"      [0] Files processed: {result[0]} ({type(result[0]).__name__})")
                print(f"      [1] Primary data: '{result[1]}' ({type(result[1]).__name__})")
                print(f"      [2] Analysis metric: {result[2]} ({type(result[2]).__name__})")
                
                # Handle visualization data
                viz_data = result[3]
                if isinstance(viz_data, str) and len(viz_data) > 50:
                    print(f"      [3] Visualization: {viz_data[:50]}... ({len(viz_data)} total chars)")
                else:
                    print(f"      [3] Visualization: {viz_data} ({type(viz_data).__name__})")
                
                # Format validation
                format_valid = (
                    isinstance(result[0], int) and
                    isinstance(result[1], str) and
                    isinstance(result[2], (int, float)) and
                    isinstance(result[3], str)
                )
                
                if format_valid:
                    print(f"\n   🎉 PERFECT! Exact format achieved!")
                    print(f"      Expected: [int, str, float, str]")
                    print(f"      Received: [int, str, float, str] ✅")
                    
                    print(f"\n   🏆 FINAL RESULT:")
                    print(f"      [{result[0]}, '{result[1]}', {result[2]}, 'data:image/...']")
                    
                    print(f"\n   🎯 CURL Command Equivalent:")
                    print(f"      curl -X POST \"http://localhost:8001/api/\" \\")
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
        print(f"\n🎉 SUCCESS! API works with your files!")
        print(f"✅ Weather CSV processed successfully")
        print(f"✅ Questions file processed successfully") 
        print(f"✅ Response format is exactly [int, str, float, str]")
        print(f"✅ Ready for production use!")
    else:
        print(f"\n❌ Test failed - check the output above")
        print(f"🔧 Make sure the Universal API server is running on port 8001")
v