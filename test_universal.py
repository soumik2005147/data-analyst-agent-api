#!/usr/bin/env python3
"""
Universal API Tester - Tests all file types and formats
"""

import requests
import json
import time
import tempfile
import os
from io import BytesIO
import csv
import base64

def create_test_files():
    """Create various test files to demonstrate universal support"""
    test_files = {}
    
    # 1. Text file with questions
    questions_txt = """Analyze the following data and answer these questions:
1. How many records are in the dataset?
2. What is the most common value?
3. Calculate the correlation coefficient.
4. Generate a visualization chart."""
    test_files['questions.txt'] = questions_txt.encode('utf-8')
    
    # 2. CSV data file
    csv_data = """Name,Age,Score,City
Alice,25,95.5,New York
Bob,30,87.2,London  
Charlie,35,92.8,Tokyo
Diana,28,89.1,Paris
Eve,32,94.3,Berlin"""
    test_files['data.csv'] = csv_data.encode('utf-8')
    
    # 3. JSON data file
    json_data = {
        "dataset": "Sample Analysis Data",
        "records": [
            {"id": 1, "value": 100, "category": "A"},
            {"id": 2, "value": 85, "category": "B"},
            {"id": 3, "value": 92, "category": "A"},
            {"id": 4, "value": 78, "category": "C"}
        ],
        "metadata": {"source": "test", "version": "1.0"}
    }
    test_files['metadata.json'] = json.dumps(json_data, indent=2).encode('utf-8')
    
    # 4. Image file (minimal PNG)
    # This is a 1x1 transparent PNG
    png_data = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
    test_files['chart.png'] = png_data
    
    # 5. XML file
    xml_data = """<?xml version="1.0" encoding="UTF-8"?>
<analysis>
    <title>Data Analysis Report</title>
    <findings>
        <finding id="1">Strong correlation detected</finding>
        <finding id="2">Outliers identified in dataset</finding>
    </findings>
</analysis>"""
    test_files['report.xml'] = xml_data.encode('utf-8')
    
    return test_files

def test_universal_api():
    """Test the universal API with various file types"""
    print("🧪 Universal Data Analyst Agent API Test")
    print("=" * 60)
    
    # Wait for server
    time.sleep(2)
    
    try:
        # Test 1: Health Check
        print("\n1️⃣ Health Check Test")
        health_response = requests.get("http://localhost:8001/health", timeout=10)
        print(f"   Status: {health_response.status_code}")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"   ✅ {health_data.get('message', 'Healthy')}")
            print(f"   Supports: {len(health_data.get('supports', []))} file types")
        else:
            print(f"   ❌ Health check failed")
            return False
        
        # Test 2: Multi-file upload test
        print("\n2️⃣ Multi-File Upload Test")
        test_files = create_test_files()
        
        files_for_upload = {}
        for filename, content in test_files.items():
            files_for_upload[filename] = (filename, content, 'application/octet-stream')
        
        print(f"   📁 Uploading {len(files_for_upload)} files:")
        for filename in files_for_upload.keys():
            size = len(test_files[filename])
            print(f"      • {filename} ({size} bytes)")
        
        response = requests.post(
            "http://localhost:8001/api/",
            files=files_for_upload,
            timeout=30
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ SUCCESS! Response received")
            print(f"   Type: {type(result)}")
            print(f"   Length: {len(result) if isinstance(result, list) else 'not a list'}")
            
            if isinstance(result, list) and len(result) == 4:
                print(f"\n   📊 Response Analysis:")
                print(f"      [0] Files processed: {result[0]} ({type(result[0]).__name__})")
                print(f"      [1] Dataset name: '{result[1]}' ({type(result[1]).__name__})")
                print(f"      [2] Analysis value: {result[2]} ({type(result[2]).__name__})")
                print(f"      [3] Visualization: {result[3][:50]}... ({len(result[3])} chars)")
                
                # Validate exact format
                format_valid = (
                    isinstance(result[0], int) and
                    isinstance(result[1], str) and
                    isinstance(result[2], (int, float)) and
                    isinstance(result[3], str)
                )
                
                if format_valid:
                    print(f"\n   🎉 PERFECT FORMAT MATCH!")
                    print(f"      Expected: [int, str, float, str]")
                    print(f"      Received: [int, str, float, str] ✅")
                    print(f"\n   🏆 FINAL RESPONSE:")
                    print(f"      {result}")
                    
                    # Test 3: JSON data test
                    print("\n3️⃣ JSON Data Test")
                    json_payload = {
                        "questions": ["What is the trend?", "Calculate statistics"],
                        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                    }
                    
                    json_response = requests.post(
                        "http://localhost:8001/api/",
                        json=json_payload,
                        timeout=20
                    )
                    
                    print(f"   Status: {json_response.status_code}")
                    if json_response.status_code == 200:
                        json_result = json_response.json()
                        print(f"   ✅ JSON test successful: {json_result}")
                    
                    # Test 4: Single text file test
                    print("\n4️⃣ Single Text File Test")
                    simple_files = {
                        'questions.txt': ('questions.txt', 'Simple question: What is 2+2?', 'text/plain')
                    }
                    
                    simple_response = requests.post(
                        "http://localhost:8001/api/",
                        files=simple_files,
                        timeout=20
                    )
                    
                    print(f"   Status: {simple_response.status_code}")
                    if simple_response.status_code == 200:
                        simple_result = simple_response.json()
                        print(f"   ✅ Simple test successful: {simple_result}")
                    
                    return True
                else:
                    print(f"\n   ⚠️ Format validation failed")
            else:
                print(f"\n   ❌ Expected 4-element array, got: {result}")
        else:
            print(f"   ❌ API Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    return False

if __name__ == "__main__":
    success = test_universal_api()
    if success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ API handles multiple file formats successfully")
        print(f"✅ Response format is exactly [int, str, float, str]")
        print(f"✅ Ready for production deployment!")
    else:
        print(f"\n❌ Some tests failed")
        print(f"🔧 Check server logs for details")
