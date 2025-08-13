#!/usr/bin/env python3
"""
Test the Data Analyst Agent API with the provided sales data
"""

import requests
import json
import time
import os

def test_sales_data_analysis():
    """Test the API with sales data and questions"""
    
    api_url = "http://localhost:8000/api/"
    
    # Wait a moment for API to be ready
    print("🔄 Waiting for API to start...")
    for i in range(10):
        try:
            health_response = requests.get("http://localhost:8000/health", timeout=2)
            if health_response.status_code == 200:
                print("✅ API is ready!")
                break
        except:
            time.sleep(2)
            print(f"   Attempt {i+1}/10...")
    else:
        print("❌ API failed to start. Please check the server.")
        return
    
    # Prepare test files
    sales_data_path = "test_sales.csv"
    questions_path = "test_questions.txt"
    
    if not os.path.exists(sales_data_path):
        print(f"❌ Sales data file not found: {sales_data_path}")
        return
    
    if not os.path.exists(questions_path):
        print(f"❌ Questions file not found: {questions_path}")
        return
    
    print(f"\n📊 Testing Sales Data Analysis")
    print("=" * 50)
    
    # Read files for display
    with open(questions_path, 'r') as f:
        questions_content = f.read()
    
    print("📋 Questions being analyzed:")
    print(questions_content[:200] + "..." if len(questions_content) > 200 else questions_content)
    print()
    
    # Send request to API
    try:
        with open(questions_path, 'rb') as qf, open(sales_data_path, 'rb') as sf:
            files = {
                'files': ('questions.txt', qf, 'text/plain')
            }
            
            # Add the sales data file  
            files['files'] = [
                ('questions.txt', open(questions_path, 'rb'), 'text/plain'),
                ('test_sales.csv', open(sales_data_path, 'rb'), 'text/csv')
            ]
            
            print("🚀 Sending analysis request...")
            response = requests.post(api_url, files={
                'files': ('questions.txt', open(questions_path, 'rb')),
                'files2': ('sales_data.csv', open(sales_data_path, 'rb'))
            }, timeout=180)
        
        print(f"📨 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("\n✅ Analysis Results:")
                print("=" * 30)
                
                if isinstance(result, list):
                    for i, answer in enumerate(result, 1):
                        if isinstance(answer, str) and len(answer) > 100:
                            print(f"Answer {i}: {answer[:100]}... (truncated)")
                        else:
                            print(f"Answer {i}: {answer}")
                
                elif isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"{key}: {value[:100]}... (truncated)")
                        else:
                            print(f"{key}: {value}")
                else:
                    print(f"Result: {result}")
                
                print("\n🎯 Analysis Summary:")
                if isinstance(result, list):
                    print(f"   📊 Returned {len(result)} answers")
                elif isinstance(result, dict):
                    print(f"   📊 Returned {len(result)} key-value pairs")
                
                # Check for visualizations
                if isinstance(result, list):
                    viz_count = sum(1 for item in result if isinstance(item, str) and item.startswith('data:image'))
                    if viz_count > 0:
                        print(f"   📈 Generated {viz_count} visualizations")
                        
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON response: {response.text[:500]}...")
                
        else:
            print(f"❌ API Error: {response.text[:500]}...")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure it's running at http://localhost:8000")
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_api_capabilities():
    """Test API basic capabilities"""
    print("\n🔧 Testing API Capabilities:")
    print("-" * 30)
    
    # Test health endpoint
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"Health Check: {health_response.status_code} - {health_response.json()}")
    except Exception as e:
        print(f"Health Check Failed: {e}")
    
    # Test root endpoint
    try:
        root_response = requests.get("http://localhost:8000/", timeout=5)
        print(f"Root Endpoint: {root_response.status_code}")
        if root_response.status_code == 200:
            info = root_response.json()
            print(f"   API: {info.get('message', 'Unknown')}")
            print(f"   Version: {info.get('version', 'Unknown')}")
    except Exception as e:
        print(f"Root Endpoint Failed: {e}")

if __name__ == "__main__":
    print("🧪 Data Analyst Agent API Test")
    print("=" * 50)
    
    # Test basic capabilities first
    test_api_capabilities()
    
    # Test with sales data
    test_sales_data_analysis()
    
    print("\n" + "=" * 50)
    print("🏁 Test Complete!")
