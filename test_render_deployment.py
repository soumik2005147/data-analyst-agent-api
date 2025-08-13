#!/usr/bin/env python3
"""
Test script for deployed Data Analyst Agent API on Render
Usage: python test_render_deployment.py <your-render-app-url>
"""

import requests
import sys
import json

def test_health_endpoint(base_url):
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_root_endpoint(base_url):
    """Test the root endpoint"""
    print("\n🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print("✅ Root endpoint accessible")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_api_endpoint_with_sample_data(base_url):
    """Test the main API endpoint with sample data"""
    print("\n🔍 Testing API endpoint with sample CSV data...")
    
    # Create a simple CSV data
    csv_content = """Product,Sales,Region
Product A,100,North
Product B,150,South
Product C,80,East
Product D,120,West"""
    
    # Create questions file
    questions_content = """Analyze the provided sales data and answer:

1. What is the total sales across all products?
2. Which region has the highest sales?
3. Create a bar chart showing sales by region.
4. What is the average sales per product?"""
    
    try:
        files = [
            ('files', ('questions.txt', questions_content, 'text/plain')),
            ('files', ('test_data.csv', csv_content, 'text/csv'))
        ]
        
        response = requests.post(
            f"{base_url}/api/", 
            files=files,
            timeout=120  # 2 minute timeout for processing
        )
        
        if response.status_code == 200:
            print("✅ API endpoint working")
            result = response.json()
            print(f"Analysis generated: {len(result.get('analysis', ''))} characters")
            if result.get('visualizations'):
                print(f"Visualizations created: {len(result['visualizations'])}")
            return True
        else:
            print(f"❌ API endpoint failed: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error details: {error_detail}")
            except:
                print(f"Error text: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ API endpoint error: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_render_deployment.py <your-render-app-url>")
        print("Example: python test_render_deployment.py https://your-app.onrender.com")
        print("\nGet your Render app URL from: https://dashboard.render.com/")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    print(f"🚀 Testing Data Analyst Agent API at: {base_url}")
    print("=" * 60)
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_health_endpoint(base_url):
        tests_passed += 1
    
    if test_root_endpoint(base_url):
        tests_passed += 1
    
    if test_api_endpoint_with_sample_data(base_url):
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your API is working correctly on Render.")
        print("✅ Ready for production use!")
    elif tests_passed >= 2:
        print("⚠️  Most tests passed. API is mostly functional.")
        print("💡 Check logs in Render dashboard for any issues.")
    else:
        print("❌ Multiple tests failed. Check your deployment configuration.")
        print("🔧 Common issues:")
        print("   - OPENAI_API_KEY not set in Render environment variables")
        print("   - App still starting up (wait 2-3 minutes)")
        print("   - Build failed (check Render logs)")
    
    sys.exit(0 if tests_passed >= 2 else 1)

if __name__ == "__main__":
    main()
