#!/usr/bin/env python3
"""
Test script to verify the API handles multipart/form-data correctly
"""

import requests
import json
import pandas as pd
from io import StringIO

def test_multipart_api():
    """Test the /api/ endpoint with multipart/form-data"""
    
    # Create sample CSV data
    sample_data = """Region,Sales,Product
North,15000,Widget A
South,12000,Widget B  
East,18000,Widget A
West,9000,Widget C
North,22000,Widget B
South,14000,Widget A"""
    
    # Create sample questions
    sample_questions = """What are the total sales?
Which region has the highest sales?
What's the average sales per region?
Create a chart showing sales by region."""
    
    # Prepare multipart data
    files = {
        'files': ('sales_data.csv', sample_data, 'text/csv'),
        'files': ('questions.txt', sample_questions, 'text/plain')
    }
    
    try:
        print("🧪 Testing multipart/form-data API endpoint...")
        response = requests.post('http://localhost:8000/api/', files=files, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Response successful!")
            print(f"Response keys: {list(result.keys())}")
            
            # Check for key fields
            if 'total_sales' in result:
                print(f"Total Sales: {result.get('total_sales')}")
            if 'top_region' in result:
                print(f"Top Region: {result.get('top_region')}")
            if 'bar_chart' in result:
                print(f"Chart generated: {'Yes' if result.get('bar_chart') else 'No'}")
                
            print("\n📊 Full Response:")
            print(json.dumps(result, indent=2))
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_multipart_api()
