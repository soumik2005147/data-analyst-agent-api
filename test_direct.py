#!/usr/bin/env python3
"""
Test analysis functions directly
"""

import sys
import os
sys.path.append(os.getcwd())

from production_ready_api import analyze_sales_data, analyze_weather_data, analyze_network_data

def test_direct_analysis():
    print("Testing direct analysis functions...")
    
    # Test sales analysis
    print("\n1. Testing sales analysis...")
    try:
        result = analyze_sales_data("What is the total sales?")
        print(f"Sales result keys: {list(result.keys())}")
        if 'error' in result:
            print(f"Sales error: {result['error']}")
        else:
            print(f"Total sales: {result.get('total_sales', 'Missing')}")
    except Exception as e:
        print(f"Sales analysis error: {e}")
    
    # Test weather analysis  
    print("\n2. Testing weather analysis...")
    try:
        result = analyze_weather_data("What is the average temperature?")
        print(f"Weather result keys: {list(result.keys())}")
        if 'error' in result:
            print(f"Weather error: {result['error']}")
        else:
            print(f"Avg temp: {result.get('average_temp_c', 'Missing')}")
    except Exception as e:
        print(f"Weather analysis error: {e}")
    
    # Test network analysis
    print("\n3. Testing network analysis...")
    try:
        result = analyze_network_data("How many edges are there?")
        print(f"Network result keys: {list(result.keys())}")
        if 'error' in result:
            print(f"Network error: {result['error']}")
        else:
            print(f"Edge count: {result.get('edge_count', 'Missing')}")
    except Exception as e:
        print(f"Network analysis error: {e}")

if __name__ == "__main__":
    test_direct_analysis()
