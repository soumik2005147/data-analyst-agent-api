#!/usr/bin/env python3
"""
Quick test to debug API issues
"""

import requests
import json

def quick_test():
    base_url = "http://localhost:8000"
    
    # Test simple questions
    questions_content = "Analyze `sample-sales.csv`. What is the total sales across all regions?"
    
    csv_content = """date,region,sales_amount
2024-01-01,North,80
2024-01-02,South,90
2024-01-03,East,100"""

    try:
        files = [
            ('files', ('questions.txt', questions_content, 'text/plain')),
            ('files', ('sample-sales.csv', csv_content, 'text/csv'))
        ]
        
        print("Testing file upload...")
        response = requests.post(f"{base_url}/api/", files=files, timeout=30)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    quick_test()
