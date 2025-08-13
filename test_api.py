"""
Test script for the Data Analyst Agent API
"""

import requests
import json
import tempfile
import os


def test_api_local():
    """Test the API running locally"""
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"Root endpoint: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Root endpoint failed: {e}")
    
    # Test main API endpoint with sample questions
    sample_questions = """
    Scrape the list of highest grossing films from Wikipedia. It is at the URL:
    https://en.wikipedia.org/wiki/List_of_highest-grossing_films

    Answer the following questions and respond with a JSON array of strings containing the answer.

    1. How many $2 bn movies were released before 2000?
    2. Which is the earliest film that grossed over $1.5 bn?
    3. What's the correlation between the Rank and Peak?
    4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
       Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes.
    """
    
    # Create temporary questions file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_questions)
        questions_file = f.name
    
    try:
        # Test API endpoint
        with open(questions_file, 'rb') as f:
            files = {'questions.txt': f}
            response = requests.post(f"{base_url}/api/", files=files)
        
        print(f"API Response Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"API Response: {result}")
            
            # Validate response format
            if isinstance(result, list) and len(result) == 4:
                print("✓ Response format is correct (4-element array)")
                print(f"✓ Answer 1 (count): {result[0]}")
                print(f"✓ Answer 2 (film): {result[1]}")
                print(f"✓ Answer 3 (correlation): {result[2]}")
                print(f"✓ Answer 4 (plot): {result[3][:50]}..." if isinstance(result[3], str) else result[3])
            else:
                print(f"✗ Unexpected response format: {type(result)}")
        else:
            print(f"API Error: {response.text}")
            
    except Exception as e:
        print(f"API test failed: {e}")
    
    finally:
        # Cleanup
        if os.path.exists(questions_file):
            os.unlink(questions_file)


def test_with_sample_data():
    """Test API with sample CSV data"""
    base_url = "http://localhost:8000"
    
    # Create sample CSV data
    sample_csv = """name,age,salary
John,25,50000
Jane,30,60000
Bob,35,70000
Alice,28,55000"""
    
    sample_questions = """
    Analyze the uploaded CSV file and provide:
    1. Basic statistics
    2. Average salary
    3. Age distribution
    """
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_questions)
        questions_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(sample_csv)
        data_file = f.name
    
    try:
        # Test API with both files
        with open(questions_file, 'rb') as qf, open(data_file, 'rb') as df:
            files = {
                'questions.txt': qf,
                'data.csv': df
            }
            response = requests.post(f"{base_url}/api/", files=files)
        
        print(f"CSV Analysis Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"CSV Analysis Result: {json.dumps(result, indent=2)}")
        else:
            print(f"CSV Analysis Error: {response.text}")
            
    except Exception as e:
        print(f"CSV test failed: {e}")
    
    finally:
        # Cleanup
        for file_path in [questions_file, data_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)


if __name__ == "__main__":
    print("Testing Data Analyst Agent API...")
    print("=" * 50)
    
    # Test basic functionality
    test_api_local()
    
    print("\n" + "=" * 50)
    print("Testing with sample data...")
    
    # Test with sample data
    test_with_sample_data()
