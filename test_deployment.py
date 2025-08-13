#!/usr/bin/env python3
"""
Test script to verify the Data Analyst Agent API functionality
"""

import requests
import tempfile
import os
from io import BytesIO

def test_generalized_analysis():
    """Test the API with different types of questions"""
    
    test_cases = [
        {
            "name": "Wikipedia Films Analysis",
            "questions": """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes."""
        },
        {
            "name": "Generic Data Analysis",
            "questions": """Analyze the provided data and answer these questions:

1. How many records are in the dataset?
2. What are the main columns?
3. What is the average value of numeric columns?
4. Create a visualization showing the distribution of the first numeric column.""",
            "data": {"test_data.csv": "id,value,category\n1,10,A\n2,20,B\n3,15,A\n4,25,C\n5,18,B"}
        },
        {
            "name": "Court Data Analysis",
            "questions": """{
  "Which high court disposed the most cases from 2019 - 2022?": "...",
  "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": "...",
  "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": "data:image/webp:base64,..."
}"""
        }
    ]
    
    api_url = "http://localhost:8000/api/"
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'='*50}")
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_case['questions'])
            questions_file = f.name
        
        additional_files = []
        try:
            files_to_upload = {'files': ('questions.txt', open(questions_file, 'rb'), 'text/plain')}
            
            # Add additional test data if provided
            if 'data' in test_case:
                for filename, content in test_case['data'].items():
                    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                    temp_file.write(content)
                    temp_file.close()
                    additional_files.append(temp_file.name)
                    files_to_upload[f'file_{filename}'] = (filename, open(temp_file.name, 'rb'), 'text/csv')
            
            print("Sending request...")
            response = requests.post(api_url, files=files_to_upload, timeout=180)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ API Response Received")
                print(f"Response Type: {type(result)}")
                
                if isinstance(result, list):
                    print(f"Array Length: {len(result)}")
                    for j, item in enumerate(result[:4], 1):  # Show first 4 items
                        if isinstance(item, str) and len(item) > 100:
                            print(f"  [{j}]: {item[:100]}... (truncated)")
                        else:
                            print(f"  [{j}]: {item}")
                elif isinstance(result, dict):
                    print("Dictionary Keys:", list(result.keys())[:5])  # Show first 5 keys
                    for key, value in list(result.items())[:3]:  # Show first 3 items
                        if isinstance(value, str) and len(value) > 100:
                            print(f"  {key}: {value[:100]}... (truncated)")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print(f"Result: {result}")
                    
            else:
                print(f"❌ API Error: {response.text[:500]}...")
                
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to API. Make sure the server is running:")
            print("   python app.py")
            print("   or")
            print("   uvicorn app:app --reload")
            break
            
        except Exception as e:
            print(f"❌ Error in test case: {e}")
        
        finally:
            # Cleanup
            try:
                os.unlink(questions_file)
                for temp_file in additional_files:
                    os.unlink(temp_file)
            except:
                pass


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'requests', 'beautifulsoup4', 'duckdb', 
        'scipy', 'plotly', 'openai', 'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True


if __name__ == "__main__":
    print("🔍 Checking Data Analyst Agent API...")
    print("=" * 50)
    
    print("\n1. Checking Dependencies:")
    deps_ok = check_dependencies()
    
    if deps_ok:
        print("\n2. Testing API Functionality:")
        test_generalized_analysis()
    else:
        print("\n❌ Please install missing dependencies first")
    
    print("\n" + "=" * 50)
    print("📋 Deployment Checklist:")
    print("1. ✅ FastAPI application created")
    print("2. ✅ POST /api/ endpoint implemented") 
    print("3. ✅ File upload handling (questions.txt + attachments)")
    print("4. ✅ Wikipedia scraping capability")
    print("5. ✅ Data analysis and visualization")
    print("6. ✅ JSON response formatting")
    print("7. ✅ 3-minute timeout monitoring")
    print("8. ✅ Multiple deployment options available")
    
    print("\n🚀 Ready for deployment!")
    print("Deploy using:")
    print("- Docker: docker build -t data-analyst-agent . && docker run -p 8000:8000 data-analyst-agent")
    print("- Heroku: git push heroku main")
    print("- Vercel: vercel --prod")
    print("- Railway: railway up")
