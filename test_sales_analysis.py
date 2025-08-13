import requests
import json

# Test the AI-enhanced data analysis API with sales data
def test_sales_analysis():
    base_url = "http://localhost:8005"
    
    # Test 1: Health check
    print("=" * 50)
    print("Testing API Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test 2: Upload and analyze sales data
    print("\n" + "=" * 50)
    print("Testing Sales Data Analysis...")
    
    try:
        # Read the sales data file and questions file
        with open('sales_data.csv', 'rb') as csv_file, open('sales_questions.txt', 'rb') as questions_file:
            files = [
                ('files', ('sales_data.csv', csv_file, 'text/csv')),
                ('files', ('sales_questions.txt', questions_file, 'text/plain'))
            ]
            
            print("Uploading sales data and questions file...")
            print("Questions file contains sales analysis questions")
            
            response = requests.post(f"{base_url}/api/", files=files)
            
            print(f"\nResponse Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Success: {result.get('success', False)}")
                print(f"Filename: {result.get('filename', 'N/A')}")
                print(f"File Size: {result.get('file_size', 'N/A')} bytes")
                print(f"Data Shape: {result.get('data_shape', 'N/A')}")
                
                print("\n" + "-" * 40)
                print("AI ANALYSIS RESULTS:")
                print("-" * 40)
                analysis = result.get('analysis', 'No analysis available')
                print(analysis)
                
                # Print insights if available
                if 'insights' in result:
                    print("\n" + "-" * 40)
                    print("KEY INSIGHTS:")
                    print("-" * 40)
                    for insight in result['insights']:
                        print(f"• {insight}")
                
            else:
                print(f"Error: {response.text}")
                
    except Exception as e:
        print(f"Sales analysis test failed: {e}")

if __name__ == "__main__":
    test_sales_analysis()
