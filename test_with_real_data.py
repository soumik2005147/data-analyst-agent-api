import requests
import io
import csv

def test_api_with_real_sales_data():
    """Test API with actual sales data"""
    
    # Create sample sales CSV data
    sales_data = """date,region,sales_amount,product
2024-01-01,North,1500,Widget A
2024-01-02,South,2000,Widget B  
2024-01-03,East,1200,Widget A
2024-01-04,North,1800,Widget C
2024-01-05,West,2500,Widget B
2024-01-06,South,1900,Widget A
2024-01-07,East,1600,Widget C
2024-01-08,North,2200,Widget B
2024-01-09,West,1400,Widget A
2024-01-10,South,2100,Widget C"""
    
    # Create questions content
    questions_content = """Analyze the sales data and provide:
1. Total sales amount
2. Top performing region
3. Median sales value
4. Sales trend analysis with charts"""
    
    # Prepare multipart form data
    files = {
        'files': ('questions.txt', questions_content, 'text/plain'),
        'files': ('sales_data.csv', sales_data, 'text/csv')
    }
    
    try:
        # Make API request
        response = requests.post('http://localhost:8000/api/', files=files, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Response successful!")
            print(f"Response keys: {list(data.keys())}")
            
            # Check specific values
            print(f"\nAnalysis Results:")
            print(f"Total Sales: {data.get('total_sales', 0)}")
            print(f"Top Region: {data.get('top_region', 'N/A')}")
            print(f"Median Sales: {data.get('median_sales', 0)}")
            print(f"Day Correlation: {data.get('day_sales_correlation', 0)}")
            print(f"Bar Chart: {'Generated' if data.get('bar_chart') else 'None'}")
            print(f"Cumulative Chart: {'Generated' if data.get('cumulative_sales_chart') else 'None'}")
            
            # Verify it's analyzing real data (not just defaults)
            if data.get('total_sales', 0) > 0:
                print("\n🎉 SUCCESS: API is processing real data!")
            else:
                print("\n⚠️  WARNING: API returned defaults - check data processing")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")

if __name__ == "__main__":
    test_api_with_real_sales_data()
