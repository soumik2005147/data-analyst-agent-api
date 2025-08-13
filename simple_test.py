#!/usr/bin/env python3
"""
Simple test of the Data Analyst Agent API with sales data
"""

import requests
import json
import time
import os

def test_api_simple():
    """Simple test of the API"""
    
    print("🧪 Testing Data Analyst Agent API with Sales Data")
    print("=" * 60)
    
    # Test data for the questions
    questions_content = """Analyze the provided sales data and answer the following questions:

1. How many sales orders are in the dataset?
2. Which region has the highest total sales?
3. What is the average sales amount across all orders?
4. What is the total sales for the East region?"""
    
    # Create a simple CSV data for testing
    sales_data = """order_id,date,region,sales
1,2024-01-01,East,100
2,2024-01-02,West,200
3,2024-01-03,East,150
4,2024-01-04,North,50
5,2024-01-05,South,120
6,2024-01-06,West,220
7,2024-01-07,East,130
8,2024-01-08,South,170"""
    
    # Write test files
    with open('temp_questions.txt', 'w') as f:
        f.write(questions_content)
    
    with open('temp_sales.csv', 'w') as f:
        f.write(sales_data)
    
    print("📋 Questions:")
    print(questions_content)
    print("\n📊 Sample Data (8 rows):")
    print("East: 3 orders, West: 2 orders, North: 1 order, South: 2 orders")
    print("Total sales by region - East: 380, West: 420, North: 50, South: 290")
    
    try:
        # Test basic API functionality first
        print("\n🔄 Testing API directly...")
        
        # Import and test the agent directly
        import sys
        sys.path.append('.')
        
        from app import DataAnalystAgent
        import pandas as pd
        
        # Create agent instance
        agent = DataAnalystAgent()
        
        # Set questions
        agent.questions = questions_content
        
        # Create DataFrame from the sales data
        import io
        df = pd.read_csv(io.StringIO(sales_data))
        agent.data_context['sales_data'] = df
        
        print(f"✅ Loaded data: {len(df)} rows, {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Regions: {df['region'].unique()}")
        print(f"   Total Sales: {df['sales'].sum()}")
        
        # Analyze questions
        question_analysis = agent.analyze_questions()
        print(f"\n🧠 Question Analysis:")
        print(f"   Found {len(question_analysis.get('question_list', []))} questions")
        
        # Execute analysis plan
        print(f"\n⚙️  Executing Analysis...")
        try:
            import asyncio
            results = asyncio.run(agent.execute_analysis_plan()) if hasattr(agent, 'execute_analysis_plan') else "Direct analysis not available"
        except Exception as e:
            results = f"Analysis failed: {e}"
            print(f"   Error: {e}")
        
        print(f"\n✅ Results:")
        if isinstance(results, list):
            for i, result in enumerate(results, 1):
                print(f"   Answer {i}: {result}")
        elif isinstance(results, dict):
            for key, value in results.items():
                print(f"   {key}: {value}")
        else:
            print(f"   {results}")
            
        # Manual analysis for verification
        print(f"\n🔍 Manual Verification:")
        print(f"   1. Number of orders: {len(df)}")
        print(f"   2. Region with highest total sales: {df.groupby('region')['sales'].sum().idxmax()}")
        print(f"   3. Average sales amount: {df['sales'].mean():.2f}")
        print(f"   4. East region total sales: {df[df['region'] == 'East']['sales'].sum()}")
        
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            os.remove('temp_questions.txt')
            os.remove('temp_sales.csv')
        except:
            pass

if __name__ == "__main__":
    test_api_simple()
