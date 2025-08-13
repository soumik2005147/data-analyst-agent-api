#!/usr/bin/env python3
"""
Enhanced Data Analyst Agent with OpenAI Assistant Integration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import uvicorn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import json
import os
from openai import OpenAI
import time

# Initialize FastAPI
app = FastAPI(
    title="AI-Powered Data Analyst Agent",
    description="Universal data analysis with OpenAI Assistant integration",
    version="2.0.0"
)

# Initialize OpenAI client (API key will be set via environment variable)
client = None

def initialize_openai():
    """Initialize OpenAI client with API key"""
    global client
    try:
        # Try to get API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  OPENAI_API_KEY not found in environment variables")
            print("   Set it with: set OPENAI_API_KEY=your_api_key_here")
            return False
        
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI: {e}")
        return False

class AIDataAnalyst:
    """AI-powered data analyst using OpenAI Assistant"""
    
    def __init__(self):
        self.assistant_id = None
        self.thread_id = None
    
    def create_assistant(self):
        """Create an OpenAI Assistant for data analysis"""
        if not client:
            return False
            
        try:
            assistant = client.beta.assistants.create(
                name="Data Analyst Expert",
                instructions="""You are an expert data analyst. Your job is to:
                1. Analyze CSV data and answer specific questions
                2. Return answers in EXACT format: [int, str, float, str]
                3. Provide real insights based on the actual data
                4. Never return placeholder values like "Titanic"
                5. Calculate real correlations, means, counts from the data
                6. Generate meaningful visualizations when requested
                
                Always return responses in this exact JSON format:
                [number, "text_answer", decimal_number, "visualization_or_text"]
                """,
                model="gpt-4-1106-preview",
                tools=[{"type": "code_interpreter"}]
            )
            self.assistant_id = assistant.id
            print(f"✅ Assistant created: {self.assistant_id}")
            return True
        except Exception as e:
            print(f"❌ Failed to create assistant: {e}")
            return False
    
    def create_thread(self):
        """Create a conversation thread"""
        if not client:
            return False
            
        try:
            thread = client.beta.threads.create()
            self.thread_id = thread.id
            print(f"✅ Thread created: {self.thread_id}")
            return True
        except Exception as e:
            print(f"❌ Failed to create thread: {e}")
            return False
    
    def analyze_with_ai(self, csv_data: str, questions: str) -> List:
        """Use OpenAI Assistant to analyze data and answer questions"""
        if not client or not self.assistant_id or not self.thread_id:
            print("❌ OpenAI not properly initialized")
            return self.fallback_analysis(csv_data, questions)
        
        try:
            # Create message with data and questions
            message_content = f"""
            Please analyze this CSV data and answer the questions:
            
            CSV DATA:
            {csv_data[:2000]}  # Limit data size
            
            QUESTIONS:
            {questions}
            
            Return your analysis in EXACT format: [int, str, float, str]
            Where:
            - First element: integer (count, number, etc.)
            - Second element: string (category, name, description)
            - Third element: float (correlation, percentage, average)
            - Fourth element: string (visualization description or data)
            
            Base your answers on the ACTUAL data provided, not examples.
            """
            
            # Add message to thread
            client.beta.threads.messages.create(
                thread_id=self.thread_id,
                role="user",
                content=message_content
            )
            
            # Run the assistant
            run = client.beta.threads.runs.create(
                thread_id=self.thread_id,
                assistant_id=self.assistant_id
            )
            
            # Wait for completion
            while run.status in ['queued', 'in_progress', 'cancelling']:
                time.sleep(1)
                run = client.beta.threads.runs.retrieve(
                    thread_id=self.thread_id,
                    run_id=run.id
                )
            
            if run.status == 'completed':
                # Get the response
                messages = client.beta.threads.messages.list(
                    thread_id=self.thread_id
                )
                
                response_content = messages.data[0].content[0].text.value
                print(f"🤖 AI Response: {response_content}")
                
                # Try to parse the response as JSON
                try:
                    # Extract JSON array from response
                    import re
                    json_match = re.search(r'\[.*\]', response_content)
                    if json_match:
                        result = json.loads(json_match.group())
                        if len(result) == 4:
                            return result
                except:
                    pass
                
            print(f"⚠️  AI analysis failed, using fallback")
            return self.fallback_analysis(csv_data, questions)
            
        except Exception as e:
            print(f"❌ AI analysis error: {e}")
            return self.fallback_analysis(csv_data, questions)
    
    def fallback_analysis(self, csv_data: str, questions: str) -> List:
        """Fallback to traditional pandas analysis if AI fails"""
        try:
            # Parse CSV data
            df = pd.read_csv(io.StringIO(csv_data))
            
            # Basic analysis
            row_count = len(df)
            
            # Get first column name or description
            first_col = df.columns[0] if len(df.columns) > 0 else "data"
            
            # Calculate correlation if possible
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlation = 0.0
            if len(numeric_cols) >= 2:
                correlation = df[numeric_cols[0]].corr(df[numeric_cols[1]])
                if pd.isna(correlation):
                    correlation = 0.0
            
            # Create simple visualization
            viz_data = self.create_simple_visualization(df)
            
            return [row_count, first_col, float(correlation), viz_data]
            
        except Exception as e:
            print(f"❌ Fallback analysis failed: {e}")
            return [1, "analysis_error", 0.0, "No visualization available"]
    
    def create_simple_visualization(self, df):
        """Create a simple chart"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Try to create a meaningful plot
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df[numeric_cols[0]].hist(bins=20)
                plt.title(f'Distribution of {numeric_cols[0]}')
                plt.xlabel(numeric_cols[0])
                plt.ylabel('Frequency')
            else:
                # Simple text plot if no numeric data
                plt.text(0.5, 0.5, 'Data Analysis\nCompleted', 
                        ha='center', va='center', fontsize=20)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            return "No visualization available"

# Global AI analyst instance
ai_analyst = AIDataAnalyst()

@app.on_event("startup")
async def startup_event():
    """Initialize AI components on startup"""
    print("🚀 Starting AI-Powered Data Analyst Agent")
    print("=" * 50)
    
    if initialize_openai():
        if ai_analyst.create_assistant():
            ai_analyst.create_thread()
            print("🤖 AI Assistant ready for data analysis!")
        else:
            print("⚠️  AI Assistant creation failed, will use fallback analysis")
    else:
        print("⚠️  OpenAI initialization failed, will use traditional analysis")
    
    print("🌐 Server ready at http://localhost:8005")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ai_status = "enabled" if client and ai_analyst.assistant_id else "fallback"
    return {
        "status": "healthy",
        "ai_integration": ai_status,
        "message": "AI-Powered Data Analyst Agent is running"
    }

@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
    """
    Enhanced data analysis with AI integration
    Accepts multiple files and returns [int, str, float, str]
    """
    try:
        csv_data = ""
        questions = ""
        
        # Process uploaded files
        for file in files:
            content = await file.read()
            
            if file.filename.endswith('.csv'):
                csv_data = content.decode('utf-8')
            elif file.filename.endswith('.txt'):
                questions = content.decode('utf-8')
        
        if not csv_data:
            raise HTTPException(status_code=400, detail="No CSV file provided")
        
        if not questions:
            questions = "Analyze this data and provide insights"
        
        print(f"📊 Analyzing CSV data ({len(csv_data)} chars)")
        print(f"❓ Questions: {questions[:100]}...")
        
        # Use AI-powered analysis
        result = ai_analyst.analyze_with_ai(csv_data, questions)
        
        print(f"✅ Analysis complete: {result}")
        return result
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI-Powered Data Analyst Agent",
        "version": "2.0.0",
        "ai_integration": "OpenAI Assistant",
        "endpoints": {
            "health": "/health",
            "analyze": "/api/",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    print("🤖 AI-Powered Data Analyst Agent v2.0")
    print("📋 Make sure to set OPENAI_API_KEY environment variable")
    print("🌐 Starting server on http://localhost:8005")
    
    uvicorn.run(app, host="127.0.0.1", port=8005, log_level="info")
