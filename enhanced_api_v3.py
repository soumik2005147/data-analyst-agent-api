from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import os
from openai import OpenAI
import uvicorn
from PIL import Image
import requests
from io import BytesIO

# Initialize FastAPI app
app = FastAPI(title="Data Analyst Agent API", version="3.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client initialization
client = None

def initialize_openai():
    """Initialize OpenAI client"""
    global client
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            # Test the connection
            models = client.models.list()
            print(f"✅ OpenAI connected successfully ({len(models.data)} models available)")
            return True
        except Exception as e:
            print(f"❌ OpenAI initialization failed: {e}")
            return False
    else:
        print("⚠️  OPENAI_API_KEY not found in environment")
        return False

class EnhancedDataAnalyst:
    def __init__(self):
        self.client = client
        self.supported_formats = {
            'csv': self.process_csv,
            'txt': self.process_text,
            'json': self.process_json,
            'png': self.process_image,
            'jpg': self.process_image,
            'jpeg': self.process_image,
            'gif': self.process_image,
            'bmp': self.process_image,
            'xlsx': self.process_excel,
            'xls': self.process_excel
        }
    
    async def process_files(self, files: List[UploadFile]) -> Dict[str, Any]:
        """Process all uploaded files"""
        processed_data = {
            'questions': '',
            'datasets': [],
            'images': [],
            'text_files': [],
            'metadata': {}
        }
        
        for file in files:
            content = await file.read()
            file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
            
            # Handle questions.txt specifically
            if file.filename.lower() == 'questions.txt':
                processed_data['questions'] = content.decode('utf-8')
                continue
            
            # Process based on file type
            if file_ext in self.supported_formats:
                try:
                    result = await self.supported_formats[file_ext](content, file.filename)
                    if file_ext in ['csv', 'xlsx', 'xls', 'json']:
                        processed_data['datasets'].append(result)
                    elif file_ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
                        processed_data['images'].append(result)
                    else:
                        processed_data['text_files'].append(result)
                except Exception as e:
                    print(f"Error processing {file.filename}: {e}")
        
        return processed_data
    
    async def process_csv(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Process CSV files"""
        try:
            csv_string = content.decode('utf-8')
            df = pd.read_csv(io.StringIO(csv_string))
            
            return {
                'type': 'csv',
                'filename': filename,
                'data': df,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
            }
        except Exception as e:
            raise Exception(f"CSV processing error: {e}")
    
    async def process_excel(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Process Excel files"""
        try:
            df = pd.read_excel(io.BytesIO(content))
            return {
                'type': 'excel',
                'filename': filename,
                'data': df,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
            }
        except Exception as e:
            raise Exception(f"Excel processing error: {e}")
    
    async def process_json(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Process JSON files"""
        try:
            json_data = json.loads(content.decode('utf-8'))
            # Try to convert to DataFrame if it's tabular
            if isinstance(json_data, list) and len(json_data) > 0 and isinstance(json_data[0], dict):
                df = pd.DataFrame(json_data)
                return {
                    'type': 'json_tabular',
                    'filename': filename,
                    'data': df,
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'raw_json': json_data
                }
            else:
                return {
                    'type': 'json_raw',
                    'filename': filename,
                    'data': json_data,
                    'structure': type(json_data).__name__
                }
        except Exception as e:
            raise Exception(f"JSON processing error: {e}")
    
    async def process_text(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Process text files"""
        try:
            text = content.decode('utf-8')
            return {
                'type': 'text',
                'filename': filename,
                'content': text,
                'length': len(text),
                'lines': len(text.split('\n'))
            }
        except Exception as e:
            raise Exception(f"Text processing error: {e}")
    
    async def process_image(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Process image files"""
        try:
            image = Image.open(io.BytesIO(content))
            # Convert to base64 for storage/transmission
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return {
                'type': 'image',
                'filename': filename,
                'size': image.size,
                'mode': image.mode,
                'format': image.format,
                'base64': f"data:image/png;base64,{img_base64}"
            }
        except Exception as e:
            raise Exception(f"Image processing error: {e}")
    
    def create_visualizations(self, datasets: List[Dict[str, Any]]) -> List[str]:
        """Create visualizations from datasets"""
        visualizations = []
        
        for dataset in datasets:
            if 'data' in dataset and isinstance(dataset['data'], pd.DataFrame):
                df = dataset['data']
                
                try:
                    # Create multiple charts
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                    fig.suptitle(f"Analysis of {dataset['filename']}", fontsize=16)
                    
                    # Chart 1: Numeric columns distribution
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        df[numeric_cols[0]].hist(bins=20, ax=axes[0, 0])
                        axes[0, 0].set_title(f'Distribution of {numeric_cols[0]}')
                    
                    # Chart 2: Correlation heatmap (if multiple numeric columns)
                    if len(numeric_cols) > 1:
                        corr = df[numeric_cols].corr()
                        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[0, 1])
                        axes[0, 1].set_title('Correlation Matrix')
                    
                    # Chart 3: Categorical data (if exists)
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    if len(categorical_cols) > 0:
                        df[categorical_cols[0]].value_counts().head(10).plot(kind='bar', ax=axes[1, 0])
                        axes[1, 0].set_title(f'Top Values in {categorical_cols[0]}')
                        axes[1, 0].tick_params(axis='x', rotation=45)
                    
                    # Chart 4: Summary statistics
                    if len(numeric_cols) > 0:
                        df[numeric_cols].sum().plot(kind='bar', ax=axes[1, 1])
                        axes[1, 1].set_title('Sum of Numeric Columns')
                        axes[1, 1].tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    
                    # Convert to base64
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                    buffer.seek(0)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    plt.close()
                    
                    visualizations.append(f"data:image/png;base64,{img_base64}")
                    
                except Exception as e:
                    print(f"Visualization error for {dataset['filename']}: {e}")
        
        return visualizations
    
    async def analyze_with_ai(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data using AI"""
        if not self.client:
            return self.fallback_analysis(processed_data)
        
        try:
            # Prepare data summary for AI
            data_summary = self.prepare_data_summary(processed_data)
            
            # Create AI prompt
            prompt = f"""
            I am a data analyst AI. Please analyze the following data and answer the questions.
            
            QUESTIONS:
            {processed_data.get('questions', 'Provide general insights about this data')}
            
            DATA SUMMARY:
            {data_summary}
            
            Please provide:
            1. Key insights and findings
            2. Data quality assessment
            3. Recommended actions
            4. Statistical summary
            5. Patterns and trends identified
            
            Format your response as clear, actionable insights.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert data analyst. Provide clear, actionable insights based on the data provided."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000
            )
            
            return {
                'ai_analysis': response.choices[0].message.content,
                'analysis_type': 'ai_powered',
                'model_used': 'gpt-3.5-turbo'
            }
            
        except Exception as e:
            print(f"AI analysis failed: {e}")
            return self.fallback_analysis(processed_data)
    
    def prepare_data_summary(self, processed_data: Dict[str, Any]) -> str:
        """Prepare a summary of processed data for AI analysis"""
        summary_parts = []
        
        # Datasets summary
        if processed_data['datasets']:
            summary_parts.append("DATASETS:")
            for i, dataset in enumerate(processed_data['datasets']):
                if 'data' in dataset and isinstance(dataset['data'], pd.DataFrame):
                    df = dataset['data']
                    summary_parts.append(f"Dataset {i+1} ({dataset['filename']}):")
                    summary_parts.append(f"- Shape: {df.shape}")
                    summary_parts.append(f"- Columns: {', '.join(df.columns.tolist())}")
                    
                    # Add sample data (first few rows)
                    if not df.empty:
                        sample = df.head(3).to_string()
                        summary_parts.append(f"- Sample data:\n{sample}")
        
        # Images summary
        if processed_data['images']:
            summary_parts.append(f"\nIMAGES: {len(processed_data['images'])} image(s) provided")
        
        # Text files summary
        if processed_data['text_files']:
            summary_parts.append(f"\nTEXT FILES: {len(processed_data['text_files'])} text file(s) provided")
        
        return '\n'.join(summary_parts)
    
    def fallback_analysis(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when AI is not available"""
        insights = []
        
        for dataset in processed_data['datasets']:
            if 'data' in dataset and isinstance(dataset['data'], pd.DataFrame):
                df = dataset['data']
                
                insights.append(f"Dataset: {dataset['filename']}")
                insights.append(f"- Rows: {df.shape[0]}, Columns: {df.shape[1]}")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    insights.append(f"- Numeric columns: {len(numeric_cols)}")
                    insights.append(f"- Average values: {df[numeric_cols].mean().to_dict()}")
                
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    insights.append(f"- Categorical columns: {len(categorical_cols)}")
        
        return {
            'ai_analysis': '\n'.join(insights) + "\n\nNote: This is a basic statistical analysis. For deeper insights, please ensure OpenAI API is configured.",
            'analysis_type': 'statistical_fallback',
            'model_used': 'statistical_methods'
        }

# Global analyst instance
analyst = EnhancedDataAnalyst()

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    print("🚀 Starting Enhanced Data Analyst Agent API v3.0")
    print("=" * 60)
    
    if initialize_openai():
        print("🤖 AI-powered analysis ready!")
    else:
        print("⚠️  Running in fallback mode (statistical analysis only)")
    
    print("🌐 API ready at http://localhost:8006")
    print("📋 Supported formats: CSV, Excel, JSON, Images, Text files")
    print("📄 Always include questions.txt with your requests")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ai_status = "enabled" if client else "fallback"
    return {
        "status": "healthy",
        "version": "3.0.0",
        "ai_integration": ai_status,
        "supported_formats": list(analyst.supported_formats.keys()),
        "message": "Enhanced Data Analyst Agent is running"
    }

@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
    """
    Enhanced data analysis endpoint
    Accepts multiple files including questions.txt and returns comprehensive analysis
    
    Expected usage:
    curl "http://localhost:8005/api/" -F "questions.txt=@questions.txt" -F "data.csv=@data.csv" -F "image.png=@image.png"
    """
    try:
        print(f"📊 Received {len(files)} files for analysis")
        
        # Process all uploaded files
        processed_data = await analyst.process_files(files)
        
        # Validate questions.txt was provided
        if not processed_data['questions']:
            raise HTTPException(status_code=400, detail="questions.txt file is required")
        
        print(f"❓ Questions: {processed_data['questions'][:100]}...")
        print(f"📁 Datasets: {len(processed_data['datasets'])}")
        print(f"🖼️  Images: {len(processed_data['images'])}")
        
        # Perform AI analysis
        analysis_result = await analyst.analyze_with_ai(processed_data)
        
        # Create visualizations
        visualizations = analyst.create_visualizations(processed_data['datasets'])
        
        # Prepare comprehensive response
        response = {
            'success': True,
            'timestamp': pd.Timestamp.now().isoformat(),
            'files_processed': len(files),
            'questions': processed_data['questions'],
            'analysis': analysis_result['ai_analysis'],
            'analysis_type': analysis_result['analysis_type'],
            'datasets_summary': [
                {
                    'filename': ds['filename'],
                    'type': ds['type'],
                    'shape': ds.get('shape'),
                    'columns': ds.get('columns')
                } for ds in processed_data['datasets']
            ],
            'images_summary': [
                {
                    'filename': img['filename'],
                    'size': img['size'],
                    'format': img.get('format')
                } for img in processed_data['images']
            ],
            'visualizations': visualizations,
            'metadata': {
                'total_datasets': len(processed_data['datasets']),
                'total_images': len(processed_data['images']),
                'total_text_files': len(processed_data['text_files']),
                'ai_model': analysis_result.get('model_used', 'statistical_methods')
            }
        }
        
        print("✅ Analysis completed successfully")
        return response
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Data Analyst Agent API v3.0",
        "description": "AI-powered API for data sourcing, preparation, analysis, and visualization",
        "endpoints": {
            "analyze": "POST /api/ - Main analysis endpoint",
            "health": "GET /health - Health check",
            "docs": "GET /docs - API documentation"
        },
        "supported_formats": list(analyst.supported_formats.keys()),
        "requirements": "Always include questions.txt file with your analysis requests",
        "example_usage": "curl 'http://localhost:8005/api/' -F 'questions.txt=@questions.txt' -F 'data.csv=@data.csv'"
    }

if __name__ == "__main__":
    print("🤖 Enhanced Data Analyst Agent API v3.0")
    print("📋 Set OPENAI_API_KEY environment variable for AI-powered analysis")
    print("🌐 Starting server on http://localhost:8006")
    
    uvicorn.run(app, host="127.0.0.1", port=8006, log_level="info")
