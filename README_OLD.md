# Data Analyst Agent API

A comprehensive FastAPI-based data analyst agent that uses LLMs to source, prepare, analyze, and visualize data. This API can scrape data from web sources, process uploaded files, perform statistical analysis, and generate visualizations.

## 🚀 Features

- **Web Data Scraping**: Scrape data from Wikipedia and other sources
- **File Processing**: Handle CSV, JSON, Excel files
- **Statistical Analysis**: Perform correlation analysis, regression, and more
- **Data Visualization**: Generate charts, scatterplots with regression lines
- **LLM Integration**: Optional OpenAI integration for advanced analysis
- **Database Support**: DuckDB integration for large dataset queries
- **Multiple Deployment Options**: Local, Docker, cloud platforms

## 🏗️ Project Structure

```
├── .github/
│   └── copilot-instructions.md
├── app.py                    # Main FastAPI application
├── test_api.py              # API testing script
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
├── vercel.json            # Vercel deployment config
├── start.sh               # Linux/Mac startup script
├── start.bat              # Windows startup script
├── .env.example           # Environment variables template
├── data/                  # Data storage directories
│   ├── raw/              # Raw data files
│   ├── processed/        # Cleaned and processed data
│   └── external/         # External data sources
├── src/                  # Source code modules
│   ├── data/            # Data processing utilities
│   ├── features/        # Feature engineering
│   ├── models/          # Machine learning models
│   └── visualization/   # Plotting utilities
├── tests/               # Unit tests
├── reports/             # Generated analysis reports
├── deployment/          # Deployment documentation
└── README.md           # This file
```

## 🚦 Quick Start

### Local Development

1. **Clone and setup**:
   ```bash
   # On Windows, run:
   start.bat
   
   # On Linux/Mac, run:
   chmod +x start.sh
   ./start.sh
   ```

2. **Manual setup**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your API keys (optional)
   
   # Start the server
   python app.py
   ```

3. **Access the API**:
   - API: `http://localhost:8000/api/`
   - Documentation: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/health`

### Docker Deployment

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Or build and run manually
docker build -t data-analyst-api .
docker run -p 8000:8000 data-analyst-api
```

## 📋 API Usage

### Endpoint: `POST /api/`

The API accepts multipart/form-data with:
- **`questions.txt`** (required): Text file containing analysis questions
- Additional files: CSV, JSON, Excel files for analysis

### Example Usage

```bash
# Basic usage with questions only
curl "http://localhost:8000/api/" -F "questions.txt=@questions.txt"

# With additional data files
curl "http://localhost:8000/api/" \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@dataset.csv" \
  -F "image.png=@chart.png"
```

### Sample Questions File

**Example 1: Wikipedia Film Analysis**
```text
Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI under 100,000 bytes.
```

**Example 2: Court Data Analysis**
```text
Analyze the Indian high court judgement dataset.

Answer the following questions and respond with a JSON object containing the answer.

{
  "Which high court disposed the most cases from 2019 - 2022?": "...",
  "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": "...",
  "Plot the year and # of days of delay as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": "data:image/webp:base64,..."
}
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file (copy from `.env.example`):

```env
# Optional - for LLM integration
OPENAI_API_KEY=your_openai_api_key_here

# API Configuration
PORT=8000
DEBUG=False

# AWS Configuration (for S3 access if needed)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
```

## 🚀 Deployment Options

### 1. Railway
```bash
railway login
railway link
railway up
```

### 2. Render
1. Connect GitHub repo
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### 3. Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/data-analyst-api
gcloud run deploy --image gcr.io/PROJECT-ID/data-analyst-api --platform managed
```

### 4. Heroku
```bash
heroku create your-data-analyst-api
git push heroku main
```

### 5. DigitalOcean App Platform
1. Connect repository
2. Configure as Web Service
3. Build: `pip install -r requirements.txt`
4. Run: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## 🧪 Testing

### Run Tests
```bash
# Test the API locally
python test_api.py

# Or use pytest for unit tests
pytest tests/
```

### Manual Testing
```bash
# Health check
curl http://localhost:8000/health

# Test with sample questions
echo "What is 2+2?" > test_questions.txt
curl "http://localhost:8000/api/" -F "questions.txt=@test_questions.txt"
```

## 📊 Supported Analysis Types

1. **Web Scraping**: Wikipedia, public APIs, web tables
2. **File Analysis**: CSV, JSON, Excel file processing
3. **Statistical Analysis**: Correlations, regressions, distributions
4. **Data Visualization**: Scatterplots, histograms, time series
5. **Large Dataset Queries**: DuckDB integration for big data
6. **Machine Learning**: Basic ML workflows with scikit-learn

## 📈 Performance

- **Response Time**: Under 3 minutes (API requirement)
- **File Size Limits**: Configurable, optimized for web deployment
- **Concurrent Requests**: Supports multiple simultaneous analyses
- **Memory Usage**: Optimized for cloud deployment constraints

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `pytest`
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the `/docs` endpoint for API documentation
2. Review the `deployment/README.md` for deployment help
3. Use the test script to validate your setup
4. Check logs for detailed error information
