# Data Analyst Agent API

A comprehensive, generalized data analysis API that uses intelligent planning to analyze any type of data and answer questions. The API automatically detects question types, creates analysis plans, and executes them to provide accurate answers.

## 🤖 **Intelligent Analysis Features**

### **Automatic Question Understanding**
- **Question Type Detection**: Automatically identifies count, comparison, statistical, and visualization questions
- **Context Analysis**: Understands data requirements from question content
- **Multi-format Support**: Handles JSON arrays, JSON objects, and custom response formats

### **Dynamic Analysis Planning** 
- **Step-by-Step Planning**: Creates intelligent analysis plans based on questions
- **Multi-source Data**: Combines web scraping, file processing, and database queries
- **Adaptive Execution**: Adjusts analysis approach based on available data

### **Advanced Data Processing**
- **Web Scraping**: Extracts data from Wikipedia and other web sources
- **File Processing**: Handles CSV, JSON, Excel files automatically
- **Database Integration**: DuckDB support for large dataset analysis
- **Statistical Analysis**: Correlation, regression, aggregations, and more

### **Smart Visualizations**
- **Automatic Plot Generation**: Creates appropriate charts based on question context
- **Multiple Formats**: Scatterplots, histograms, bar charts, time series
- **Optimized Output**: Base64-encoded images under size limits
- **Regression Lines**: Automatic trend line fitting

## 🚀 **API Usage**

### **Endpoint**: `POST /api/`

The API accepts multipart form data with:
- **`questions.txt`** (required): Natural language questions and instructions
- **Additional files** (optional): CSV, JSON, Excel files for analysis

```bash
curl "https://your-api-url.com/api/" \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv" \
  -F "metadata.json=@metadata.json"
```

## 📝 **Question Examples**

### **1. Web Scraping + Analysis**
```
Scrape data from https://example.com/data-table

1. How many records have values above $1000?
2. Which category has the highest average?
3. Create a bar chart showing distribution by category.
```

### **2. File Data Analysis**
```
Analyze the uploaded CSV file:

1. What is the correlation between column A and column B?
2. Which row has the maximum value in column C?
3. Plot a scatterplot of A vs B with regression line.
```

### **3. Database Queries**
```
Query the court database:

1. Which court processed the most cases?
2. What's the average processing time?
3. Show trend analysis over time.
```

## 🧠 **Intelligent Features**

### **Automatic Data Source Detection**
- URLs → Web scraping
- File attachments → File processing  
- Database references → Query execution

### **Question Type Classification**
- **Count Questions**: "How many...", "Number of..."
- **Comparison Questions**: "Which is highest...", "Earliest..."
- **Statistical Questions**: "Correlation...", "Average..."
- **Visualization Questions**: "Plot...", "Chart...", "Graph..."

### **Smart Response Formatting**
- **JSON Arrays**: `[answer1, answer2, answer3]`
- **JSON Objects**: `{"question1": "answer1", "question2": "answer2"}`
- **Custom Formats**: Adapts to specified output requirements

## 🔧 **Technical Architecture**

### **Core Components**
1. **Question Parser**: Analyzes natural language questions
2. **Analysis Planner**: Creates step-by-step execution plans  
3. **Data Processor**: Handles multiple data sources
4. **Visualization Engine**: Creates charts and plots
5. **Results Formatter**: Structures output appropriately

### **Data Flow**
```
Questions.txt → Question Analysis → Plan Creation → 
Data Collection → Analysis Execution → Visualization → 
Results Formatting → JSON Response
```

## 📊 **Supported Data Sources**

- **Web Pages**: Wikipedia, tables, structured content
- **Files**: CSV, JSON, Excel, text files
- **Databases**: DuckDB, S3 data, large datasets
- **APIs**: RESTful endpoints (configurable)

## 🎯 **Response Examples**

### **Array Format**
```json
[1, "Titanic", 0.485782, "data:image/png;base64,iVBORw0..."]
```

### **Object Format**  
```json
{
  "How many movies grossed over $2bn before 2000?": 1,
  "Which is the earliest $1.5bn film?": "Titanic", 
  "Correlation between Rank and Peak?": 0.485782,
  "Scatterplot visualization": "data:image/png;base64,..."
}
```

## 🚀 **Deployment**

### **Local Development**
```bash
pip install -r requirements.txt
python app.py
# API available at http://localhost:8000
```

### **Docker**
```bash
docker build -t data-analyst-agent .
docker run -p 8000:8000 data-analyst-agent
```

### **Cloud Platforms**
- **Heroku**: `git push heroku main`
- **Vercel**: `vercel --prod` 
- **Railway**: `railway up`
- **Google Cloud Run**: Use provided `Dockerfile`

## 🧪 **Testing**

```bash
python test_deployment.py
```

Runs comprehensive tests including:
- Dependency verification
- API endpoint testing
- Multiple question types
- Response format validation

## 📈 **Performance**

- **Response Time**: < 3 minutes (monitored)
- **File Size Limits**: Images < 100KB
- **Concurrent Requests**: Scalable architecture
- **Error Handling**: Graceful fallbacks and retries

## 🔍 **Advanced Features**

### **LLM Integration** (Optional)
- OpenAI API support for complex question understanding
- Enhanced natural language processing
- Context-aware analysis suggestions

### **Extensible Architecture**
- Plugin system for custom data sources
- Configurable analysis modules  
- Custom visualization templates

### **Monitoring & Logging**
- Comprehensive request logging
- Performance monitoring
- Error tracking and alerts

## 📚 **API Documentation**

### **Endpoints**
- `GET /` - API information and usage
- `POST /api/` - Main analysis endpoint
- `GET /health` - Health check

### **Response Codes**
- `200` - Success
- `400` - Invalid request format  
- `500` - Server error
- `504` - Timeout (>3 minutes)

---

## 🎯 **Ready for Production**

This API is designed to handle the full range of data analysis questions with:
- ✅ **Automatic question understanding**
- ✅ **Intelligent analysis planning** 
- ✅ **Multi-source data processing**
- ✅ **Dynamic visualization generation**
- ✅ **Flexible response formatting**
- ✅ **Production-ready deployment**

Perfect for evaluation systems that require adaptable, intelligent data analysis capabilities!
