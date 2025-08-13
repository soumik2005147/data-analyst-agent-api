# Render Deployment Guide for Data Analyst Agent API

## Quick Deploy to Render

### Option 1: One-Click Deploy (Recommended)
1. Push this repository to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure as follows:
   - **Name**: `data-analyst-agent-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python production_ready_api.py`

### Option 2: Using render.yaml (Infrastructure as Code)
1. The `render.yaml` file in this repository will automatically configure your deployment
2. Simply connect the repository to Render

## Environment Variables

### Required
- `OPENAI_API_KEY`: Your OpenAI API key for AI-powered analysis

### Automatic (Render provides)
- `PORT`: Automatically provided by Render

## Production Features
- ✅ 5-minute request timeout
- ✅ Support for 3 simultaneous requests  
- ✅ Comprehensive error handling
- ✅ Health check endpoint at `/health`
- ✅ CORS enabled for web access
- ✅ MIT licensed

## API Endpoints

### Health Check
```
GET /health
```

### Main Analysis Endpoint
```
POST /api/
Content-Type: multipart/form-data
Files: Upload CSV, Excel, JSON, images, or text files
```

### Root Endpoint
```
GET /
```

## Testing Your Deployment

Once deployed, test with:
```bash
curl https://your-app-name.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "4.0.0",
  "ai_available": true,
  "timestamp": "2025-08-12T..."
}
```

## File Upload Limits
- Max file size: 16MB per file
- Supported formats: CSV, Excel, JSON, PNG, JPG, TXT
- Multiple files supported in single request

## Billing Information
- This API can run on Render's **free tier**
- Free tier includes 750 hours/month
- Automatic scaling based on traffic

## Security Notes
- Set your `OPENAI_API_KEY` as an environment variable in Render dashboard
- Never commit API keys to your repository
- HTTPS enabled automatically on Render

## Support
Visit the Render documentation for additional deployment options and troubleshooting.
