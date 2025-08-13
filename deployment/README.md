# Deployment Scripts

This directory contains deployment scripts for different platforms.

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API locally
python app.py

# Or using uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Docker Deployment

```bash
# Build the Docker image
docker build -t data-analyst-api .

# Run with Docker
docker run -p 8000:8000 data-analyst-api

# Or use docker-compose
docker-compose up -d
```

## Cloud Deployment Options

### 1. Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway link
railway up
```

### 2. Render
1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Use these settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### 3. Heroku
```bash
# Install Heroku CLI and login
heroku login

# Create app
heroku create your-data-analyst-api

# Set environment variables
heroku config:set PORT=8000

# Deploy
git push heroku main
```

### 4. Google Cloud Run
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/data-analyst-api

# Deploy to Cloud Run
gcloud run deploy --image gcr.io/PROJECT-ID/data-analyst-api --platform managed
```

### 5. DigitalOcean App Platform
1. Connect your repository
2. Configure as a Web Service
3. Set build command: `pip install -r requirements.txt`
4. Set run command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### 6. AWS ECS/Fargate
See `aws-deployment.md` for detailed instructions.

### 7. Vercel (Serverless)
Add `vercel.json` configuration and deploy with Vercel CLI.

## Environment Variables

Make sure to set these environment variables in your deployment:
- `OPENAI_API_KEY` (optional, for LLM features)
- `PORT` (will be set automatically by most platforms)

## Testing Your Deployment

After deployment, test your API:

```bash
# Health check
curl https://your-api-url.com/health

# Test with sample data
curl "https://your-api-url.com/api/" -F "questions.txt=@test_questions.txt"
```
