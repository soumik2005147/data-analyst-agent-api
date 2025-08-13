@echo off
REM Data Analyst Agent API Startup Script for Windows

echo 🚀 Starting Data Analyst Agent API...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not installed. Please install pip first.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo ⚙️  Creating .env file from template...
    copy .env.example .env
    echo 📝 Please edit .env file with your API keys before running the server.
)

REM Run the API
echo 🌐 Starting the API server...
echo 📍 Server will be available at: http://localhost:8000
echo 📖 API documentation at: http://localhost:8000/docs
echo 🏥 Health check at: http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
