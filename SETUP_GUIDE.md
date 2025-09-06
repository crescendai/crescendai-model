# CrescendAI Piano Analysis Server - Setup Guide

This guide walks you through setting up and running the CrescendAI Piano Performance Analysis API on any laptop or VM. The server analyzes piano performance across 19 perceptual dimensions using a fine-tuned Audio Spectrogram Transformer.

## 🎯 Quick Overview

**What this does:** Analyzes uploaded piano audio files and returns scores (0-1) for 19 perceptual dimensions like timing stability, articulation, pedal usage, etc.

**Tech stack:** FastAPI + JAX/Flax + Audio Spectrogram Transformer

**Input:** MP3/WAV audio files (max 30 seconds, 10MB)

**Output:** JSON with overall scores + temporal analysis by chunks

## 📋 System Requirements

### Minimum Requirements

- **OS:** Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **RAM:** 4GB minimum, 8GB+ recommended
- **Storage:** 2GB free space
- **CPU:** Multi-core processor (Intel i5/AMD equivalent or better)

### Recommended for Production

- **RAM:** 16GB+
- **Storage:** SSD with 5GB+ free space
- **CPU:** 8+ cores
- **GPU:** Optional but recommended for faster inference

## 🚀 Installation Guide

### Step 1: Install Python

**Option A: Using Python 3.9-3.11 (Recommended)**

```bash
# Check if Python is installed
python3 --version
# Should show Python 3.9.x, 3.10.x, or 3.11.x

# If not installed:
# macOS: brew install python@3.10
# Ubuntu: sudo apt update && sudo apt install python3.10 python3.10-venv python3.10-pip
# Windows: Download from python.org
```

**Option B: Using Conda/Miniconda**

```bash
# Create new environment
conda create -n crescendai python=3.10
conda activate crescendai
```

### Step 2: Install uv (Fast Python Package Manager)

```bash
# Install uv (recommended for faster installs)
curl -LsSf https://astral.sh/uv/install.sh | sh
# OR: pip install uv

# Verify installation
uv --version
```

### Step 3: Clone or Download the Project

**Option A: Git Clone**

```bash
git clone <your-repo-url>
cd crescendai-model
```

**Option B: Download ZIP**

1. Download the project as ZIP from your repository
2. Extract to a folder (e.g., `crescendai-model`)
3. Open terminal in that folder

### Step 4: Install System Dependencies

**macOS:**

```bash
brew install libsndfile ffmpeg
```

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install libsndfile1 ffmpeg python3-dev build-essential
```

**Windows:**

```bash
# Using chocolatey (recommended)
choco install ffmpeg

# OR download FFmpeg from https://ffmpeg.org/download.html
# Add to PATH
```

### Step 5: Install Python Dependencies

```bash
cd crescendai-model

# Create virtual environment (if not using conda)
python3 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies with uv (faster)
uv pip install -r requirements.txt

# OR with regular pip (slower)
pip install -r requirements.txt
```

### Step 6: Download/Place the Model

⚠️ **CRITICAL STEP** - The API won't work without the model file!

```bash
# The model should be placed at:
./models/final_finetuned_model.pkl

# Check if model exists
ls -la models/final_finetuned_model.pkl

# If the model file doesn't exist, you need to:
# 1. Copy it from your training environment
# 2. Download it from your model storage
# 3. Contact the model creator
```

**Model file requirements:**

- File: `final_finetuned_model.pkl`
- Size: ~350MB
- Format: Pickled JAX/Flax model with scaler

## 🏃‍♂️ Running the Server

### Basic Startup

```bash
# Make sure you're in the project directory
cd crescendai-model

# Activate virtual environment (if using one)
source venv/bin/activate  # macOS/Linux
# OR: venv\Scripts\activate  # Windows

# Start the server
python main.py
```

You should see:

```
🎹 Starting Piano Performance Analysis API...
Model path: ./models/final_finetuned_model.pkl
✅ Model loaded successfully!
🚀 API ready on version 1.0.0
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Advanced Configuration

```bash
# Custom settings
export MODEL_PATH="/path/to/your/model.pkl"
export PORT=8080
export HOST="127.0.0.1"  # localhost only
export RELOAD=true  # development mode

python main.py
```

### Using Docker (Alternative)

```bash
# Build Docker image
docker build -t crescendai-api .

# Run container
docker run -p 8000:8000 -v $(pwd)/models:/app/models crescendai-api
```

## 🧪 Testing the Installation

### Step 1: Check Server Health

```bash
# In a new terminal window
curl http://localhost:8000/api/v1/health

# Should return:
# {"status": "healthy", "model_loaded": true, ...}
```

### Step 2: Test API Documentation

Open in browser:

- **Interactive Docs:** <http://localhost:8000/docs>
- **ReDoc:** <http://localhost:8000/redoc>
- **API Info:** <http://localhost:8000>

### Step 3: Test Audio Analysis

```bash
# Test with sample audio (you need a piano audio file)
curl -X POST "http://localhost:8000/api/v1/analyze-piano-performance" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@/path/to/piano_sample.mp3" \
  -F "chunk_duration=3.0"

# Should return JSON with scores for all 19 dimensions
```

### Step 4: Test with Python Client

```python
import requests

# Test health
response = requests.get("http://localhost:8000/api/v1/health")
print(f"Health check: {response.json()}")

# Test analysis (replace with actual audio file)
with open("sample_piano.mp3", "rb") as audio_file:
    files = {"audio": audio_file}
    data = {"chunk_duration": 3.0}
    
    response = requests.post(
        "http://localhost:8000/api/v1/analyze-piano-performance",
        files=files,
        data=data
    )
    
    result = response.json()
    print(f"Analysis result: {result['overall_scores']}")
```

## 🔧 Configuration Options

### Environment Variables

```bash
# Core settings
MODEL_PATH="./models/final_finetuned_model.pkl"  # Path to model file
PORT=8000                                        # Server port
HOST="0.0.0.0"                                   # Server host (0.0.0.0 = all interfaces)
RELOAD=false                                     # Auto-reload on code changes

# Audio processing
MAX_DURATION=30                                  # Max audio length (seconds)
MAX_FILE_SIZE=10485760                          # Max file size (bytes)
DEFAULT_CHUNK_DURATION=3.0                      # Default analysis chunk size
```

### Performance Tuning

**For Low-Memory Environments (4GB RAM):**

```python
# Edit core/config.py
TARGET_SHAPE = (64, 64)  # Reduce from (128, 128)
DEFAULT_CHUNK_DURATION = 2.0  # Smaller chunks
```

**For High-Performance:**

```python
# Edit core/config.py
TARGET_SHAPE = (256, 256)  # Increase resolution
DEFAULT_CHUNK_DURATION = 5.0  # Larger chunks
```

## 📡 API Usage Guide

### Core Endpoints

| Endpoint | Method | Purpose |
|----------|--------|----------|
| `/` | GET | API information |
| `/api/v1/health` | GET | Health check |
| `/api/v1/analyze-piano-performance` | POST | Analyze audio |

### Request Format

```bash
POST /api/v1/analyze-piano-performance
Content-Type: multipart/form-data

# Form fields:
# audio: File (MP3/WAV, max 10MB, max 30 seconds)
# chunk_duration: Float (optional, default 3.0, range 1.0-10.0)
# overlap: Float (optional, default 1.0)
# thresholds: JSON object (optional, custom alert thresholds)
```

### Response Format

```json
{
  "overall_scores": {
    "timing_stable_unstable": 0.72,
    "articulation_short_long": 0.68,
    "pedal_sparse_dry_saturated_wet": 0.45,
    "...": "... (all 19 dimensions)"
  },
  "temporal_analysis": [
    {
      "timestamp": "0:00-0:03",
      "timestamp_seconds": [0, 3],
      "scores": {
        "timing_stable_unstable": 0.74,
        "...": "..."
      },
      "alerts": [
        {
          "dimension": "timing_stable_unstable",
          "score": 0.74,
          "threshold": 0.6,
          "message": "High timing instability detected"
        }
      ]
    }
  ],
  "metadata": {
    "total_duration": 18.5,
    "total_chunks": 6,
    "chunk_duration": 3.0,
    "overlap": 1.0,
    "sample_rate": 22050
  }
}
```

### Frontend Integration Examples

**JavaScript/React:**

```javascript
const analyzeAudio = async (audioFile) => {
  const formData = new FormData();
  formData.append('audio', audioFile);
  formData.append('chunk_duration', 3.0);
  
  const response = await fetch('http://localhost:8000/api/v1/analyze-piano-performance', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
};
```

**Python:**

```python
import requests

def analyze_piano_performance(audio_path, chunk_duration=3.0):
    with open(audio_path, 'rb') as f:
        files = {'audio': f}
        data = {'chunk_duration': chunk_duration}
        
        response = requests.post(
            'http://localhost:8000/api/v1/analyze-piano-performance',
            files=files,
            data=data
        )
        
        return response.json()
```

**cURL:**

```bash
curl -X POST "http://localhost:8000/api/v1/analyze-piano-performance" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@piano_recording.mp3" \
  -F "chunk_duration=3.0" \
  -F "thresholds={\"timing_stable_unstable\": 0.7}"
```

## 🚨 Troubleshooting

### Common Issues & Solutions

**1. "Model file not found"**

```bash
# Check model exists
ls -la models/final_finetuned_model.pkl

# If missing, copy from your training environment
cp /path/to/model.pkl models/final_finetuned_model.pkl

# Or set custom path
export MODEL_PATH="/absolute/path/to/model.pkl"
```

**2. "JAX installation failed"**

```bash
# For CPU-only (most common)
uv pip install "jax[cpu]" flax optax --upgrade

# For NVIDIA GPU (if available)
uv pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Test JAX
python3 -c "import jax; print(jax.devices())"
```

**3. "Audio processing errors"**

```bash
# Install system audio libraries
# macOS:
brew install libsndfile ffmpeg

# Ubuntu:
sudo apt-get install libsndfile1 ffmpeg

# Test librosa
python3 -c "import librosa; print('Audio processing OK')"
```

**4. "Memory errors during inference"**

```python
# Edit core/config.py - reduce memory usage
TARGET_SHAPE = (64, 64)  # Smaller spectrograms
DEFAULT_CHUNK_DURATION = 2.0  # Smaller chunks
```

**5. "Port already in use"**

```bash
# Find process using port 8000
lsof -ti:8000

# Kill process (be careful!)
kill -9 $(lsof -ti:8000)

# Or use different port
export PORT=8080
python main.py
```

**6. "CORS errors from web frontend"**

```python
# Edit main.py - update CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Debug Commands

```bash
# Check server is running
curl -i http://localhost:8000/api/v1/health

# Check logs
tail -f piano_api.log

# Test dependencies
python3 -c "import jax, flax, librosa, fastapi; print('All imports OK')"

# Check model loading
python3 -c "from core.model_loader import load_model_on_startup; load_model_on_startup('./models/final_finetuned_model.pkl')"

# Memory usage
ps aux | grep python
top -p $(pgrep -f "python main.py")
```

## 🌐 Production Deployment

### Cloud Deployment Options

**1. Hugging Face Spaces (Easiest)**

```bash
# 1. Create account at huggingface.co
# 2. Create new Space with Gradio/FastAPI
# 3. Upload all files including model
# 4. Set Python version to 3.10
# 5. Hardware: CPU Basic (or GPU if available)
```

**2. Google Cloud Run**

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/crescendai-api
gcloud run deploy --image gcr.io/PROJECT-ID/crescendai-api --platform managed
```

**3. AWS EC2**

```bash
# Launch Ubuntu 20.04 instance (t3.medium or larger)
# Follow installation steps above
# Configure security group to allow port 8000
# Use PM2 or systemd for process management
```

**4. Railway/Render**

```bash
# Connect GitHub repo
# These platforms auto-detect FastAPI apps
# Ensure model file is included or uploaded separately
```

### Process Management

**Using PM2:**

```bash
# Install PM2
npm install -g pm2

# Start app
pm2 start "python main.py" --name crescendai-api

# Monitor
pm2 status
pm2 logs crescendai-api

# Auto-restart on reboot
pm2 startup
pm2 save
```

**Using systemd (Linux):**

```bash
# Create service file
sudo nano /etc/systemd/system/crescendai-api.service

# Service content:
[Unit]
Description=CrescendAI Piano Analysis API
After=network.target

[Service]
User=your-user
WorkingDirectory=/path/to/crescendai-model
Environment=PATH=/path/to/venv/bin
ExecStart=/path/to/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable crescendai-api
sudo systemctl start crescendai-api
```

### Security & Performance

**Basic Security:**

```python
# Add to main.py for production
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Rate limiting
@limiter.limit("10/minute")
@app.post("/api/v1/analyze-piano-performance")
```

**Nginx Reverse Proxy:**

```nginx
# /etc/nginx/sites-available/crescendai-api
server {
    listen 80;
    server_name your-domain.com;
    
    client_max_body_size 10M;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📊 Monitoring & Logs

### Key Metrics to Monitor

- **Response Times:** < 5s for typical 10-second audio files
- **Memory Usage:** Should stay under 2GB during inference
- **Error Rates:** < 5% for valid audio files
- **Throughput:** Depends on audio length and hardware

### Log Analysis

```bash
# Monitor real-time logs
tail -f piano_api.log

# Find errors
grep -i "error\|exception" piano_api.log

# Request timing analysis
grep "📤" piano_api.log | grep -o "([0-9]\+\.[0-9]\+s)" | sort -n
```

### Health Monitoring Script

```bash
#!/bin/bash
# health_check.sh
STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/v1/health)

if [ $STATUS -eq 200 ]; then
    echo "$(date): API healthy"
else
    echo "$(date): API unhealthy (HTTP $STATUS)"
    # Restart service if needed
    # systemctl restart crescendai-api
fi
```

## 🎯 Next Steps

### For Development Team

1. **Set up development environment** following this guide
2. **Test all endpoints** with sample audio files
3. **Integrate with frontend** using provided examples
4. **Configure monitoring** for production deployment
5. **Document any customizations** or additional features

### For Production

1. **Choose deployment platform** (Hugging Face, AWS, etc.)
2. **Set up proper monitoring** and logging
3. **Configure rate limiting** and security measures
4. **Set up CI/CD** for automatic deployments
5. **Create backup strategy** for model files

## 📞 Support

If you encounter issues:

1. **Check this troubleshooting section** first
2. **Review logs** in `piano_api.log`
3. **Test each component** separately (Python imports, model loading, etc.)
4. **Check system resources** (RAM, disk space)
5. **Verify model file** integrity and permissions

---

**Happy analyzing! 🎹✨**

This server provides a powerful API for piano performance analysis. The 19 perceptual dimensions give detailed insights into playing style, technique, and musical expression.
