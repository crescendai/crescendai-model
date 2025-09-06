# Deployment Guide - Piano Performance Analysis API

This directory contains deployment configurations for various platforms.

## 🚀 Quick Deployment Options

### Option 1: Hugging Face Spaces (Recommended for Hackathon)

**Pros:** Free, easy setup, good for demos  
**Cons:** Limited compute, may be slower

1. **Create Hugging Face Space:**
   ```bash
   # Go to https://huggingface.co/spaces
   # Create new Space with Python/Gradio
   # Upload files from this hackathon/ directory
   ```

2. **Upload required files:**
   - All Python files from hackathon/
   - `requirements.txt`
   - Your model checkpoint in `models/final_finetuned_model.pkl`
   - `app.py` (Gradio wrapper)

3. **Configure Space:**
   - Set Python version to 3.9+
   - Hardware: CPU Basic (or GPU if available)

### Option 2: Local Development

```bash
# Install dependencies
uv pip install -r requirements.txt

# Ensure model checkpoint exists
ls models/final_finetuned_model.pkl

# Run API server
python3 main.py
```

Access at: `http://localhost:8000`

### Option 3: Docker Deployment

```bash
# Build image
docker build -t piano-api .

# Run container
docker run -p 8000:8000 piano-api
```

## 📋 Pre-deployment Checklist

### Required Files
- [ ] `models/final_finetuned_model.pkl` - Your trained model
- [ ] All Python source files
- [ ] `requirements.txt` with correct versions
- [ ] API documentation accessible

### Model Verification
- [ ] Model loads without errors
- [ ] Test inference works on sample audio
- [ ] Output dimensions match expected (19)
- [ ] Label scaler included in checkpoint

### API Testing
- [ ] Health endpoint responds: `/api/v1/health`
- [ ] Analysis endpoint works: `/api/v1/analyze-piano-performance`
- [ ] Error handling works for invalid files
- [ ] Response format matches specification

## 🔧 Configuration

### Environment Variables

```bash
# Optional - API will use defaults if not set
export MODEL_PATH="./models/final_finetuned_model.pkl"
export PORT=8000
export HOST="0.0.0.0"
export RELOAD=false  # Set to true for development
```

### Memory Requirements

- **Minimum**: 4GB RAM
- **Recommended**: 8GB+ RAM
- **Model size**: ~350MB
- **Per request**: ~100MB peak

## 🐛 Troubleshooting

### Common Issues

**1. Model not found:**
```bash
# Check file exists
ls -la models/final_finetuned_model.pkl

# Check permissions
chmod 644 models/final_finetuned_model.pkl
```

**2. JAX/Flax installation issues:**
```bash
# For CPU-only deployment
uv pip install "jax[cpu]" flax optax --upgrade

# Check JAX is working
python3 -c "import jax; print(jax.devices())"
```

**3. Audio processing errors:**
```bash
# Install system audio libraries (Ubuntu/Debian)
sudo apt-get install libsndfile1 ffmpeg

# Or for macOS
brew install libsndfile ffmpeg
```

**4. Memory errors:**
```bash
# Reduce batch size in audio processing
# Edit core/audio_processing.py if needed
```

### Debug Commands

```bash
# Test health endpoint
curl http://localhost:8000/api/v1/health

# Test with sample file
curl -X POST "http://localhost:8000/api/v1/analyze-piano-performance" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@sample.mp3" \
  -F "chunk_duration=3.0"

# Check logs
tail -f piano_api.log
```

## 📊 Performance Optimization

### For Production

1. **Use GPU if available:**
   ```python
   # Install JAX with GPU support
   uv pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

2. **Optimize batch processing:**
   - Process multiple chunks together
   - Adjust chunk sizes based on available memory

3. **Add caching:**
   - Cache model in memory (already implemented)
   - Consider Redis for multi-instance deployments

4. **Scale horizontally:**
   - Use multiple API instances behind load balancer
   - Consider async processing for large files

## 🔐 Security Considerations

### For Production Deployment

1. **File upload limits:**
   - Already configured: 10MB max file size
   - 30 seconds max duration

2. **Rate limiting:**
   ```python
   # Add to main.py
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   
   @limiter.limit("10/minute")
   @app.post("/analyze-piano-performance")
   ```

3. **CORS configuration:**
   ```python
   # Update in main.py for production
   allow_origins=["https://yourdomain.com"]  # Not "*"
   ```

4. **API authentication:**
   ```python
   # Add API key validation if needed
   from fastapi import Depends, HTTPException, Header
   ```

## 📈 Monitoring & Logs

### Key Metrics to Monitor

- Request count and response times
- Memory usage during inference
- Error rates by endpoint
- File upload sizes and types

### Logging

```python
# API logs are written to:
# - Console (stdout)
# - piano_api.log (if writable)

# Important log events:
# - Model loading success/failure
# - Request processing times
# - Error details with stack traces
```

## 🎯 Team Integration

### For Your Hackathon Team

1. **Share API URL:**
   ```javascript
   // Frontend team can use:
   const API_URL = "https://your-app.hf.space/api/v1";
   
   // Example fetch request:
   const formData = new FormData();
   formData.append('audio', audioFile);
   formData.append('chunk_duration', 3.0);
   
   fetch(`${API_URL}/analyze-piano-performance`, {
     method: 'POST',
     body: formData
   });
   ```

2. **API Documentation:**
   - Interactive docs: `https://your-app.hf.space/docs`
   - ReDoc: `https://your-app.hf.space/redoc`

3. **Response Format:**
   - See `api/models.py` for complete TypeScript-compatible types
   - All scores are 0-1 normalized
   - Temporal analysis provides chunk-by-chunk breakdown

## ✅ Final Deployment Steps

1. **Upload model checkpoint**
2. **Test all endpoints work**  
3. **Share URL with team**
4. **Monitor initial requests**
5. **Document any issues for quick fixes**

Good luck with your hackathon! 🎹✨