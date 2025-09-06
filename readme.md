# Piano Performance Analysis API

**6-Hour Hackathon Implementation**

This directory contains everything needed to build a production-ready API for analyzing piano performance across 19 perceptual dimensions using a pre-trained Audio Spectrogram Transformer.

## Quick Start

1. **Install Dependencies**:

```bash
cd crescendai-model
uv pip install -r requirements.txt
```

2. **Download Model** (if not already available):

```bash
# Your trained model checkpoint should be at:
# ./models/final_finetuned_model.pkl
```

3. **Run API**:

```bash
python3 main.py
```

4. **Test API**:

```bash
curl -X POST "http://localhost:8000/analyze-piano-performance" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@sample_piano.mp3" \
  -F "chunk_duration=3.0"
```

## 🎯 Hackathon Checklist

Follow the 6-hour implementation plan:

- [ ] **Hour 1**: Model verification & core pipeline (`core/model_loader.py`)
- [ ] **Hour 2-3**: FastAPI implementation (`main.py`, `api/endpoints.py`)
- [ ] **Hour 4**: Alert system & refinement (`api/analysis.py`)
- [ ] **Hour 5**: Cloud deployment (Hugging Face Spaces)
- [ ] **Hour 6**: Documentation & handoff

## 📁 Project Structure

```
hackathon/
├── README.md                 # This file
├── main.py                   # FastAPI application entry point
├── requirements.txt          # Dependencies
├── core/
│   ├── __init__.py
│   ├── model_loader.py      # Load pre-trained AST model
│   ├── audio_processing.py  # Audio chunking and preprocessing
│   └── config.py            # Configuration settings
├── api/
│   ├── __init__.py
│   ├── endpoints.py         # API route handlers
│   ├── analysis.py          # Analysis logic and alert system
│   └── models.py            # Pydantic request/response models
├── models/                  # Model checkpoint directory
│   └── .gitkeep
├── examples/
│   ├── sample_request.py    # Example API usage
│   └── test_audio/          # Sample audio files
│       └── .gitkeep
└── deployment/
    ├── Dockerfile           # Docker deployment
    ├── app.py              # Hugging Face Spaces app
    └── README.md           # Deployment instructions
```

## 🎹 19 Perceptual Dimensions

The API analyzes these musical qualities (scores 0-1):

1. **Timing_Stable_Unstable** - Rhythmic stability
2. **Articulation_Short_Long** - Note length style  
3. **Articulation_Soft_cushioned_Hard_solid** - Touch quality
4. **Pedal_Sparse/dry_Saturated/wet** - Pedal usage
5. **Pedal_Clean_Blurred** - Pedal precision
6. **Timbre_Even_Colorful** - Tonal variation
7. **Timbre_Shallow_Rich** - Tone depth
8. **Timbre_Bright_Dark** - Sound brightness
9. **Timbre_Soft_Loud** - Volume control
10. **Dynamic_Sophisticated/mellow_Raw/crude** - Dynamic refinement
11. **Dynamic_Little_range_Large_range** - Dynamic variation
12. **Music_Making_Fast_paced_Slow_paced** - Tempo/pacing
13. **Music_Making_Flat_Spacious** - Spatial quality
14. **Music_Making_Disproportioned_Balanced** - Balance
15. **Music_Making_Pure_Dramatic/expressive** - Expression
16. **Emotion_&_Mood_Optimistic/pleasant_Dark** - Emotional valence
17. **Emotion_&_Mood_Low_Energy_High_Energy** - Energy level
18. **Emotion_&_Mood_Honest_Imaginative** - Authenticity
19. **Interpretation_Unsatisfactory/doubtful_Convincing** - Quality

## 🚀 Model Details

- **Architecture**: 12-layer Audio Spectrogram Transformer
- **Parameters**: ~86M (pre-trained + regression head)
- **Input**: MP3/WAV files, max 30 seconds
- **Processing**: 3-second chunks with 1-second overlap
- **Output**: 19 regression scores + temporal analysis

## 📊 API Response Format

```json
{
  "overall_scores": {
    "timing_stable_unstable": 0.72,
    "articulation_short_long": 0.68,
    // ... all 19 dimensions
  },
  "temporal_analysis": [
    {
      "timestamp": "0:00-0:03",
      "timestamp_seconds": [0, 3],
      "scores": { /* 19 dimensions */ },
      "alerts": [ /* threshold alerts */ ]
    }
  ],
  "metadata": {
    "total_duration": 180.5,
    "total_chunks": 60,
    "chunk_duration": 3.0,
    "overlap": 1.0
  }
}
```
