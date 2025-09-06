"""
Piano Performance Analysis API - FastAPI Application

Main entry point for the 6-hour hackathon implementation.
"""

import logging
import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.config import API_TITLE, API_VERSION, API_DESCRIPTION, MODEL_PATH
from core.model_loader import load_model_on_startup
from api.endpoints import router
from api.models import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('piano_api.log') if os.access('.', os.W_OK) else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    
    # Startup
    logger.info("🎹 Starting Piano Performance Analysis API...")
    logger.info(f"Model path: {MODEL_PATH}")
    
    # Load model on startup
    model_loaded = load_model_on_startup(MODEL_PATH)
    
    if model_loaded:
        logger.info("✅ Model loaded successfully!")
    else:
        logger.error("❌ Model loading failed - API will return errors")
        # Don't fail startup - let health endpoint report the issue
    
    logger.info(f"🚀 API ready on version {API_VERSION}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Piano Performance Analysis API...")

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware for web app integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["Piano Analysis"])

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        routes=app.routes,
    )
    
    # Add custom metadata
    openapi_schema["info"]["x-logo"] = {
        "url": "https://via.placeholder.com/120x120.png?text=🎹"
    }
    
    openapi_schema["info"]["contact"] = {
        "name": "Piano Transformer API",
        "url": "https://github.com/your-repo/piano-perception-transformer"
    }
    
    # Add server information
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Local development server"},
        {"url": "https://your-app.hf.space", "description": "Production server"}
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🎹 Piano Performance Analysis API",
        "version": API_VERSION,
        "description": "Analyze piano performance across 19 perceptual dimensions",
        "docs": "/docs",
        "health": "/api/v1/health",
        "analyze": "/api/v1/analyze-piano-performance"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception in {request.method} {request.url}: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred. Please try again or contact support."
        ).dict()
    )

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests"""
    start_time = request.state.start_time = time.time() if 'time' in globals() else 0
    
    # Log request
    logger.info(f"📥 {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = (time.time() - start_time) if 'time' in globals() else 0
    logger.info(f"📤 {request.method} {request.url.path} -> {response.status_code} ({process_time:.3f}s)")
    
    return response

# Import time for middleware
import time

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    logger.info(f"🎯 Starting server on {host}:{port}")
    
    # Run server
    try:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
            access_log=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        sys.exit(1)