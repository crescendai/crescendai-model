"""
LLM service for generating structured pedagogical feedback from piano performance analysis.
Integrates with Anthropic API to provide expert-level piano instruction based on model output.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from pathlib import Path

import anthropic
from anthropic import Anthropic

from core.config import API_DIMENSION_NAMES
from .models import AnalysisResponse, DimensionScores, TemporalAnalysisItem

logger = logging.getLogger(__name__)

# Score categorization thresholds
LOW_THRESHOLD = 0.1
GOOD_THRESHOLD = 0.3

class LLMService:
    """Service for generating pedagogical feedback using Anthropic Claude"""
    
    def __init__(self):
        """Initialize Anthropic client"""
        self.client = None
        self.system_prompt = None
        self._initialize_client()
        self._load_system_prompt()
    
    def _initialize_client(self):
        """Initialize Anthropic client with API key from environment"""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not found in environment variables. LLM service will be unavailable.")
            self.client = None
            return
        
        try:
            # Initialize with minimal parameters to avoid compatibility issues
            self.client = Anthropic(
                api_key=api_key,
                # Remove any parameters that might cause issues
            )
            logger.info("Anthropic client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            logger.error(f"Anthropic version: {anthropic.__version__}")
            self.client = None
    
    def _load_system_prompt(self):
        """Load the piano feedback system prompt from file"""
        try:
            prompt_path = Path(__file__).parent.parent / "piano_feedback_system_prompt.md"
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
            logger.info("Piano feedback system prompt loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load system prompt: {e}")
            self.system_prompt = None
    
    def is_available(self) -> bool:
        """Check if LLM service is available and properly configured"""
        return self.client is not None and self.system_prompt is not None
    
    def categorize_score(self, score: float) -> str:
        """Categorize a score as low, moderate, or good"""
        if score <= LOW_THRESHOLD:
            return "low"
        elif score >= GOOD_THRESHOLD:
            return "good"
        else:
            return "moderate"
    
    def filter_model_output(self, analysis_response: AnalysisResponse) -> Dict[str, Any]:
        """
        Filter and categorize model output according to specified rules:
        - Ignore 0 values (model artifacts)
        - Categorize ≤0.1 as low, ≥0.3 as good, others as moderate
        """
        filtered_data = {
            "overall_scores": {},
            "temporal_analysis": [],
            "summary": {
                "total_dimensions": len(API_DIMENSION_NAMES),
                "filtered_dimensions": 0,
                "score_distribution": {"low": 0, "moderate": 0, "good": 0}
            }
        }
        
        # Process overall scores
        overall_dict = analysis_response.overall_scores.dict()
        for dimension, score in overall_dict.items():
            # Skip zero values (model artifacts)
            if score == 0.0:
                continue
                
            category = self.categorize_score(score)
            filtered_data["overall_scores"][dimension] = {
                "score": score,
                "category": category
            }
            filtered_data["summary"]["filtered_dimensions"] += 1
            filtered_data["summary"]["score_distribution"][category] += 1
        
        # Process temporal analysis
        for temporal_item in analysis_response.temporal_analysis:
            filtered_temporal = {
                "timestamp": temporal_item.timestamp,
                "timestamp_seconds": temporal_item.timestamp_seconds,
                "scores": {},
                "notable_moments": []
            }
            
            temporal_dict = temporal_item.scores.dict()
            for dimension, score in temporal_dict.items():
                # Skip zero values
                if score == 0.0:
                    continue
                
                category = self.categorize_score(score)
                filtered_temporal["scores"][dimension] = {
                    "score": score,
                    "category": category
                }
                
                # Mark particularly notable moments (very low or very high scores)
                if score <= 0.05 or score >= 0.9:
                    filtered_temporal["notable_moments"].append({
                        "dimension": dimension,
                        "score": score,
                        "category": category,
                        "note": "exceptional" if score >= 0.9 else "concerning"
                    })
            
            filtered_data["temporal_analysis"].append(filtered_temporal)
        
        # Add metadata
        filtered_data["metadata"] = {
            "total_duration": analysis_response.metadata.total_duration,
            "total_chunks": analysis_response.metadata.total_chunks,
            "chunk_duration": analysis_response.metadata.chunk_duration,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return filtered_data
    
    def generate_user_prompt(self, filtered_data: Dict[str, Any]) -> str:
        """Generate user prompt containing the filtered performance data"""
        prompt_parts = [
            "Please analyze this piano performance data and provide structured pedagogical feedback.",
            "",
            "# Performance Analysis Data",
            "",
            "## Overall Performance Scores",
            json.dumps(filtered_data["overall_scores"], indent=2),
            "",
            "## Temporal Analysis (Time-Segmented Performance)",
            json.dumps(filtered_data["temporal_analysis"], indent=2),
            "",
            "## Analysis Summary",
            json.dumps(filtered_data["summary"], indent=2),
            "",
            "## Performance Metadata",
            json.dumps(filtered_data["metadata"], indent=2),
            "",
            "Please provide your response in the structured JSON format specified in the system prompt, "
            "focusing on actionable feedback for an advanced piano student. Remember to:",
            "- Acknowledge strengths before suggesting improvements",
            "- Provide specific, practical advice",
            "- Use appropriate musical terminology",
            "- Focus on the most significant patterns in the data",
            "- Include time-specific feedback for notable moments"
        ]
        
        return "\n".join(prompt_parts)
    
    async def generate_feedback(self, analysis_response: AnalysisResponse) -> Optional[Dict[str, Any]]:
        """
        Generate structured pedagogical feedback from analysis response
        
        Args:
            analysis_response: Raw analysis response from the model
            
        Returns:
            Structured feedback dictionary or None if service unavailable
        """
        if not self.is_available():
            logger.warning("LLM service not available - skipping feedback generation")
            return None
        
        try:
            # Filter and process model output
            filtered_data = self.filter_model_output(analysis_response)
            
            # Check if we have meaningful data to analyze
            if filtered_data["summary"]["filtered_dimensions"] == 0:
                logger.warning("No meaningful scores to analyze (all zeros)")
                raise Exception("No meaningful performance data available for analysis - all scores are zero (model artifacts)")
            
            # Generate user prompt
            user_prompt = self.generate_user_prompt(filtered_data)
            
            logger.info(f"Generating feedback for performance with {filtered_data['summary']['filtered_dimensions']} dimensions")
            
            # Call Anthropic API
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Using current model
                max_tokens=4000,
                temperature=0.3,
                system=self.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            
            # Parse response
            feedback_text = response.content[0].text
            
            # Try to extract JSON from the response
            try:
                # Look for JSON block in the response
                if "```json" in feedback_text:
                    start = feedback_text.find("```json") + 7
                    end = feedback_text.find("```", start)
                    json_str = feedback_text[start:end].strip()
                else:
                    # Assume the entire response is JSON
                    json_str = feedback_text.strip()
                
                feedback_data = json.loads(json_str)
                
                # Add metadata about the analysis
                feedback_data["_metadata"] = {
                    "generated_at": datetime.now().isoformat(),
                    "model_used": "claude-3-5-sonnet-20241022",
                    "filtered_dimensions": filtered_data["summary"]["filtered_dimensions"],
                    "score_distribution": filtered_data["summary"]["score_distribution"]
                }
                
                logger.info("Successfully generated structured pedagogical feedback")
                return feedback_data
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response from Claude: {e}")
                logger.error(f"Raw response: {feedback_text[:500]}...")  # Log first 500 chars for debugging
                # Raise exception instead of returning fallback error dict
                raise Exception(f"Failed to parse JSON response from Claude: {e}")
        
        except anthropic.APITimeoutError as e:
            logger.error("Anthropic API timeout")
            raise Exception(f"LLM service timeout: {str(e)}")
        
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise Exception(f"LLM service error: {str(e)}")
        
        except Exception as e:
            logger.error(f"Unexpected error in feedback generation: {e}", exc_info=True)
            raise

# Global service instance
_llm_service = None

def get_llm_service() -> LLMService:
    """Get the global LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
