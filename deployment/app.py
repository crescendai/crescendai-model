"""
Hugging Face Spaces Gradio App for Piano Performance Analysis

This provides a web interface wrapper around the FastAPI backend
for easy deployment on Hugging Face Spaces.
"""

import gradio as gr
import requests
import json
import tempfile
import os
import subprocess
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Start FastAPI server in background
API_PORT = 7861  # Different from Gradio port (7860)
API_URL = f"http://localhost:{API_PORT}/api/v1"

def start_fastapi_server():
    """Start FastAPI server in background thread"""
    try:
        # Set environment variables
        os.environ["PORT"] = str(API_PORT)
        os.environ["HOST"] = "0.0.0.0"
        
        # Start server
        subprocess.run([
            "python", "main.py"
        ], check=False)
    except Exception as e:
        print(f"Failed to start FastAPI server: {e}")

def wait_for_api():
    """Wait for API to be ready"""
    for _ in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

# Start API server in background
print("🚀 Starting FastAPI server...")
api_thread = threading.Thread(target=start_fastapi_server, daemon=True)
api_thread.start()

# Wait for API to be ready
print("⏳ Waiting for API to be ready...")
if not wait_for_api():
    print("❌ API failed to start")

def analyze_audio(audio_file, chunk_duration, overlap, custom_thresholds):
    """
    Analyze uploaded audio file
    
    Args:
        audio_file: Gradio audio file object
        chunk_duration: Duration of each chunk
        overlap: Overlap between chunks  
        custom_thresholds: JSON string with thresholds
        
    Returns:
        Analysis results formatted for display
    """
    try:
        if audio_file is None:
            return "❌ Please upload an audio file", None, None
        
        print(f"🎹 Processing audio file: {audio_file}")
        
        # Prepare request
        files = {'audio': open(audio_file, 'rb')}
        data = {
            'chunk_duration': chunk_duration,
            'overlap': overlap
        }
        
        # Add custom thresholds if provided
        if custom_thresholds and custom_thresholds.strip():
            try:
                json.loads(custom_thresholds)  # Validate JSON
                data['thresholds'] = custom_thresholds
            except json.JSONDecodeError:
                return "❌ Invalid JSON in custom thresholds", None, None
        
        # Make request to API
        response = requests.post(
            f"{API_URL}/analyze-piano-performance",
            files=files,
            data=data,
            timeout=60
        )
        
        files['audio'].close()
        
        if response.status_code != 200:
            error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
            return f"❌ Analysis failed: {error_data.get('error', 'Unknown error')}", None, None
        
        results = response.json()
        
        # Format results for display
        summary = format_results_summary(results)
        scores_plot = create_scores_plot(results)
        temporal_plot = create_temporal_plot(results)
        
        return summary, scores_plot, temporal_plot
        
    except Exception as e:
        print(f"Error in analyze_audio: {e}")
        return f"❌ Analysis failed: {str(e)}", None, None

def format_results_summary(results):
    """Format analysis results as readable text"""
    try:
        overall = results['overall_scores']
        metadata = results['metadata']
        
        # Calculate summary statistics
        scores_list = list(overall.values())
        avg_score = sum(scores_list) / len(scores_list)
        
        # Count alerts
        total_alerts = sum(len(chunk['alerts']) for chunk in results['temporal_analysis'])
        high_alerts = sum(1 for chunk in results['temporal_analysis'] 
                         for alert in chunk['alerts'] if alert['severity'] == 'high')
        
        summary = f"""
🎹 **PIANO PERFORMANCE ANALYSIS RESULTS**

📊 **Overall Performance**
• Average Score: {avg_score:.3f} / 1.000
• Duration Analyzed: {metadata['total_duration']:.1f} seconds
• Analysis Chunks: {metadata['total_chunks']}

⚠️ **Performance Alerts**
• Total Alerts: {total_alerts}
• High Priority: {high_alerts}

🎯 **Top Performing Dimensions**
"""
        
        # Find top 5 dimensions
        sorted_scores = sorted(overall.items(), key=lambda x: x[1], reverse=True)
        for i, (dim, score) in enumerate(sorted_scores[:5]):
            dim_name = dim.replace('_', ' ').title()
            summary += f"• {dim_name}: {score:.3f}\n"
        
        summary += f"\n🔍 **Areas for Improvement**\n"
        
        # Find bottom 5 dimensions
        for i, (dim, score) in enumerate(sorted_scores[-5:]):
            dim_name = dim.replace('_', ' ').title()
            summary += f"• {dim_name}: {score:.3f}\n"
        
        return summary
        
    except Exception as e:
        return f"❌ Error formatting results: {e}"

def create_scores_plot(results):
    """Create bar plot of overall scores"""
    try:
        overall = results['overall_scores']
        
        # Prepare data
        dimensions = []
        scores = []
        
        for dim, score in overall.items():
            # Clean up dimension names
            clean_name = dim.replace('_', ' ').replace('/', ' ').title()
            if len(clean_name) > 25:  # Truncate long names
                clean_name = clean_name[:22] + "..."
            dimensions.append(clean_name)
            scores.append(score)
        
        # Create plot
        plt.figure(figsize=(14, 10))
        bars = plt.barh(dimensions, scores)
        
        # Color code bars
        colors = []
        for score in scores:
            if score >= 0.7:
                colors.append('green')
            elif score >= 0.5:
                colors.append('orange') 
            else:
                colors.append('red')
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_alpha(0.7)
        
        plt.xlabel('Score')
        plt.title('Piano Performance Analysis - Overall Scores')
        plt.xlim(0, 1)
        plt.grid(axis='x', alpha=0.3)
        
        # Add score labels
        for i, score in enumerate(scores):
            plt.text(score + 0.01, i, f'{score:.3f}', va='center')
        
        plt.tight_layout()
        
        # Save to temporary file
        temp_path = tempfile.mktemp(suffix='.png')
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return temp_path
        
    except Exception as e:
        print(f"Error creating scores plot: {e}")
        return None

def create_temporal_plot(results):
    """Create temporal analysis plot"""
    try:
        temporal_data = results['temporal_analysis']
        
        if len(temporal_data) <= 1:
            return None  # Not enough data for temporal plot
        
        # Extract data
        timestamps = []
        avg_scores = []
        alert_counts = []
        
        for chunk in temporal_data:
            start_time = chunk['timestamp_seconds'][0]
            timestamps.append(start_time)
            
            # Calculate average score for this chunk
            chunk_scores = list(chunk['scores'].values())
            avg_scores.append(sum(chunk_scores) / len(chunk_scores))
            
            # Count alerts
            alert_counts.append(len(chunk['alerts']))
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Top plot: Average scores over time
        ax1.plot(timestamps, avg_scores, marker='o', linewidth=2, markersize=6)
        ax1.set_ylabel('Average Score')
        ax1.set_title('Performance Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Good (0.7)')
        ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Fair (0.5)')
        ax1.legend()
        
        # Bottom plot: Alert counts
        colors = ['red' if count > 2 else 'orange' if count > 0 else 'green' for count in alert_counts]
        ax2.bar(timestamps, alert_counts, alpha=0.7, color=colors, width=1.0)
        ax2.set_ylabel('Alert Count')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_title('Performance Alerts Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to temporary file
        temp_path = tempfile.mktemp(suffix='.png')
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return temp_path
        
    except Exception as e:
        print(f"Error creating temporal plot: {e}")
        return None

# Create Gradio interface
def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="🎹 Piano Performance Analyzer") as interface:
        
        gr.Markdown("""
        # 🎹 Piano Performance Analyzer
        
        **Analyze piano performance across 19 perceptual dimensions using a state-of-the-art Audio Spectrogram Transformer.**
        
        Upload a piano recording (MP3, WAV, FLAC, M4A) and get detailed analysis of timing, articulation, pedaling, dynamics, expression, and more.
        
        **Limits:** Max 30 seconds, 10MB file size
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                gr.Markdown("## 📁 Upload & Settings")
                
                audio_input = gr.Audio(
                    label="🎵 Piano Recording",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                
                with gr.Row():
                    chunk_duration = gr.Slider(
                        minimum=1.0, maximum=10.0, value=3.0, step=0.5,
                        label="Chunk Duration (seconds)",
                        info="Length of each analysis segment"
                    )
                    
                    overlap = gr.Slider(
                        minimum=0.0, maximum=5.0, value=1.0, step=0.1,
                        label="Overlap (seconds)",
                        info="Overlap between chunks"
                    )
                
                custom_thresholds = gr.Textbox(
                    label="Custom Thresholds (Optional)",
                    placeholder='{"timing_stable_unstable": 0.8, "articulation_short_long": 0.6}',
                    info="JSON format - set custom alert thresholds",
                    lines=3
                )
                
                analyze_btn = gr.Button("🔍 Analyze Performance", variant="primary", size="lg")
                
                gr.Markdown("""
                ### 📖 How to Use:
                1. Upload a piano recording (up to 30 seconds)
                2. Adjust chunk duration for analysis granularity
                3. Set overlap for smoother temporal analysis
                4. Optionally set custom alert thresholds
                5. Click "Analyze Performance"
                
                ### 🎯 Analysis Dimensions:
                **Timing & Rhythm:** Stability, pacing  
                **Articulation:** Note length, touch quality  
                **Pedaling:** Usage and clarity  
                **Timbre:** Color, richness, brightness  
                **Dynamics:** Range and sophistication  
                **Expression:** Balance, drama, emotion  
                **Overall:** Interpretation quality
                """)
            
            with gr.Column(scale=2):
                # Output area
                gr.Markdown("## 📊 Analysis Results")
                
                results_text = gr.Textbox(
                    label="Summary",
                    lines=15,
                    max_lines=20,
                    value="Upload an audio file and click 'Analyze Performance' to see results here."
                )
                
                with gr.Row():
                    scores_plot = gr.Image(
                        label="Overall Performance Scores",
                        type="filepath"
                    )
                    
                    temporal_plot = gr.Image(
                        label="Performance Over Time", 
                        type="filepath"
                    )
        
        # Connect the analyze button
        analyze_btn.click(
            fn=analyze_audio,
            inputs=[audio_input, chunk_duration, overlap, custom_thresholds],
            outputs=[results_text, scores_plot, temporal_plot]
        )
        
        gr.Markdown("""
        ---
        **🎹 Piano Performance Analyzer** | Built with Audio Spectrogram Transformer  
        *Analyze timing, articulation, pedaling, dynamics, expression and more*
        """)
    
    return interface

if __name__ == "__main__":
    # Create and launch interface
    interface = create_interface()
    
    # Launch with public sharing for Hugging Face Spaces
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Set to True for public sharing during development
    )