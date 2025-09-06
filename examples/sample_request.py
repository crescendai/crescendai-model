"""
Example usage of the Piano Performance Analysis API

This script demonstrates how to:
1. Send audio files to the API
2. Handle responses and errors
3. Parse analysis results
4. Work with alerts and thresholds
"""

import requests
import json
import time
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
# For production: API_BASE_URL = "https://your-app.hf.space/api/v1"

def check_api_health():
    """Check if the API is running and ready"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        health_data = response.json()
        
        print("🏥 API Health Check:")
        print(f"   Status: {health_data['status']}")
        print(f"   Model Loaded: {health_data['model_loaded']}")
        print(f"   Version: {health_data['version']}")
        
        return health_data['model_loaded']
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def analyze_piano_file(
    audio_file_path: str,
    chunk_duration: float = 3.0,
    overlap: float = 1.0,
    thresholds: dict = None
):
    """
    Analyze a piano audio file
    
    Args:
        audio_file_path: Path to audio file (MP3, WAV, FLAC, M4A)
        chunk_duration: Duration of each chunk in seconds (1.0-10.0)
        overlap: Overlap between chunks in seconds (0.0 to chunk_duration)
        thresholds: Optional custom thresholds for alerts
        
    Returns:
        Analysis results or None if failed
    """
    
    # Check if file exists
    if not Path(audio_file_path).exists():
        print(f"❌ File not found: {audio_file_path}")
        return None
    
    print(f"🎹 Analyzing: {Path(audio_file_path).name}")
    print(f"   Chunk duration: {chunk_duration}s")
    print(f"   Overlap: {overlap}s")
    
    try:
        # Prepare request data
        files = {'audio': open(audio_file_path, 'rb')}
        data = {
            'chunk_duration': chunk_duration,
            'overlap': overlap
        }
        
        # Add thresholds if provided
        if thresholds:
            data['thresholds'] = json.dumps(thresholds)
            print(f"   Custom thresholds: {len(thresholds)} dimensions")
        
        # Make request
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/analyze-piano-performance",
            files=files,
            data=data,
            timeout=60  # Allow up to 60 seconds for analysis
        )
        
        elapsed = time.time() - start_time
        
        # Close file
        files['audio'].close()
        
        # Handle response
        if response.status_code == 200:
            print(f"✅ Analysis completed in {elapsed:.2f}s")
            return response.json()
        else:
            error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
            print(f"❌ Analysis failed ({response.status_code})")
            print(f"   Error: {error_data.get('error', 'Unknown error')}")
            print(f"   Detail: {error_data.get('detail', 'No details')}")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out - try with shorter audio or larger chunks")
        return None
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def print_analysis_results(results: dict):
    """Pretty print analysis results"""
    if not results:
        return
    
    print("\n📊 ANALYSIS RESULTS")
    print("=" * 50)
    
    # Overall scores
    overall = results['overall_scores']
    print("\n🎯 Overall Performance Scores:")
    
    # Group dimensions for better readability
    dimension_groups = {
        "Timing & Rhythm": [
            ("timing_stable_unstable", "Timing Stability"),
            ("music_making_fast_paced_slow_paced", "Musical Pacing")
        ],
        "Articulation & Touch": [
            ("articulation_short_long", "Note Length"),
            ("articulation_soft_cushioned_hard_solid", "Touch Quality")
        ],
        "Pedaling": [
            ("pedal_sparse_dry_saturated_wet", "Pedal Usage"),
            ("pedal_clean_blurred", "Pedal Clarity")
        ],
        "Timbre & Tone": [
            ("timbre_even_colorful", "Tonal Variation"),
            ("timbre_shallow_rich", "Tone Richness"),
            ("timbre_bright_dark", "Brightness"),
            ("timbre_soft_loud", "Volume Control")
        ],
        "Dynamics": [
            ("dynamic_sophisticated_mellow_raw_crude", "Dynamic Refinement"),
            ("dynamic_little_range_large_range", "Dynamic Range")
        ],
        "Musical Expression": [
            ("music_making_flat_spacious", "Spatial Quality"),
            ("music_making_disproportioned_balanced", "Balance"),
            ("music_making_pure_dramatic_expressive", "Expression")
        ],
        "Emotion & Mood": [
            ("emotion_mood_optimistic_pleasant_dark", "Emotional Valence"),
            ("emotion_mood_low_energy_high_energy", "Energy Level"),
            ("emotion_mood_honest_imaginative", "Authenticity")
        ],
        "Overall Quality": [
            ("interpretation_unsatisfactory_convincing", "Interpretation")
        ]
    }
    
    for group_name, dimensions in dimension_groups.items():
        print(f"\n   {group_name}:")
        for dim_key, dim_name in dimensions:
            score = overall[dim_key]
            # Color code scores
            if score >= 0.7:
                status = "🟢"
            elif score >= 0.5:
                status = "🟡"
            else:
                status = "🔴"
            print(f"     {status} {dim_name}: {score:.3f}")
    
    # Metadata
    metadata = results['metadata']
    print(f"\n📈 Analysis Details:")
    print(f"   Duration: {metadata['total_duration']:.1f}s")
    print(f"   Chunks analyzed: {metadata['total_chunks']}")
    print(f"   Chunk size: {metadata['chunk_duration']}s")
    
    # Count alerts
    total_alerts = sum(len(chunk['alerts']) for chunk in results['temporal_analysis'])
    high_alerts = sum(1 for chunk in results['temporal_analysis'] 
                     for alert in chunk['alerts'] if alert['severity'] == 'high')
    
    print(f"   Total alerts: {total_alerts}")
    if high_alerts > 0:
        print(f"   🚨 High severity alerts: {high_alerts}")
    
    # Show temporal analysis summary if there are alerts
    if total_alerts > 0:
        print(f"\n⚠️ Alert Summary:")
        for i, chunk in enumerate(results['temporal_analysis']):
            if chunk['alerts']:
                print(f"   {chunk['timestamp']}:")
                for alert in chunk['alerts']:
                    severity_icon = "🚨" if alert['severity'] == 'high' else "⚠️"
                    print(f"     {severity_icon} {alert['dimension']}: {alert['score']:.3f} < {alert['threshold']:.3f}")

def example_basic_analysis():
    """Example 1: Basic analysis with default settings"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Analysis")
    print("="*60)
    
    # Replace with your actual audio file path
    audio_file = "./examples/test_audio/sample_piano.mp3"
    
    results = analyze_piano_file(audio_file)
    print_analysis_results(results)

def example_custom_chunks():
    """Example 2: Custom chunk settings for different analysis granularity"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Chunk Settings")
    print("="*60)
    
    audio_file = "./examples/test_audio/sample_piano.mp3"
    
    # Analyze with smaller chunks for more detailed temporal analysis
    results = analyze_piano_file(
        audio_file,
        chunk_duration=2.0,  # Smaller chunks
        overlap=0.5          # Less overlap
    )
    print_analysis_results(results)

def example_custom_thresholds():
    """Example 3: Custom thresholds for specific performance criteria"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Thresholds")
    print("="*60)
    
    audio_file = "./examples/test_audio/sample_piano.mp3"
    
    # Set higher standards for timing and interpretation
    custom_thresholds = {
        "timing_stable_unstable": 0.8,  # Very high standard for timing
        "interpretation_unsatisfactory_convincing": 0.7,  # High standard for interpretation
        "articulation_soft_cushioned_hard_solid": 0.6,  # Medium-high for touch
        "music_making_pure_dramatic_expressive": 0.65  # Good expression required
    }
    
    results = analyze_piano_file(
        audio_file, 
        thresholds=custom_thresholds
    )
    print_analysis_results(results)

def example_batch_analysis():
    """Example 4: Analyze multiple files"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Analysis")
    print("="*60)
    
    # List of audio files to analyze
    audio_files = [
        "./examples/test_audio/sample_piano.mp3",
        "./examples/test_audio/chopin_etude.wav",
        "./examples/test_audio/bach_invention.mp3"
    ]
    
    results_comparison = []
    
    for audio_file in audio_files:
        if Path(audio_file).exists():
            print(f"\n🎼 Analyzing: {Path(audio_file).name}")
            results = analyze_piano_file(audio_file)
            if results:
                results_comparison.append({
                    'filename': Path(audio_file).name,
                    'overall_score': sum(results['overall_scores'].values()) / 19,
                    'alerts': sum(len(chunk['alerts']) for chunk in results['temporal_analysis'])
                })
        else:
            print(f"⏭️ Skipping missing file: {Path(audio_file).name}")
    
    # Compare results
    if results_comparison:
        print(f"\n📊 BATCH ANALYSIS SUMMARY")
        print("-" * 40)
        for result in results_comparison:
            print(f"{result['filename']}: Avg Score {result['overall_score']:.3f}, Alerts: {result['alerts']}")

def main():
    """Main example runner"""
    print("🎹 Piano Performance Analysis API - Example Usage")
    print("=" * 60)
    
    # Check API health first
    if not check_api_health():
        print("\n❌ API is not ready. Please:")
        print("   1. Start the API server: python3 main.py")
        print("   2. Ensure the model checkpoint is available")
        print("   3. Check the API logs for errors")
        return
    
    print("\n✅ API is ready! Running examples...")
    
    # Run examples (comment out as needed)
    print("\n💡 Note: Replace file paths with your actual audio files")
    
    # example_basic_analysis()
    # example_custom_chunks() 
    # example_custom_thresholds()
    # example_batch_analysis()
    
    print("\n📚 To use with your own files:")
    print("   1. Place audio files in ./examples/test_audio/")
    print("   2. Update file paths in the examples above")
    print("   3. Uncomment the example functions you want to run")

if __name__ == "__main__":
    main()