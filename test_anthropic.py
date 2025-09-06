#!/usr/bin/env python3
"""
Test script to isolate Anthropic client initialization issues
"""

import os
from dotenv import load_dotenv

print("🧪 Testing Anthropic client initialization...")

# Load environment variables
load_dotenv()

# Check if API key is available
api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key:
    print("❌ ANTHROPIC_API_KEY not found")
    exit(1)

print(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")

# Test importing anthropic
try:
    import anthropic
    print(f"✅ Anthropic imported successfully, version: {anthropic.__version__}")
except ImportError as e:
    print(f"❌ Failed to import anthropic: {e}")
    exit(1)

# Test basic client initialization
try:
    from anthropic import Anthropic
    
    # Try minimal initialization
    print("🔄 Attempting basic client initialization...")
    client = Anthropic(api_key=api_key)
    print("✅ Basic client initialization successful!")
    
    # Test a simple API call
    print("🔄 Testing API call...")
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=10,
        messages=[{"role": "user", "content": "Hello"}]
    )
    print("✅ API call successful!")
    print(f"📝 Response: {response.content[0].text}")
    
except Exception as e:
    print(f"❌ Client initialization or API call failed: {e}")
    print(f"📋 Error type: {type(e).__name__}")
    
    # Try to get more details about the error
    if hasattr(e, '__dict__'):
        print(f"📋 Error details: {e.__dict__}")

print("\n🏁 Test complete!")
