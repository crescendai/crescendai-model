#!/usr/bin/env python3
"""
Simple test script to verify environment variables are loaded correctly
"""

import os
from dotenv import load_dotenv

print("🧪 Testing environment variable loading...")

# Test loading .env file
print("📁 Loading .env file...")
load_result = load_dotenv()
print(f"✅ load_dotenv() returned: {load_result}")

# Check if API key is available
api_key = os.getenv('ANTHROPIC_API_KEY')
if api_key:
    print(f"✅ ANTHROPIC_API_KEY found: {api_key[:8]}...{api_key[-4:]}")
    print(f"📏 API key length: {len(api_key)} characters")
else:
    print("❌ ANTHROPIC_API_KEY not found")

# List all environment variables that contain 'ANTHROPIC'
print("\n🔍 Searching for ANTHROPIC variables...")
anthropic_vars = {k: v for k, v in os.environ.items() if 'ANTHROPIC' in k.upper()}
if anthropic_vars:
    for key, value in anthropic_vars.items():
        masked_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
        print(f"  {key}: {masked_value}")
else:
    print("  No ANTHROPIC variables found in environment")

# Check .env file existence and content
env_file_path = ".env"
if os.path.exists(env_file_path):
    print(f"\n📄 .env file exists at: {os.path.abspath(env_file_path)}")
    with open(env_file_path, 'r') as f:
        lines = f.readlines()
    print(f"📊 .env file has {len(lines)} lines")
    
    # Check if ANTHROPIC_API_KEY is in the file
    has_api_key = any('ANTHROPIC_API_KEY' in line for line in lines)
    print(f"🔑 ANTHROPIC_API_KEY in .env file: {has_api_key}")
else:
    print(f"\n❌ .env file not found at: {os.path.abspath(env_file_path)}")

print("\n🏁 Test complete!")
