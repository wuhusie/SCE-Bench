import sys
import os
import asyncio
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from server.config import OPENAI_API_BASE
from server.llm_client import get_llm_response, client
 
async def test_connection():
    print(f"📊 Current Configuration:")
    print(f"   - Backend: Local vLLM (Dedicated)")
    print(f"   - API Base: {OPENAI_API_BASE}")
    
    # 1. Detect Model
    print("🔍 Detecting running model...")
    try:
        models = await client.models.list()
        if not models.data:
             raise ValueError("No models found")
        current_model_name = models.data[0].id
        print(f"   - Model: {current_model_name}")
    except Exception as e:
        print(f"❌ Could not detect model: {e}")
        print("   Using fallback: default-model")
        current_model_name = "default-model"
    
    print("\n⏳ Sending test request...")
    try:
        result = await get_llm_response(
            system_prompt="You are a helpful AI assistant. Please answer concisely.",
            user_prompt="What is the capital of France?",
            max_retries=1,
            model=current_model_name  # Explicitly pass the detected model
        )
        print(f"\n✅ Response received:")
        print(f"Content: {result['content']}")
        print(f"Metrics: Latency={result['latency']}s, Tokens={result['total_tokens']}")
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print("Note: If using Local vLLM, ensure 'python src/server/launch_model.py 6' is running in another terminal.")

if __name__ == "__main__":
    asyncio.run(test_connection())
