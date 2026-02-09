from openai import AsyncOpenAI
import time
import asyncio
import sys
from pathlib import Path

# Add project root directory to Python path
# Project root directory: /root/autodl-tmp/src
# Current file: /root/autodl-tmp/src/server/llm_client.py
project_root = Path(__file__).parent.parent  # One level up to src/
sys.path.insert(0, str(project_root))

from server.config import ACTIVE_PROVIDER, load_provider_config

# =============================================================================
# OpenAI API Initialization
# =============================================================================

def _create_client(provider_cfg):
    """Create AsyncOpenAI client based on provider configuration."""
    return AsyncOpenAI(
        api_key=provider_cfg['api_key'],
        base_url=provider_cfg['api_base'],
    )

# Default global client
client = _create_client(ACTIVE_PROVIDER)

async def get_llm_response(system_prompt, user_prompt, max_retries=3, model=None, provider_name=None, think_mode=None, **kwargs):
    """
    [Asynchronous] Send request to LLM and get response, including retry mechanism.

    Arguments:
        system_prompt (str): System prompt (System Role).
        user_prompt (str): User prompt (User Role).
        max_retries (int): Maximum number of retries.
        model (str, optional): Specified model name. If None, uses provider's default_model.
        provider_name (str, optional): Temporarily use specified provider; uses default provider if None.

    Returns:
        dict: Contains content, latency, prompt_tokens, completion_tokens, total_tokens.
    """
    # Determine provider configuration and client
    if provider_name:
        provider_cfg = load_provider_config(provider_name)
        active_client = _create_client(provider_cfg)
    else:
        provider_cfg = ACTIVE_PROVIDER
        active_client = client

    # Determine model name
    actual_model_name = model or provider_cfg.get('default_model') or "default-model"

    # Priority logic: CLI arguments (kwargs) > model configuration (provider_cfg)
    # If a value exists in kwargs and is not None, use kwargs; otherwise use provider_cfg
    temperature = kwargs.get('temperature')
    if temperature is None:
        temperature = provider_cfg.get('temperature', 1)

    max_tokens = provider_cfg.get('max_tokens', 500) # max_tokens is usually not overridden by CLI

    top_p = kwargs.get('top_p')
    if top_p is None:
        top_p = provider_cfg.get('top_p', 1.0)
        
    top_k = kwargs.get('top_k')
    if top_k is None:
        top_k = provider_cfg.get('top_k')

    min_p = kwargs.get('min_p')
    if min_p is None:
        min_p = provider_cfg.get('min_p')

    # Qwen3 thinking mode control: append /no_think only when no_think is explicitly specified
    # Instruct-2507 series defaults to no_think, no instruction needed.
    _m = actual_model_name.lower()
    if think_mode == "no_think" and "qwen3" in _m and "instruct" not in _m:
        if "/no_think" not in user_prompt:
            user_prompt = f"{user_prompt} /no_think"

    retry_count = 0
    # print(f"DEBUG: Connecting to {active_client.base_url} with model {actual_model_name}")
    while retry_count < max_retries:
        try:
            request_kwargs = {
                "model": actual_model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",  "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }

            # [New] Support reasoning_effort (for reasoning models like o1/gpt-5)
            reasoning_effort = provider_cfg.get('reasoning_effort')
            if reasoning_effort:
                request_kwargs['reasoning_effort'] = reasoning_effort
            
            # [New] vLLM specific parameters passed via extra_body
            extra_body = {}
            if top_k is not None:
                extra_body['top_k'] = top_k
            if min_p is not None:
                extra_body['min_p'] = min_p
            
            if extra_body:
                request_kwargs['extra_body'] = extra_body
                # print(f"DEBUG: Using extra sampling params: {extra_body}")

            start_time = time.time()
            response = await active_client.chat.completions.create(**request_kwargs)
            end_time = time.time()
            duration = end_time - start_time

            content = response.choices[0].message.content
            usage = response.usage

            result = {
                "content": content,
                "latency": round(duration, 4),
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0
            }

            return result

        except Exception as e:
            retry_count += 1
            print(f"⚠️ API call error (Attempt {retry_count}/{max_retries}): {str(e)}")
            await asyncio.sleep(2 * retry_count)

    return {
        "content": f"ERROR: Failed after {max_retries} attempts.",
        "latency": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }