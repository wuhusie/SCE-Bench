import os
import subprocess
import sys
import signal

# ================= Core Config =================
MODEL_REGISTRY = {
    # Group 1: Llama (Located in the Llama folder)
    "1": {
        "id": "llama-3.1-70b-bnb-4bit", 
        "path": "/root/autodl-fs/models/Llama/Meta-Llama-3.1-70B-bnb-4bit", 
        "name": "Meta-Llama-3.1-70B-bnb-4bit", 
        "len": 32768,  # Llama 3.1 standard context is 128k
        "dtype": "bfloat16", # 4bit quantization usually uses bf16 for computation during loading
        "chat_template": "templates/chat_template_llama3.jinja",
        "sampling_params": {
            "temperature": 1,
            "top_p": 0.9,
            "top_k": 50
        }
    },
    "2": {
        "id": "llama-3.1-70b-instruct-bnb-4bit", 
        "path": "/root/autodl-fs/models/Llama/Meta-Llama-3.1-70B-Instruct-bnb-4bit", 
        "name": "Meta-Llama-3.1-70B-Instruct-bnb-4bit", 
        "len": 32768, 
        "dtype": "bfloat16",
        "chat_template": "templates/chat_template_llama3.jinja",
        "sampling_params": {
            "temperature": 1,
            "top_p": 0.9,
            "top_k": 50
        }
    },
    
    # Group 2: Qwen (Located in the Qwen folder)
    "3": {
        "id": "qwen3-30b-fp8", 
        "path": "/root/autodl-fs/models/Qwen/Qwen3-30B-A3B-FP8", 
        "name": "Qwen3-30B-A3B-FP8", 
        "len": 32768,   # Assumed context length, depends on model config
        "dtype": "bfloat16",
        "sampling_params": {
            "temperature": 1,
            "top_p": 0.8,
            "top_k": 40,
            "min_p": 0.05
        }
    },
    "4": {
        "id": "qwen3-30b-instruct-fp8", 
        "path": "/root/autodl-fs/models/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", 
        "name": "Qwen3-30B-A3B-Instruct-2507-FP8", 
        "len": 32768, 
        "dtype": "bfloat16",
        "sampling_params": {
            "temperature": 1,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0
        }
    },
    "5": {
        "id": "qwen3-0.6b", 
        "path": "/root/autodl-fs/models/Qwen/Qwen3-0.6B", 
        "name": "Qwen3-0.6B", 
        "len": 32768, 
        "dtype": "auto",
        "sampling_params": {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.05
        }
    },
    "6": {
        "id": "llama-3.1-70b-instruct-gptq-int4", 
        "path": "/root/autodl-fs/models/Llama/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4", 
        "name": "Meta-Llama-3.1-70B-Instruct-GPTQ-INT4", 
        "len": 32768, 
        "dtype": "auto",
        "sampling_params": {
            "temperature": 1,
            "top_p": 0.9,
            "top_k": 50
        }
    }
}

# Alias for compatibility if MODELS is used elsewhere
MODELS = MODEL_REGISTRY

def run_server():
    # -------------------------------------------------------------------------
    # Model Selection Logic
    # -------------------------------------------------------------------------
    
    # 1. Attempt to get from command line arguments
    if len(sys.argv) > 1:
        target_key = sys.argv[1]
    
    # 2. If no command line arguments, enter interactive selection mode
    else:
        print("\n🤖 Select a model to launch:")
        print("=" * 60)
        
        # Sort and display all available models
        available_keys = sorted(MODELS.keys())
        for key in available_keys:
            info = MODELS[key]
            print(f"  [{key}] {info['name']}")
            print(f"      ID: {info['id']}")
            # print(f"      Path: {info['path']}") 
            print("-" * 60)
            
        print("=" * 60)

        try:
            # Get user input
            choice = input("\n👉 Enter number [Default: 4]: ").strip()
            
            if choice == "":
                target_key = "4"  # Default value
            else:
                target_key = choice
                
        except KeyboardInterrupt:
             print("\n\n❌ Operation cancelled by user")
             sys.exit(0)


    cfg = None
    if target_key in MODELS:
        cfg = MODELS[target_key]
    else:
        for v in MODELS.values():
            if v["id"] == target_key:
                cfg = v
                break
    
    if not cfg:
        print(f"❌ Error: Model {target_key} not found")
        return

    print(f"\n🚀 [Extreme Silent Mode] Launching: {cfg['name']}")
    print(f"🤫 Log filtering enabled: Only show startup progress and errors, hide HTTP flooding")
    print(f"⚖️  GPU Memory Utilization: 95% | Max Concurrent: 256")
    
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg["path"],
        "--served-model-name", cfg["name"],
        "--trust-remote-code",
        "--port", "8000",
        "--tensor-parallel-size", "1",
        "--max-model-len", str(cfg["len"]),
        "--gpu-memory-utilization", "0.95", 
        "--max-num-seqs", "256",
        "--max-num-batched-tokens", "8192", 

        "--disable-log-stats"
    ]

    # Add chat_template parameter (if configured)
    if "chat_template" in cfg:
        # Get absolute path based on the directory of the current script (src)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(base_dir, cfg["chat_template"])
        
        if os.path.exists(template_path):
            cmd.extend(["--chat-template", template_path])
        else:
             print(f"⚠️ Warning: Chat template file not found: {template_path}")

    if "Int4" not in cfg["path"]:
        cmd.extend(["--dtype", cfg["dtype"]])

    # === Core modification: Smart pipe filtering ===
    # Use Popen to take over the output stream
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, # Merge error stream into output for filtering
        text=True,
        bufsize=1 # Line buffering for real-time display
    )

    try:
        # Read output line by line
        for line in process.stdout:
            # 🔍 Filter: Discard annoying HTTP access logs
            if "POST /v1/chat/completions" in line and "200 OK" in line:
                continue
            
            # Print other important logs (e.g., startup progress, errors)
            print(line, end='')
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping service...")
        process.send_signal(signal.SIGINT)
        process.wait()
        print("✅ Service closed")

if __name__ == "__main__":
    run_server()