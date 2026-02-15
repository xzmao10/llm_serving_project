LLM Serving Project

Purpose
-------
This repository provides a small FastAPI-based example for serving a causal language model (LLM) and simple experiment scripts for benchmarking and profiling. It's intentionally lightweight so you can:

- Run a CPU-friendly server locally for development and small tests.
- Run load tests (concurrency/latency measurements) against the server.
- Benchmark the model using a GPU (e.g., in Google Colab) to measure throughput and latency.

Quick start (local CPU)
-----------------------
1. Create and activate a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Start the server (use the small default model for quick testing)

```bash
MODEL_NAME=gpt2 python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

3. Smoke test (in another terminal)

```bash
curl -s -X POST http://localhost:8000/generate -H "Content-Type: application/json" \
  -d '{"prompt":"Hello, world"}' | python -m json.tool
```

4. Run the load test

```bash
python experiments/load_test.py
```

Notes
-----
- Setting `MODEL_NAME` lets you pick models for testing. `gpt2` is small and fast for development; `facebook/opt-125m` is slightly larger but still CPU-friendly. For real benchmarking on GPU use larger models in Colab.
- On first run the model will be downloaded; be patient.

Benchmarking on Google Colab (GPU)
---------------------------------
This section shows a minimal Colab workflow to benchmark model throughput and latency on a GPU. Make sure the Colab runtime is configured for a GPU (Runtime -> Change runtime type -> GPU).

Minimal Colab notebook cells
---------------------------
1) Install dependencies

```python
# Colab: install recent Transformers and accelerate (GPU-capable)
!pip install -q --upgrade pip
!pip install -q transformers accelerate safetensors
# If you want to use a specific torch build, install it accordingly; many Colab runtimes already include a GPU-enabled torch.
import torch
print('Torch version:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
```

2) (Optional) clone or upload this repo

```bash
# If your project is in a remote git repo, clone it; otherwise upload files via the Colab file browser.
# Example: git clone <your-repo-url>
# cd llm_serving_project
```

3) Colab benchmarking script (small model, explicit .to('cuda'))

```python
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'gpt2'  # switch to a larger model for tougher benchmarks

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device, torch.cuda.get_device_name(0) if device=='cuda' else '')

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to(device)
model.eval()

# Optionally enable fp16 for faster inference and lower memory (if model supports it)
if device == 'cuda':
    try:
        model.half()
    except Exception as e:
        print('Could not use half precision:', e)

# Helper function to generate and measure
@torch.inference_mode()
def generate_and_measure(prompt, max_new_tokens=32, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    # warm-up not counted in measurement
    start = time.time()
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
    torch.cuda.synchronize() if device=='cuda' else None
    latency = time.time() - start
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # crude token count
    token_count = len(outputs[0])
    return text, latency, token_count

# Warm up
prompt = 'Explain how GPU batching helps LLM inference.'
for _ in range(3):
    _ = generate_and_measure(prompt)

# Timed runs
iterations = 10
latencies = []
tokens = []
for _ in range(iterations):
    _, latency, token_count = generate_and_measure(prompt)
    latencies.append(latency)
    tokens.append(token_count)

print('P50 latency:', sorted(latencies)[int(0.5*len(latencies))])
print('P95 latency:', sorted(latencies)[int(0.95*len(latencies))])
print('Avg latency:', sum(latencies)/len(latencies))
print('Tokens per second (avg):', sum(tokens)/sum(latencies))
if device == 'cuda':
    import torch
    print('GPU memory (allocated GB):', torch.cuda.memory_allocated() / 1e9)
```

4) Larger model multi-GPU or memory-aware loading (use accelerate / device_map)

For larger models that won't fit on a single GPU, use `device_map='auto'` (requires `accelerate` integration in Transformers) or the `bitsandbytes` + `transformers` setup. Example pattern:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
# device_map='auto' will partition layers across devices when supported
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
```

Caveats and tips
----------------
- Model downloads: large models take time and disk space. Use small models for quick experiments.
- Mixed precision: `.half()` or `torch_dtype=torch.float16` speeds up inference and reduces memory usage on GPUs that support it.
- Synchronization: when measuring GPU time, call `torch.cuda.synchronize()` before checking time to avoid async kernel execution interfering with measurements.
- Reproducibility: set seeds (torch.manual_seed) if you need deterministic results (but `do_sample=True` will keep randomness).
- For accurate throughput benchmarking, run many iterations, discard warm-up, and report percentiles.

Adding a Colab notebook
----------------------
To simplify reproducible benchmarking, you can create a Colab notebook that contains the above cells, set the runtime to GPU, and then run the cells. If you'd like, I can produce a ready-to-run Colab `.ipynb` file that you can upload to Colab or a sharable Colab link.

Next steps I can do for you
--------------------------
- Create a small `health` endpoint and make the load test wait for readiness before sending load.
- Produce a ready-to-run Colab notebook file (.ipynb) that clones this repo and runs the benchmark end-to-end.
- Add the readiness-wait logic into `experiments/load_test.py` so it retries until the server responds.

Which of these would you like me to do next?
