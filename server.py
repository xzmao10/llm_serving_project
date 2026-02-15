from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import torch
import time



app = FastAPI()

#model_name="facebook/opt-125m"  # small model for CPU
model_name="mistralai/Mistral-7B-Instruct-v0.2"  # large model for GPU

llm = LLM(
    dtype="float16",
    model=model_name
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 32


class GenerateResponse(BaseModel):
    text: str
    latency_s: float
    gpu_mem: float
    tokens_per_sec: float


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(req: GenerateRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    start = time.time()

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=req.max_tokens,
    )

    outputs = llm.generate(req.prompt, sampling_params)

    generated_text = outputs[0].outputs[0].text
    num_tokens = len(tokenizer.encode(generated_text))

    tokens_per_sec = num_tokens / (end - start)

    latency = time.time() - start
    gpu_mem = torch.cuda.memory_allocated() / 1e9

    return GenerateResponse(text=text, latency_s=latency, gpu_mem=gpu_mem, tokens_per_sec=tokens_per_sec )


