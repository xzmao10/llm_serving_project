
import asyncio
import httpx
import time
import statistics

URL = "http://localhost:8000/generate"

async def send_request(prompt):
    async with httpx.AsyncClient() as client:
        start = time.time()
        r = await client.post(URL, json={"prompt": prompt})
        latency = time.time() - start
        return latency

async def run_test(concurrency=10):
    tasks = []
    for _ in range(concurrency):
        tasks.append(send_request("Explain GPU batching in LLM inference."))

    results = await asyncio.gather(*tasks)

    print("P50:", statistics.median(results))
    print("P95:", sorted(results)[int(len(results)*0.95)])
    print("Avg:", sum(results)/len(results))

async def main():
    for c in (1, 5, 10, 20):
        await run_test(c)

asyncio.run(main())
