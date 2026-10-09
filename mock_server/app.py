import asyncio

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


app = FastAPI()


class GenerateRequest(BaseModel):
    prompt_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    priority: int = 0


async def generate_tokens(request: GenerateRequest):
    # Simulated prefill:
    # 0.1 ms per input token.
    prefill_seconds = request.prompt_tokens * 0.0001

    await asyncio.sleep(prefill_seconds)

    # Simulated decode:
    # 30 ms per generated token.
    decode_seconds = 0.030

    for token_id in range(request.output_tokens):
        await asyncio.sleep(decode_seconds)

        yield f"{token_id}\n"


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/generate")
async def generate(request: GenerateRequest):
    return StreamingResponse(
        generate_tokens(request),
        media_type="text/plain",
    )