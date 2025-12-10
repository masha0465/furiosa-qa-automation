"""
Furiosa LLM OpenAI-Compatible Mock Server
Based on Furiosa SDK 2025.3.1 API Specification
https://developer.furiosa.ai/latest/en/furiosa_llm/furiosa-llm-serve.html
"""

import time
import uuid
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json

app = FastAPI(
    title="Furiosa LLM Mock Server",
    description="Mock server for Furiosa LLM OpenAI-Compatible API testing",
    version="2025.3.1"
)


# ============================================================================
# Request/Response Models (Based on Furiosa API Spec)
# ============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    n: int = 1
    best_of: int = 1
    use_beam_search: bool = False
    length_penalty: float = 1.0
    early_stopping: bool = False
    min_tokens: int = 0


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int = 16
    n: int = 1
    best_of: int = 1
    use_beam_search: bool = False
    length_penalty: float = 1.0
    early_stopping: bool = False
    min_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "furiosa-ai"
    # Furiosa-LLM specific extensions
    artifact_id: Optional[str] = None
    max_prompt_len: Optional[int] = None
    max_context_len: Optional[int] = None
    runtime_config: Optional[Dict[str, Any]] = None


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class VersionInfo(BaseModel):
    furiosa_llm: str
    furiosa_compiler: str
    furiosa_runtime: str


# ============================================================================
# Mock Data
# ============================================================================

MOCK_MODELS = [
    ModelInfo(
        id="furiosa-ai/Llama-3.1-8B-Instruct-FP8",
        created=int(time.time()),
        artifact_id="llama-3.1-8b-instruct-fp8-v1",
        max_prompt_len=4096,
        max_context_len=8192,
        runtime_config={"bucket_size": 128, "tensor_parallel_size": 4}
    ),
    ModelInfo(
        id="furiosa-ai/DeepSeek-R1-Distill-Llama-8B",
        created=int(time.time()),
        artifact_id="deepseek-r1-distill-llama-8b-v1",
        max_prompt_len=4096,
        max_context_len=8192,
        runtime_config={"bucket_size": 128, "tensor_parallel_size": 4}
    )
]

VERSION_INFO = VersionInfo(
    furiosa_llm="0.1.0",
    furiosa_compiler="2025.3.1",
    furiosa_runtime="2025.3.1"
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    Chat Completion API - OpenAI Compatible
    https://platform.openai.com/docs/api-reference/chat
    """
    if request.stream:
        return StreamingResponse(
            generate_chat_stream(request),
            media_type="text/event-stream"
        )
    
    # Mock response generation
    response_content = generate_mock_response(request.messages[-1].content)
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_content),
                finish_reason="stop"
            )
        ],
        usage={
            "prompt_tokens": sum(len(m.content.split()) for m in request.messages),
            "completion_tokens": len(response_content.split()),
            "total_tokens": sum(len(m.content.split()) for m in request.messages) + len(response_content.split())
        }
    )


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    """
    Completions API - OpenAI Compatible
    https://platform.openai.com/docs/api-reference/completions
    """
    if request.stream:
        return StreamingResponse(
            generate_completion_stream(request),
            media_type="text/event-stream"
        )
    
    response_text = generate_mock_response(request.prompt)
    
    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            CompletionChoice(
                index=0,
                text=response_text,
                finish_reason="stop"
            )
        ],
        usage={
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(request.prompt.split()) + len(response_text.split())
        }
    )


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """
    List available models
    """
    return ModelsResponse(data=MOCK_MODELS)


@app.get("/v1/models/{model_id:path}", response_model=ModelInfo)
async def get_model(model_id: str):
    """
    Get specific model information
    """
    for model in MOCK_MODELS:
        if model.id == model_id:
            return model
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


@app.get("/version")
async def get_version():
    """
    Furiosa SDK version information
    """
    return VERSION_INFO


@app.get("/metrics")
async def get_metrics():
    """
    Prometheus-compatible metrics endpoint
    """
    metrics = """# HELP furiosa_llm_num_requests_running Number of requests running on RNGD
# TYPE furiosa_llm_num_requests_running gauge
furiosa_llm_num_requests_running{model_name="furiosa-ai/Llama-3.1-8B-Instruct-FP8"} 0

# HELP furiosa_llm_num_requests_waiting Number of requests waiting to be processed
# TYPE furiosa_llm_num_requests_waiting gauge
furiosa_llm_num_requests_waiting{model_name="furiosa-ai/Llama-3.1-8B-Instruct-FP8"} 0

# HELP furiosa_llm_request_received_total Number of received requests in total
# TYPE furiosa_llm_request_received_total counter
furiosa_llm_request_received_total{model_name="furiosa-ai/Llama-3.1-8B-Instruct-FP8"} 100

# HELP furiosa_llm_request_success_total Number of successfully processed requests
# TYPE furiosa_llm_request_success_total counter
furiosa_llm_request_success_total{model_name="furiosa-ai/Llama-3.1-8B-Instruct-FP8"} 98

# HELP furiosa_llm_prompt_tokens_total Total number of prefill tokens processed
# TYPE furiosa_llm_prompt_tokens_total counter
furiosa_llm_prompt_tokens_total{model_name="furiosa-ai/Llama-3.1-8B-Instruct-FP8"} 15000

# HELP furiosa_llm_generation_tokens_total Total number of generation tokens processed
# TYPE furiosa_llm_generation_tokens_total counter
furiosa_llm_generation_tokens_total{model_name="furiosa-ai/Llama-3.1-8B-Instruct-FP8"} 8500

# HELP furiosa_llm_kv_cache_usage_perc KV-cache usage percentage
# TYPE furiosa_llm_kv_cache_usage_perc gauge
furiosa_llm_kv_cache_usage_perc{model_name="furiosa-ai/Llama-3.1-8B-Instruct-FP8",device_index="0"} 0.45
"""
    return metrics


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}


# ============================================================================
# Helper Functions
# ============================================================================

def generate_mock_response(prompt: str) -> str:
    """Generate mock response based on prompt"""
    if "capital" in prompt.lower() and "france" in prompt.lower():
        return "The capital of France is Paris."
    elif "weather" in prompt.lower():
        return "I'm an AI and don't have access to real-time weather data."
    elif "hello" in prompt.lower() or "hi" in prompt.lower():
        return "Hello! How can I assist you today?"
    else:
        return f"This is a mock response from Furiosa LLM. Your prompt was: {prompt[:50]}..."


async def generate_chat_stream(request: ChatCompletionRequest):
    """Generate streaming response for chat completions"""
    response_content = generate_mock_response(request.messages[-1].content)
    words = response_content.split()
    
    for i, word in enumerate(words):
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": word + " " if i < len(words) - 1 else word},
                "finish_reason": None if i < len(words) - 1 else "stop"
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    
    yield "data: [DONE]\n\n"


async def generate_completion_stream(request: CompletionRequest):
    """Generate streaming response for completions"""
    response_text = generate_mock_response(request.prompt)
    words = response_text.split()
    
    for i, word in enumerate(words):
        chunk = {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "text": word + " " if i < len(words) - 1 else word,
                "finish_reason": None if i < len(words) - 1 else "stop"
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
