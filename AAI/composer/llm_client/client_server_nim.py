import asyncio
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse
from functools import partial
import httpx
import json
import os
import traceback

# Initialize FastAPI application
app = FastAPI()

###################################################################################
## Configuration and Defaults

DEFAULTS = {
    "TIMEOUT": 60,  # Request timeout in seconds
    "URLS": "nim-llm:8000,nim-embedding:8000,nim-ranking:8000,llm_client:9000",  # Backend URLs
    "FILTER_KEYWORDS": "",  # Filter keywords
    "LOG_DIR": "/logs",
}

def get_var(key: str) -> str:
    """Fetch a configuration variable, falling back to defaults."""
    return os.getenv(key, DEFAULTS.get(key, ""))

def get_urls() -> list:
    """Parse and normalize backend URLs."""
    return [url if "://" in url else f"http://{url.strip()}" for url in get_var("URLS").split(",")]

def get_filter_keywords() -> list:
    """Parse filter keywords for model filtering."""
    return [kw.strip().lower() for kw in get_var("FILTER_KEYWORDS").split(",")]

###################################################################################
## Utilities

async def fetch_models_from_server(server_url: str) -> dict:
    """Fetch available models from a single backend server."""
    try:
        async with httpx.AsyncClient(timeout=int(get_var("TIMEOUT"))) as client:
            try:
                response = await client.get(f"{server_url}/v1/models")
            except httpx.ConnectError:
                return {"error": f"Failed to fetch models from {server_url} with httpx.ConnectError. Is it running?"}
            if response.status_code == 200:
                return {"server": server_url, "models": response.json().get("data", [])}
            return {"error": f"Failed to fetch models from {server_url} with status {response.status_code}"}
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Exception during model discovery from {server_url}: {str(e)}"}

async def discover_models() -> tuple:
    """
    Discover models from all backend servers.
    Returns:
        - aggregated_models: List of models with metadata.
        - errors: List of errors encountered during discovery.
    """
    tasks = [fetch_models_from_server(server) for server in get_urls()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    aggregated_models, errors = [], []
    for server, result in zip(get_urls(), results):
        if isinstance(result, dict) and "models" in result:
            # Tag models with their originating server
            for model in result["models"]:
                model["server"] = server
                aggregated_models.append(model)
        elif isinstance(result, dict) and "error" in result:
            errors.append(result["error"])
        else:
            errors.append(f"Unexpected result from {server}: {result}")
    return aggregated_models, errors

def filter_models(models: list) -> list:
    """Filter models based on predefined keywords."""
    keywords = get_filter_keywords()
    server_counter = Counter([model["server"] for model in models])
    return [model for model in models if any(kw in model.get("id", "").lower() for kw in keywords) or server_counter[model["server"]] == 1]

###################################################################################
## Endpoints

@app.get("/v1/models")
async def list_models():
    """List all available models from all backend servers."""
    try:
        models, errors = await discover_models()
        filtered_models = filter_models(models)
        response = {"object": "list", "data": filtered_models}
        if errors:
            response["warnings"] = errors
        return JSONResponse(content=response)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Exception during model discovery: {str(e)}")

Client = partial(httpx.AsyncClient, timeout=httpx.Timeout(60))
savelog_lock = asyncio.Lock()

async def log_request_response(request_data, response_data):
    log_dir = get_var("LOG_DIR")
    if log_dir:
        log_file = os.path.join(log_dir, "interaction_log.json")
        async with savelog_lock:
            if not os.path.exists(log_file):
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump([], f)
            with open(log_file, "r+", encoding="utf-8") as f:
                f.seek(0)
                logs = json.load(f)
                logs.append({"request": request_data, "response": response_data})
                f.seek(0)
                f.truncate()
                json.dump(logs, f, indent=2)

async def create_completion_base(request: Request, extension: str):
    """Forwards chat completion requests to OpenAPI endpoint."""
    try:
        # Parse incoming request content
        content = await request.body()
        content = json.loads(content.decode())
        model = content.get("model")
        stream = content.get("stream", False)

        request_content_str = json.dumps(content)

        # Prepare headers for the outgoing request
        headers = {
            key: value for key, value in request.headers.items()
            if key.lower() not in ["host", "content-length"]
        }

        # Discover the target endpoint based on the requested model
        models, _ = await discover_models()
        model_entry = next((m for m in models if m["id"] == model), None)
        if not model_entry:
            possible_urls = get_urls()
            if possible_urls:
                model_entry = {"server": possible_urls[-1]}
            else: 
                raise HTTPException(status_code=404, detail=f"Model '{model}' not found.")
        target_server = model_entry["server"]

        # Prepare the request parameters
        call_kws = {
            "url": f"{target_server}/v1/{extension}",
            "headers": headers,
            "data": json.dumps(content).encode()
        }

        ############################################################
        # Handle Non-Streaming Use Case
        if not stream:
            try:
                async with Client() as client:
                    response = await client.post(**call_kws)
            except httpx.TimeoutException as e:
                raise HTTPException(status_code=408)

            filtered_headers = {
                key: value for key, value in response.headers.items()
                if key.lower() not in ["content-length", "content-encoding", "transfer-encoding"]
            }

            # Log input-output
            await log_request_response(request_content_str, response.text)

            return Response(
                content=response.content, 
                status_code=response.status_code, 
                headers=filtered_headers
            )

        ############################################################
        ## Streaming Use Case
        async def respond_and_stream():
            chunks = []
            try:
                async with Client().stream("POST", **call_kws) as response:
                    yield response
                    agen = response.aiter_bytes()
                    async for cbytes in agen:
                        chunks.append(cbytes)
                        yield cbytes
            except httpx.TimeoutException as e:
                raise HTTPException(status_code=408)
            finally:
                # Once streaming completes or an exception occurs, process and log the full output
                combined_dict = defaultdict(str)
                response_str = [chunk.decode("utf-8", errors="replace") for chunk in chunks]
                await log_request_response(request_content_str, response_str)

        agen = respond_and_stream()
        response = await agen.__anext__()
        if response.status_code != 200:
            response_bytes = await response.aread()
            content = response_bytes
            filtered_headers = {
                key: value for key, value in response.headers.items()
                if key.lower() not in ["content-length", "content-encoding", "transfer-encoding"]
            }
            # Log the error output
            await log_request_response(request_content_str, content.decode("utf-8", errors="replace"))
            return Response(content=content, status_code=response.status_code, headers=filtered_headers)
        else:
            return StreamingResponse(agen, media_type='text/event-stream')

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body.")
    except HTTPException as e:
        raise e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the request: {str(e)}"
        )

@app.post("/v1/{path:path}")
async def handle_request(request: Request, path: str):
    """Forwards requests based on the path to the appropriate OpenAPI endpoint."""
    return await create_completion_base(request, extension=path)

###################################################################################
## Health Check

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

###################################################################################
## Application Lifecycle

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown events."""
    print(f"Configured backend servers: {get_urls()}")
    print(f"Model filter keywords: {get_filter_keywords()}")
    yield
    print("Application shutdown complete.")

app.router.lifespan_context = lifespan

###################################################################################
