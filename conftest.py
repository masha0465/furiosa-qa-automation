"""
Pytest Configuration and Fixtures for Furiosa QA Automation
"""

import os
import subprocess
import sys
import time

import pytest
import requests

# Add mock_server to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mock_server"))


# ============================================================================
# Configuration
# ============================================================================

BASE_URL = "http://localhost:8000"
STARTUP_TIMEOUT = 10
SHUTDOWN_TIMEOUT = 5


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def mock_server():
    """
    Start mock server for the entire test session.
    Automatically starts before tests and stops after all tests complete.
    """
    # Start the server
    server_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "mock_server.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(__file__),
    )

    # Wait for server to be ready
    start_time = time.time()
    while time.time() - start_time < STARTUP_TIMEOUT:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=1)
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)
    else:
        server_process.terminate()
        raise RuntimeError("Mock server failed to start")

    yield BASE_URL

    # Cleanup: Stop the server
    server_process.terminate()
    try:
        server_process.wait(timeout=SHUTDOWN_TIMEOUT)
    except subprocess.TimeoutExpired:
        server_process.kill()


@pytest.fixture
def base_url(mock_server):
    """Provide base URL for API tests"""
    return mock_server


@pytest.fixture
def api_client(base_url):
    """Provide configured requests session"""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session, base_url


@pytest.fixture
def sample_chat_request():
    """Sample chat completion request payload"""
    return {
        "model": "furiosa-ai/Llama-3.1-8B-Instruct-FP8",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    }


@pytest.fixture
def sample_completion_request():
    """Sample completion request payload"""
    return {
        "model": "furiosa-ai/Llama-3.1-8B-Instruct-FP8",
        "prompt": "The capital of France is",
    }


@pytest.fixture
def invalid_model_request():
    """Request with invalid model name"""
    return {
        "model": "invalid-model-name",
        "messages": [{"role": "user", "content": "Hello"}],
    }


# ============================================================================
# Helper Functions
# ============================================================================


def assert_valid_chat_response(response_json: dict):
    """Assert that response has valid chat completion structure"""
    assert "id" in response_json
    assert response_json["id"].startswith("chatcmpl-")
    assert "choices" in response_json
    assert len(response_json["choices"]) > 0
    assert "message" in response_json["choices"][0]
    assert "content" in response_json["choices"][0]["message"]
    assert "usage" in response_json


def assert_valid_completion_response(response_json: dict):
    """Assert that response has valid completion structure"""
    assert "id" in response_json
    assert response_json["id"].startswith("cmpl-")
    assert "choices" in response_json
    assert len(response_json["choices"]) > 0
    assert "text" in response_json["choices"][0]
    assert "usage" in response_json


def assert_valid_model_info(model_info: dict):
    """Assert that model info has required fields"""
    assert "id" in model_info
    assert "object" in model_info
    assert model_info["object"] == "model"
    # Furiosa-specific extensions
    assert "artifact_id" in model_info
    assert "max_prompt_len" in model_info
    assert "max_context_len" in model_info
