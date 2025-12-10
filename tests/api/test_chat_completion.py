"""
Test Suite: Chat Completion API (/v1/chat/completions)
Based on Furiosa LLM OpenAI-Compatible Server Specification

Covers:
- Basic chat completion
- Streaming responses
- Parameter validation
- Response structure validation
"""

import pytest
import requests
import json

from conftest import assert_valid_chat_response


class TestChatCompletionBasic:
    """Basic Chat Completion API Tests"""

    @pytest.mark.api
    @pytest.mark.smoke
    def test_chat_completion_basic(self, api_client, sample_chat_request):
        """Test basic chat completion request"""
        session, base_url = api_client

        response = session.post(
            f"{base_url}/v1/chat/completions", json=sample_chat_request
        )

        assert response.status_code == 200
        response_json = response.json()
        assert_valid_chat_response(response_json)

    @pytest.mark.api
    def test_chat_completion_response_content(self, api_client, sample_chat_request):
        """Test that response contains meaningful content"""
        session, base_url = api_client

        response = session.post(
            f"{base_url}/v1/chat/completions", json=sample_chat_request
        )

        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]

        assert len(content) > 0
        assert "Paris" in content  # Expected answer for France capital

    @pytest.mark.api
    def test_chat_completion_finish_reason(self, api_client, sample_chat_request):
        """Test that finish_reason is properly set"""
        session, base_url = api_client

        response = session.post(
            f"{base_url}/v1/chat/completions", json=sample_chat_request
        )

        response_json = response.json()
        finish_reason = response_json["choices"][0]["finish_reason"]

        assert finish_reason in ["stop", "length", "tool_calls"]

    @pytest.mark.api
    def test_chat_completion_usage_tokens(self, api_client, sample_chat_request):
        """Test that usage tokens are properly counted"""
        session, base_url = api_client

        response = session.post(
            f"{base_url}/v1/chat/completions", json=sample_chat_request
        )

        response_json = response.json()
        usage = response_json["usage"]

        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )


class TestChatCompletionParameters:
    """Test Chat Completion API Parameters"""

    @pytest.mark.api
    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0, 2.0])
    def test_chat_completion_temperature(
        self, api_client, sample_chat_request, temperature
    ):
        """Test different temperature values"""
        session, base_url = api_client

        sample_chat_request["temperature"] = temperature

        response = session.post(
            f"{base_url}/v1/chat/completions", json=sample_chat_request
        )

        assert response.status_code == 200

    @pytest.mark.api
    @pytest.mark.parametrize("top_p", [0.1, 0.5, 0.9, 1.0])
    def test_chat_completion_top_p(self, api_client, sample_chat_request, top_p):
        """Test different top_p values"""
        session, base_url = api_client

        sample_chat_request["top_p"] = top_p

        response = session.post(
            f"{base_url}/v1/chat/completions", json=sample_chat_request
        )

        assert response.status_code == 200

    @pytest.mark.api
    @pytest.mark.parametrize("top_k", [-1, 10, 50, 100])
    def test_chat_completion_top_k(self, api_client, sample_chat_request, top_k):
        """Test different top_k values (Furiosa-specific parameter)"""
        session, base_url = api_client

        sample_chat_request["top_k"] = top_k

        response = session.post(
            f"{base_url}/v1/chat/completions", json=sample_chat_request
        )

        assert response.status_code == 200

    @pytest.mark.api
    def test_chat_completion_max_tokens(self, api_client, sample_chat_request):
        """Test max_tokens parameter"""
        session, base_url = api_client

        sample_chat_request["max_tokens"] = 50

        response = session.post(
            f"{base_url}/v1/chat/completions", json=sample_chat_request
        )

        assert response.status_code == 200

    @pytest.mark.api
    def test_chat_completion_max_completion_tokens(
        self, api_client, sample_chat_request
    ):
        """Test max_completion_tokens parameter (supersedes max_tokens)"""
        session, base_url = api_client

        sample_chat_request["max_completion_tokens"] = 100

        response = session.post(
            f"{base_url}/v1/chat/completions", json=sample_chat_request
        )

        assert response.status_code == 200


class TestChatCompletionStreaming:
    """Test Chat Completion Streaming"""

    @pytest.mark.api
    def test_chat_completion_stream(self, api_client, sample_chat_request):
        """Test streaming chat completion"""
        session, base_url = api_client

        sample_chat_request["stream"] = True

        response = session.post(
            f"{base_url}/v1/chat/completions", json=sample_chat_request, stream=True
        )

        assert response.status_code == 200

        chunks = []
        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data = line_str[6:]
                    if data != "[DONE]":
                        chunks.append(json.loads(data))

        assert len(chunks) > 0
        # Last chunk should have finish_reason
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


class TestChatCompletionMultiTurn:
    """Test Multi-turn Conversations"""

    @pytest.mark.api
    def test_chat_completion_multi_turn(self, api_client):
        """Test multi-turn conversation"""
        session, base_url = api_client

        request = {
            "model": "furiosa-ai/Llama-3.1-8B-Instruct-FP8",
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hello! How can I help you?"},
                {"role": "user", "content": "What is the capital of France?"},
            ],
        }

        response = session.post(f"{base_url}/v1/chat/completions", json=request)

        assert response.status_code == 200
        response_json = response.json()
        assert_valid_chat_response(response_json)

    @pytest.mark.api
    def test_chat_completion_system_message(self, api_client):
        """Test chat completion with system message"""
        session, base_url = api_client

        request = {
            "model": "furiosa-ai/Llama-3.1-8B-Instruct-FP8",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
        }

        response = session.post(f"{base_url}/v1/chat/completions", json=request)

        assert response.status_code == 200
