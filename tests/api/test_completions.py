"""
Test Suite: Completions API (/v1/completions)
Based on Furiosa LLM OpenAI-Compatible Server Specification

Covers:
- Basic text completion
- Streaming responses
- Parameter validation
"""

import json

import pytest

from conftest import assert_valid_completion_response


class TestCompletionsBasic:
    """Basic Completions API Tests"""

    @pytest.mark.api
    @pytest.mark.smoke
    def test_completion_basic(self, api_client, sample_completion_request):
        """Test basic completion request"""
        session, base_url = api_client

        response = session.post(
            f"{base_url}/v1/completions", json=sample_completion_request
        )

        assert response.status_code == 200
        response_json = response.json()
        assert_valid_completion_response(response_json)

    @pytest.mark.api
    def test_completion_response_text(self, api_client, sample_completion_request):
        """Test that response contains generated text"""
        session, base_url = api_client

        response = session.post(
            f"{base_url}/v1/completions", json=sample_completion_request
        )

        response_json = response.json()
        text = response_json["choices"][0]["text"]

        assert len(text) > 0

    @pytest.mark.api
    def test_completion_finish_reason(self, api_client, sample_completion_request):
        """Test that finish_reason is properly set"""
        session, base_url = api_client

        response = session.post(
            f"{base_url}/v1/completions", json=sample_completion_request
        )

        response_json = response.json()
        finish_reason = response_json["choices"][0]["finish_reason"]

        assert finish_reason in ["stop", "length"]

    @pytest.mark.api
    def test_completion_usage_tokens(self, api_client, sample_completion_request):
        """Test that usage tokens are properly counted"""
        session, base_url = api_client

        response = session.post(
            f"{base_url}/v1/completions", json=sample_completion_request
        )

        response_json = response.json()
        usage = response_json["usage"]

        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage


class TestCompletionsParameters:
    """Test Completions API Parameters"""

    @pytest.mark.api
    @pytest.mark.parametrize("max_tokens", [16, 50, 100, 256])
    def test_completion_max_tokens(
        self, api_client, sample_completion_request, max_tokens
    ):
        """Test different max_tokens values"""
        session, base_url = api_client

        sample_completion_request["max_tokens"] = max_tokens

        response = session.post(
            f"{base_url}/v1/completions", json=sample_completion_request
        )

        assert response.status_code == 200

    @pytest.mark.api
    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
    def test_completion_temperature(
        self, api_client, sample_completion_request, temperature
    ):
        """Test different temperature values"""
        session, base_url = api_client

        sample_completion_request["temperature"] = temperature

        response = session.post(
            f"{base_url}/v1/completions", json=sample_completion_request
        )

        assert response.status_code == 200

    @pytest.mark.api
    def test_completion_min_tokens(self, api_client, sample_completion_request):
        """Test min_tokens parameter"""
        session, base_url = api_client

        sample_completion_request["min_tokens"] = 5

        response = session.post(
            f"{base_url}/v1/completions", json=sample_completion_request
        )

        assert response.status_code == 200


class TestCompletionsStreaming:
    """Test Completions Streaming"""

    @pytest.mark.api
    def test_completion_stream(self, api_client, sample_completion_request):
        """Test streaming completion"""
        session, base_url = api_client

        sample_completion_request["stream"] = True

        response = session.post(
            f"{base_url}/v1/completions", json=sample_completion_request, stream=True
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
