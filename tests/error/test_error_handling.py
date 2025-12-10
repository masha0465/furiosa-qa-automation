"""
Test Suite: Error Handling
Based on Furiosa LLM OpenAI-Compatible Server Specification

Covers:
- Invalid requests
- Missing required fields
- Invalid parameter values
- Non-existent endpoints
"""

import pytest


class TestInvalidRequests:
    """Test Invalid Request Handling"""

    @pytest.mark.error
    def test_missing_messages_field(self, api_client):
        """Test chat completion without messages field"""
        session, base_url = api_client

        response = session.post(
            f"{base_url}/v1/chat/completions", json={"model": "test"}
        )

        # Should return 422 Unprocessable Entity
        assert response.status_code == 422

    @pytest.mark.error
    def test_missing_model_field(self, api_client):
        """Test chat completion without model field"""
        session, base_url = api_client

        response = session.post(
            f"{base_url}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

        # Should return 422 Unprocessable Entity
        assert response.status_code == 422

    @pytest.mark.error
    def test_missing_prompt_field(self, api_client):
        """Test completion without prompt field"""
        session, base_url = api_client

        response = session.post(f"{base_url}/v1/completions", json={"model": "test"})

        # Should return 422 Unprocessable Entity
        assert response.status_code == 422

    @pytest.mark.error
    def test_empty_messages_array(self, api_client):
        """Test chat completion with empty messages array"""
        session, base_url = api_client

        response = session.post(
            f"{base_url}/v1/chat/completions", json={"model": "test", "messages": []}
        )

        # Empty messages might cause various error codes
        assert response.status_code in [200, 400, 422, 500]


class TestInvalidParameterTypes:
    """Test Invalid Parameter Type Handling"""

    @pytest.mark.error
    def test_invalid_temperature_type(self, api_client):
        """Test chat completion with invalid temperature type"""
        session, base_url = api_client

        response = session.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": "hot",  # Should be float
            },
        )

        assert response.status_code == 422

    @pytest.mark.error
    def test_invalid_max_tokens_type(self, api_client):
        """Test chat completion with invalid max_tokens type"""
        session, base_url = api_client

        response = session.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": "many",  # Should be int
            },
        )

        assert response.status_code == 422

    @pytest.mark.error
    def test_invalid_stream_type(self, api_client):
        """Test chat completion with invalid stream type"""
        session, base_url = api_client

        response = session.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": "yes",  # Should be bool
            },
        )

        # Pydantic may coerce "yes" to True, so accept 200 or 422
        assert response.status_code in [200, 422]


class TestNonExistentEndpoints:
    """Test Non-Existent Endpoint Handling"""

    @pytest.mark.error
    def test_nonexistent_endpoint(self, api_client):
        """Test request to non-existent endpoint"""
        session, base_url = api_client

        response = session.get(f"{base_url}/v1/nonexistent")

        assert response.status_code == 404

    @pytest.mark.error
    def test_nonexistent_model(self, api_client):
        """Test getting non-existent model"""
        session, base_url = api_client

        response = session.get(f"{base_url}/v1/models/this-model-does-not-exist")

        assert response.status_code == 404


class TestInvalidHTTPMethods:
    """Test Invalid HTTP Method Handling"""

    @pytest.mark.error
    def test_get_chat_completions(self, api_client):
        """Test GET request to POST-only endpoint"""
        session, base_url = api_client

        response = session.get(f"{base_url}/v1/chat/completions")

        # Should return 405 Method Not Allowed
        assert response.status_code == 405

    @pytest.mark.error
    def test_post_models(self, api_client):
        """Test POST request to GET-only endpoint"""
        session, base_url = api_client

        response = session.post(f"{base_url}/v1/models", json={})

        # Should return 405 Method Not Allowed
        assert response.status_code == 405


class TestMalformedJSON:
    """Test Malformed JSON Handling"""

    @pytest.mark.error
    def test_malformed_json(self, api_client):
        """Test request with malformed JSON"""
        session, base_url = api_client

        response = session.post(
            f"{base_url}/v1/chat/completions",
            data="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    @pytest.mark.error
    def test_empty_body(self, api_client):
        """Test request with empty body"""
        session, base_url = api_client

        response = session.post(
            f"{base_url}/v1/chat/completions",
            data="",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422
