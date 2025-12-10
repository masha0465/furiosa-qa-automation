"""
Test Suite: Models API (/v1/models)
Based on Furiosa LLM OpenAI-Compatible Server Specification

Covers:
- List all models
- Get specific model
- Furiosa-specific model extensions (artifact_id, max_prompt_len, etc.)
"""

import pytest

from conftest import assert_valid_model_info


class TestModelsAPI:
    """Models API Tests"""

    @pytest.mark.api
    @pytest.mark.smoke
    def test_list_models(self, api_client):
        """Test listing all available models"""
        session, base_url = api_client
        
        response = session.get(f"{base_url}/v1/models")
        
        assert response.status_code == 200
        response_json = response.json()
        
        assert "object" in response_json
        assert response_json["object"] == "list"
        assert "data" in response_json
        assert len(response_json["data"]) > 0

    @pytest.mark.api
    def test_list_models_structure(self, api_client):
        """Test that each model has required fields"""
        session, base_url = api_client
        
        response = session.get(f"{base_url}/v1/models")
        response_json = response.json()
        
        for model in response_json["data"]:
            assert_valid_model_info(model)

    @pytest.mark.api
    def test_get_specific_model(self, api_client):
        """Test getting a specific model by ID"""
        session, base_url = api_client
        
        # First, get list of models
        list_response = session.get(f"{base_url}/v1/models")
        models = list_response.json()["data"]
        
        # Get first model by ID
        model_id = models[0]["id"]
        response = session.get(f"{base_url}/v1/models/{model_id}")
        
        assert response.status_code == 200
        model = response.json()
        assert model["id"] == model_id

    @pytest.mark.api
    def test_get_nonexistent_model(self, api_client):
        """Test getting a model that doesn't exist"""
        session, base_url = api_client
        
        response = session.get(f"{base_url}/v1/models/nonexistent-model")
        
        assert response.status_code == 404


class TestFuriosaModelExtensions:
    """Test Furiosa-specific Model Extensions"""

    @pytest.mark.api
    def test_model_artifact_id(self, api_client):
        """Test that models have artifact_id (Furiosa extension)"""
        session, base_url = api_client
        
        response = session.get(f"{base_url}/v1/models")
        models = response.json()["data"]
        
        for model in models:
            assert "artifact_id" in model
            assert model["artifact_id"] is not None

    @pytest.mark.api
    def test_model_max_prompt_len(self, api_client):
        """Test that models have max_prompt_len (Furiosa extension)"""
        session, base_url = api_client
        
        response = session.get(f"{base_url}/v1/models")
        models = response.json()["data"]
        
        for model in models:
            assert "max_prompt_len" in model
            assert isinstance(model["max_prompt_len"], int)
            assert model["max_prompt_len"] > 0

    @pytest.mark.api
    def test_model_max_context_len(self, api_client):
        """Test that models have max_context_len (Furiosa extension)"""
        session, base_url = api_client
        
        response = session.get(f"{base_url}/v1/models")
        models = response.json()["data"]
        
        for model in models:
            assert "max_context_len" in model
            assert isinstance(model["max_context_len"], int)
            assert model["max_context_len"] >= model["max_prompt_len"]

    @pytest.mark.api
    def test_model_runtime_config(self, api_client):
        """Test that models have runtime_config (Furiosa extension)"""
        session, base_url = api_client
        
        response = session.get(f"{base_url}/v1/models")
        models = response.json()["data"]
        
        for model in models:
            assert "runtime_config" in model
            if model["runtime_config"]:
                assert isinstance(model["runtime_config"], dict)
