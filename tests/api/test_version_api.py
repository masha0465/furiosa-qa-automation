"""
Test Suite: Version API (/version)
Based on Furiosa LLM OpenAI-Compatible Server Specification

Covers:
- Version endpoint availability
- SDK component versions
"""

import pytest


class TestVersionAPI:
    """Version API Tests"""

    @pytest.mark.api
    @pytest.mark.smoke
    def test_version_endpoint(self, api_client):
        """Test version endpoint returns 200"""
        session, base_url = api_client

        response = session.get(f"{base_url}/version")

        assert response.status_code == 200

    @pytest.mark.api
    def test_version_contains_furiosa_llm(self, api_client):
        """Test version contains furiosa_llm version"""
        session, base_url = api_client

        response = session.get(f"{base_url}/version")
        version_info = response.json()

        assert "furiosa_llm" in version_info
        assert version_info["furiosa_llm"] is not None

    @pytest.mark.api
    def test_version_contains_compiler(self, api_client):
        """Test version contains furiosa_compiler version"""
        session, base_url = api_client

        response = session.get(f"{base_url}/version")
        version_info = response.json()

        assert "furiosa_compiler" in version_info
        assert version_info["furiosa_compiler"] is not None

    @pytest.mark.api
    def test_version_contains_runtime(self, api_client):
        """Test version contains furiosa_runtime version"""
        session, base_url = api_client

        response = session.get(f"{base_url}/version")
        version_info = response.json()

        assert "furiosa_runtime" in version_info
        assert version_info["furiosa_runtime"] is not None

    @pytest.mark.api
    def test_version_format(self, api_client):
        """Test that version strings are in valid format"""
        session, base_url = api_client

        response = session.get(f"{base_url}/version")
        version_info = response.json()

        for key, value in version_info.items():
            assert isinstance(value, str)
            # Version should contain at least one dot (e.g., "0.1.0" or "2025.3.1")
            assert "." in value or value.isdigit()
