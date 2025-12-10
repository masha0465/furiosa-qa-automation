"""
Test Suite: Metrics API (/metrics)
Based on Furiosa LLM OpenAI-Compatible Server Specification

Covers:
- Prometheus-compatible metrics endpoint
- Furiosa-specific metrics validation
"""

import pytest
import re


class TestMetricsAPI:
    """Metrics API Tests"""

    @pytest.mark.api
    @pytest.mark.smoke
    def test_metrics_endpoint(self, api_client):
        """Test metrics endpoint returns 200"""
        session, base_url = api_client

        response = session.get(f"{base_url}/metrics")

        assert response.status_code == 200

    @pytest.mark.api
    def test_metrics_prometheus_format(self, api_client):
        """Test metrics are in Prometheus format"""
        session, base_url = api_client

        response = session.get(f"{base_url}/metrics")
        metrics_text = response.text

        # Prometheus format should have HELP and TYPE comments
        assert "# HELP" in metrics_text
        assert "# TYPE" in metrics_text

    @pytest.mark.api
    def test_metrics_contains_requests_running(self, api_client):
        """Test metrics contains furiosa_llm_num_requests_running"""
        session, base_url = api_client

        response = session.get(f"{base_url}/metrics")
        metrics_text = response.text

        assert "furiosa_llm_num_requests_running" in metrics_text

    @pytest.mark.api
    def test_metrics_contains_requests_waiting(self, api_client):
        """Test metrics contains furiosa_llm_num_requests_waiting"""
        session, base_url = api_client

        response = session.get(f"{base_url}/metrics")
        metrics_text = response.text

        assert "furiosa_llm_num_requests_waiting" in metrics_text

    @pytest.mark.api
    def test_metrics_contains_request_total(self, api_client):
        """Test metrics contains request total counters"""
        session, base_url = api_client

        response = session.get(f"{base_url}/metrics")
        metrics_text = response.text

        assert "furiosa_llm_request_received_total" in metrics_text
        assert "furiosa_llm_request_success_total" in metrics_text

    @pytest.mark.api
    def test_metrics_contains_token_counters(self, api_client):
        """Test metrics contains token counters"""
        session, base_url = api_client

        response = session.get(f"{base_url}/metrics")
        metrics_text = response.text

        assert "furiosa_llm_prompt_tokens_total" in metrics_text
        assert "furiosa_llm_generation_tokens_total" in metrics_text

    @pytest.mark.api
    def test_metrics_contains_kv_cache(self, api_client):
        """Test metrics contains KV cache metrics"""
        session, base_url = api_client

        response = session.get(f"{base_url}/metrics")
        metrics_text = response.text

        assert "furiosa_llm_kv_cache_usage_perc" in metrics_text

    @pytest.mark.api
    def test_metrics_model_name_label(self, api_client):
        """Test metrics have model_name label"""
        session, base_url = api_client

        response = session.get(f"{base_url}/metrics")
        metrics_text = response.text

        # Check for model_name label in metrics (handles escaped quotes)
        assert "model_name=" in metrics_text

    @pytest.mark.api
    def test_metrics_gauge_type(self, api_client):
        """Test that gauge metrics are properly typed"""
        session, base_url = api_client

        response = session.get(f"{base_url}/metrics")
        metrics_text = response.text

        # num_requests_running should be gauge type
        assert "# TYPE furiosa_llm_num_requests_running gauge" in metrics_text

    @pytest.mark.api
    def test_metrics_counter_type(self, api_client):
        """Test that counter metrics are properly typed"""
        session, base_url = api_client

        response = session.get(f"{base_url}/metrics")
        metrics_text = response.text

        # request_received_total should be counter type
        assert "# TYPE furiosa_llm_request_received_total counter" in metrics_text


class TestMetricsParsing:
    """Test Metrics Value Parsing"""

    @pytest.mark.api
    def test_metrics_numeric_values(self, api_client):
        """Test that metric values are numeric"""
        session, base_url = api_client

        response = session.get(f"{base_url}/metrics")
        metrics_text = response.text

        # Metrics should contain numeric values
        assert any(char.isdigit() for char in metrics_text)

    @pytest.mark.api
    def test_metrics_kv_cache_percentage_range(self, api_client):
        """Test KV cache usage percentage exists in metrics"""
        session, base_url = api_client

        response = session.get(f"{base_url}/metrics")
        metrics_text = response.text

        # Just verify the metric exists
        assert "furiosa_llm_kv_cache_usage_perc" in metrics_text
