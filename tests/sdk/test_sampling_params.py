"""
Test Suite: SamplingParams Simulation
Simulates furiosa_llm.SamplingParams behavior

Based on Furiosa LLM SamplingParams specification:
https://developer.furiosa.ai/latest/en/furiosa_llm/reference/sampling_params.html
"""

import pytest
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Mock SamplingParams (Simulating furiosa_llm.SamplingParams)
# ============================================================================

@dataclass
class MockSamplingParams:
    """
    Simulates furiosa_llm.SamplingParams
    
    Based on Furiosa SDK 2025.3.1 specification
    """
    n: int = 1
    best_of: int = 1
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    use_beam_search: bool = False
    length_penalty: float = 1.0
    early_stopping: bool = False
    max_tokens: int = 16
    min_tokens: int = 0
    logprobs: Optional[int] = None
    
    def __post_init__(self):
        """Validate parameters after initialization"""
        self._validate()
    
    def _validate(self):
        """Validate parameter values"""
        if self.n < 1:
            raise ValueError("n must be at least 1")
        if self.best_of < self.n:
            raise ValueError("best_of must be >= n")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not (0 <= self.top_p <= 1):
            raise ValueError("top_p must be between 0 and 1")
        if not (0 <= self.min_p <= 1):
            raise ValueError("min_p must be between 0 and 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.min_tokens < 0:
            raise ValueError("min_tokens must be non-negative")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must be <= max_tokens")


# ============================================================================
# Tests
# ============================================================================

class TestSamplingParamsDefaults:
    """Test SamplingParams Default Values"""

    @pytest.mark.sdk
    def test_default_values(self):
        """Test that default values are correctly set"""
        params = MockSamplingParams()
        
        assert params.n == 1
        assert params.best_of == 1
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k == -1
        assert params.min_p == 0.0
        assert params.use_beam_search is False
        assert params.max_tokens == 16
        assert params.min_tokens == 0

    @pytest.mark.sdk
    def test_custom_values(self):
        """Test setting custom parameter values"""
        params = MockSamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=100
        )
        
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.max_tokens == 100


class TestSamplingParamsValidation:
    """Test SamplingParams Validation"""

    @pytest.mark.sdk
    def test_invalid_n(self):
        """Test that n < 1 raises error"""
        with pytest.raises(ValueError, match="n must be at least 1"):
            MockSamplingParams(n=0)

    @pytest.mark.sdk
    def test_invalid_best_of(self):
        """Test that best_of < n raises error"""
        with pytest.raises(ValueError, match="best_of must be >= n"):
            MockSamplingParams(n=2, best_of=1)

    @pytest.mark.sdk
    def test_negative_temperature(self):
        """Test that negative temperature raises error"""
        with pytest.raises(ValueError, match="temperature must be non-negative"):
            MockSamplingParams(temperature=-0.5)

    @pytest.mark.sdk
    def test_invalid_top_p_high(self):
        """Test that top_p > 1 raises error"""
        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            MockSamplingParams(top_p=1.5)

    @pytest.mark.sdk
    def test_invalid_top_p_low(self):
        """Test that top_p < 0 raises error"""
        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            MockSamplingParams(top_p=-0.1)

    @pytest.mark.sdk
    def test_invalid_min_p(self):
        """Test that invalid min_p raises error"""
        with pytest.raises(ValueError, match="min_p must be between 0 and 1"):
            MockSamplingParams(min_p=2.0)

    @pytest.mark.sdk
    def test_invalid_max_tokens(self):
        """Test that max_tokens < 1 raises error"""
        with pytest.raises(ValueError, match="max_tokens must be at least 1"):
            MockSamplingParams(max_tokens=0)

    @pytest.mark.sdk
    def test_invalid_min_tokens(self):
        """Test that negative min_tokens raises error"""
        with pytest.raises(ValueError, match="min_tokens must be non-negative"):
            MockSamplingParams(min_tokens=-1)

    @pytest.mark.sdk
    def test_min_tokens_exceeds_max(self):
        """Test that min_tokens > max_tokens raises error"""
        with pytest.raises(ValueError, match="min_tokens must be <= max_tokens"):
            MockSamplingParams(min_tokens=100, max_tokens=50)


class TestSamplingParamsTemperature:
    """Test Temperature Parameter Behavior"""

    @pytest.mark.sdk
    @pytest.mark.parametrize("temp", [0.0, 0.1, 0.5, 1.0, 2.0])
    def test_valid_temperature_values(self, temp):
        """Test various valid temperature values"""
        params = MockSamplingParams(temperature=temp)
        assert params.temperature == temp

    @pytest.mark.sdk
    def test_zero_temperature(self):
        """Test temperature=0 (greedy decoding)"""
        params = MockSamplingParams(temperature=0.0)
        assert params.temperature == 0.0


class TestSamplingParamsTopK:
    """Test top_k Parameter Behavior"""

    @pytest.mark.sdk
    def test_top_k_disabled(self):
        """Test top_k=-1 (disabled)"""
        params = MockSamplingParams(top_k=-1)
        assert params.top_k == -1

    @pytest.mark.sdk
    @pytest.mark.parametrize("k", [1, 10, 50, 100])
    def test_valid_top_k_values(self, k):
        """Test various valid top_k values"""
        params = MockSamplingParams(top_k=k)
        assert params.top_k == k


class TestSamplingParamsBeamSearch:
    """Test Beam Search Parameters"""

    @pytest.mark.sdk
    def test_beam_search_disabled_by_default(self):
        """Test beam search is disabled by default"""
        params = MockSamplingParams()
        assert params.use_beam_search is False

    @pytest.mark.sdk
    def test_beam_search_enabled(self):
        """Test enabling beam search"""
        params = MockSamplingParams(use_beam_search=True, best_of=4)
        assert params.use_beam_search is True
        assert params.best_of == 4

    @pytest.mark.sdk
    def test_beam_search_with_length_penalty(self):
        """Test beam search with length penalty"""
        params = MockSamplingParams(
            use_beam_search=True,
            best_of=4,
            length_penalty=0.8
        )
        assert params.length_penalty == 0.8

    @pytest.mark.sdk
    def test_beam_search_with_early_stopping(self):
        """Test beam search with early stopping"""
        params = MockSamplingParams(
            use_beam_search=True,
            best_of=4,
            early_stopping=True
        )
        assert params.early_stopping is True
