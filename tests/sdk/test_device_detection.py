"""
Test Suite: SDK Device Detection Simulation
Simulates furiosa.runtime device detection behavior

Note: These tests simulate NPU device detection without actual hardware.
In production, these would interface with furiosa.runtime.get_devices()
"""

import pytest
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import List, Optional


# ============================================================================
# Mock Device Classes (Simulating furiosa.runtime)
# ============================================================================


@dataclass
class MockDeviceInfo:
    """Simulates furiosa device info"""

    name: str
    device_type: str = "RNGD"
    arch: str = "rngd"
    firmware_version: str = "1.0.0"
    driver_version: str = "2025.3.1"


@dataclass
class MockDevice:
    """Simulates furiosa NPU device"""

    device_id: int
    info: MockDeviceInfo
    status: str = "available"

    def device_info(self) -> MockDeviceInfo:
        return self.info

    def is_available(self) -> bool:
        return self.status == "available"


class MockRuntime:
    """Simulates furiosa.runtime module"""

    def __init__(self, devices: List[MockDevice] = None):
        self._devices = devices or []

    def get_devices(self) -> List[MockDevice]:
        return self._devices

    def list_devices(self) -> List[MockDevice]:
        return self._devices


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_single_device():
    """Single NPU device"""
    return MockRuntime([MockDevice(device_id=0, info=MockDeviceInfo(name="npu0"))])


@pytest.fixture
def mock_multiple_devices():
    """Multiple NPU devices"""
    return MockRuntime(
        [
            MockDevice(device_id=0, info=MockDeviceInfo(name="npu0")),
            MockDevice(device_id=1, info=MockDeviceInfo(name="npu1")),
            MockDevice(device_id=2, info=MockDeviceInfo(name="npu2")),
            MockDevice(device_id=3, info=MockDeviceInfo(name="npu3")),
        ]
    )


@pytest.fixture
def mock_no_devices():
    """No NPU devices"""
    return MockRuntime([])


@pytest.fixture
def mock_unavailable_device():
    """Device that is not available"""
    return MockRuntime(
        [MockDevice(device_id=0, info=MockDeviceInfo(name="npu0"), status="busy")]
    )


# ============================================================================
# Tests
# ============================================================================


class TestDeviceDetection:
    """Test NPU Device Detection"""

    @pytest.mark.sdk
    def test_detect_single_device(self, mock_single_device):
        """Test detecting a single NPU device"""
        devices = mock_single_device.get_devices()

        assert len(devices) == 1
        assert devices[0].device_info().name == "npu0"

    @pytest.mark.sdk
    def test_detect_multiple_devices(self, mock_multiple_devices):
        """Test detecting multiple NPU devices"""
        devices = mock_multiple_devices.get_devices()

        assert len(devices) == 4
        for i, device in enumerate(devices):
            assert device.device_info().name == f"npu{i}"

    @pytest.mark.sdk
    def test_no_devices_available(self, mock_no_devices):
        """Test behavior when no NPU devices are found"""
        devices = mock_no_devices.get_devices()

        assert len(devices) == 0

    @pytest.mark.sdk
    def test_device_info_properties(self, mock_single_device):
        """Test device info properties"""
        devices = mock_single_device.get_devices()
        info = devices[0].device_info()

        assert info.device_type == "RNGD"
        assert info.arch == "rngd"
        assert info.firmware_version is not None
        assert info.driver_version is not None


class TestDeviceAvailability:
    """Test Device Availability Checks"""

    @pytest.mark.sdk
    def test_device_is_available(self, mock_single_device):
        """Test that device reports as available"""
        devices = mock_single_device.get_devices()

        assert devices[0].is_available() is True

    @pytest.mark.sdk
    def test_device_is_unavailable(self, mock_unavailable_device):
        """Test that busy device reports as unavailable"""
        devices = mock_unavailable_device.get_devices()

        assert devices[0].is_available() is False

    @pytest.mark.sdk
    def test_filter_available_devices(self, mock_multiple_devices):
        """Test filtering for available devices only"""
        devices = mock_multiple_devices.get_devices()
        available = [d for d in devices if d.is_available()]

        assert len(available) == 4


class TestDeviceSelection:
    """Test Device Selection Logic"""

    @pytest.mark.sdk
    def test_select_first_available_device(self, mock_multiple_devices):
        """Test selecting first available device"""
        devices = mock_multiple_devices.get_devices()
        available = [d for d in devices if d.is_available()]

        if available:
            selected = available[0]
            assert selected.device_id == 0

    @pytest.mark.sdk
    def test_select_device_by_id(self, mock_multiple_devices):
        """Test selecting device by specific ID"""
        devices = mock_multiple_devices.get_devices()
        target_id = 2

        selected = next((d for d in devices if d.device_id == target_id), None)

        assert selected is not None
        assert selected.device_id == target_id

    @pytest.mark.sdk
    def test_device_not_found_by_id(self, mock_multiple_devices):
        """Test selecting non-existent device ID"""
        devices = mock_multiple_devices.get_devices()
        target_id = 999

        selected = next((d for d in devices if d.device_id == target_id), None)

        assert selected is None


class TestDeviceNamingConvention:
    """Test Device Naming Convention"""

    @pytest.mark.sdk
    def test_device_name_format(self, mock_multiple_devices):
        """Test that device names follow npu{N} format"""
        devices = mock_multiple_devices.get_devices()

        for device in devices:
            name = device.device_info().name
            assert name.startswith("npu")
            # Extract number after "npu"
            npu_index = int(name[3:])
            assert npu_index >= 0

    @pytest.mark.sdk
    def test_device_index_matches_name(self, mock_multiple_devices):
        """Test that device_id matches name index"""
        devices = mock_multiple_devices.get_devices()

        for device in devices:
            name = device.device_info().name
            npu_index = int(name[3:])
            assert device.device_id == npu_index
