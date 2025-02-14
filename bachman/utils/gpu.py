"""
This module provides utilities for checking GPU availability and setting the appropriate torch device.
"""

import logging
import platform
import os
import torch

logger = logging.getLogger(__name__)


def check_gpu_availability():
    """
    Check for GPU availability based on system architecture and available hardware.
    Returns a tuple of (device_type, device_name).
    """
    try:
        # First check for environment override
        forced_device = os.environ.get("BACHMAN_DEVICE_TYPE")
        if forced_device:
            logger.info(f"Using forced device type from environment: {forced_device}")
            return forced_device, f"Forced {forced_device.upper()}"

        system = platform.system()
        processor = platform.processor()
        logger.info(f"Detected system: {system}, Processor: {processor}")
        logger.info(f"PyTorch version: {torch.__version__}")

        # Production: Check for CUDA first
        if torch.cuda.is_available():
            device_type = "cuda"
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA GPU available: {device_name}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            return device_type, device_name

        # Development: Check for Apple Silicon
        if system == "Darwin" and "arm" in processor.lower():
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_type = "mps"
                device_name = "Apple Silicon"
                logger.info("Apple Silicon (MPS) available for development")
                return device_type, device_name
            else:
                logger.warning("Apple Silicon detected but MPS not available")

        # If no GPU available
        logger.warning("No GPU detected, using CPU")
        if system == "Linux":
            logger.warning(
                "On Linux but CUDA not available - check NVIDIA drivers and CUDA installation"
            )
        return "cpu", "CPU"

    except Exception as e:
        logger.warning(f"Error checking GPU availability: {str(e)}")
        logger.info("Defaulting to CPU")
        return "cpu", "CPU"


def get_device():
    """
    Get the appropriate torch device.
    Returns a torch.device object.
    """
    device_type, _ = check_gpu_availability()
    return torch.device(device_type)


def get_system_info():
    """
    Get detailed system information for debugging.
    """
    return {
        "system": platform.system(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        # "cuda_available": torch.cuda.is_available(),
        # "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "mps_available": torch.backends.mps.is_available()
        if hasattr(torch.backends, "mps")
        else False,
        # "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "environment_device": os.environ.get("BACHMAN_DEVICE_TYPE", None),
    }
