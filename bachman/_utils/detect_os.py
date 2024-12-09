"""
Detect the operating system.
"""

import platform


def detect_os():
    """Detect the operating system."""
    os_name = platform.system()
    if os_name == "Darwin":
        return "MAC"
    elif os_name == "Linux":
        return "LINUX"
    else:
        return "Unknown"
