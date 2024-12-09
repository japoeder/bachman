"""
Get the path for a given label.
"""
import os
from bachman._utils.detect_os import detect_os


def get_path(path_label: str):
    """
    Get the path for a given label.
    """
    if path_label == "creds":
        return os.getenv("APP_PATH_" + detect_os()) + "/_cred/creds.json"
    elif path_label == "job_ctrl":
        return os.getenv("APP_PATH_" + detect_os()) + "/_job_ctrl/stream_load_ctrl.json"
    elif path_label == "log":
        return os.getenv("PROJ_PATH_" + detect_os()) + "/app.log"
    else:
        return False
