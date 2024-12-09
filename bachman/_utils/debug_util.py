"""
Debugging utilities.
"""

import debugpy

debugpy.listen(5678)
debugpy.wait_for_client()


def bp():
    """
    Breakpoint.
    """
    return debugpy.breakpoint()
