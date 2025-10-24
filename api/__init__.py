"""
Built-in PyNet APIs

Some of these APIs are not compatible with each other, such as Synapse and Multinet.
Synapse is designed around sequential models, while Multinet is designed around parallel models.
"""

__all__ = ["NetCore", "NetFlash", "NetLab"]
__package__ = "pynet"