class PackingFailure(Exception):
    """Raised when a cuboid cannot be placed and fail_fast=True."""


class InfeasibleItem(Exception):
    """Raised when a cuboid is larger than the container in all orientations."""
