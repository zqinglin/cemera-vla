"""VLA package init.

Keep init lightweight to avoid importing training / RLDS deps during inference.
If training utilities are needed, import explicitly from `prismatic.vla.materialize`.
"""

__all__ = []
