"""
Viveka: Test-Time Refinement for Chest X-Ray Denoising.

Reference-free post-processing module for correcting residual artefacts
in denoised chest X-rays using physics-based constraints.
"""

from .refiner import VivekaRefiner

__version__ = "1.0.0"
__author__ = "Siju K S, Vipin Venugopal, and Soman KP"
__all__ = ["VivekaRefiner"]