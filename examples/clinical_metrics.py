"""
Clinical metric computation on pre-CLAHE Viveka output.
Demonstrates CNR, FWHM, GLCM entropy, TTPI, and SCP.
"""

import numpy as np
import cv2
from viveka import VivekaRefiner
from viveka.utils import (
    compute_cnr, compute_fwhm, compute_glcm_entropy,
    compute_ttpi, compute_scp
)


def main():
    print("Viveka Clinical Metrics Example")
    print("=" * 40)

    # Create synthetic test images
    np.random.seed(42)
    noisy = np.clip(
        np.random.rand(256, 256) + 0.1 * np.random.randn(256, 256), 0, 1
    ).astype(np.float32)
    denoised = cv2.GaussianBlur(noisy, (5, 5), 0.8)

    # Initialize Viveka
    refiner = VivekaRefiner(refinement_strength='balanced')

    # Get pre-CLAHE output for clinical metric evaluation
    _, pre_clahe = refiner.refine(denoised, noisy, return_pre_clahe=True)

    print("\nClinical Metrics (computed on pre-CLAHE output):")
    print("-" * 40)

    cnr = compute_cnr(pre_clahe, noisy_input=noisy)
    print(f"CNR:  {cnr:.2f}  (higher = better tissue differentiation)")

    fwhm = compute_fwhm(pre_clahe, noisy_input=noisy)
    if fwhm < float('inf'):
        print(f"FWHM: {fwhm:.1f} pixels  (lower = sharper edges)")
    else:
        print("FWHM: unmeasurable (no valid edges found)")

    entropy = compute_glcm_entropy(pre_clahe)
    print(f"GLCM Entropy: {entropy:.3f}  (near noisy reference ≈5.8 = preserved texture)")

    ttpi = compute_ttpi(denoised, pre_clahe)
    if ttpi > 100:
        print(f"TTPI: {ttpi:.1f}  (texture recovered from zero baseline)")
    else:
        print(f"TTPI: {ttpi:.2f}  (≈1.0 = minimal texture change)")

    scp = compute_scp(denoised, pre_clahe)
    print(f"SCP:  {scp:.3f}  (≈1.0 = preserved structure)")

    print("\nNote: These metrics are computed on synthetic data.")
    print("Values on real chest X-rays will differ.")


if __name__ == '__main__':
    main()