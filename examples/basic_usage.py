"""
Basic usage example for Viveka refinement.
Demonstrates post-CLAHE and pre-CLAHE output.
"""

import numpy as np
import cv2
from viveka import VivekaRefiner


def main():
    print("Viveka Basic Usage Example")
    print("=" * 40)

    # Create synthetic test images (256x256, range [0, 1])
    np.random.seed(42)
    noisy = np.clip(
        np.random.rand(256, 256) + 0.1 * np.random.randn(256, 256), 0, 1
    ).astype(np.float32)
    denoised = cv2.GaussianBlur(noisy, (5, 5), 0.8)

    print(f"Noisy image range: [{noisy.min():.3f}, {noisy.max():.3f}]")
    print(f"Denoised image range: [{denoised.min():.3f}, {denoised.max():.3f}]")

    # Initialize Viveka
    refiner = VivekaRefiner(refinement_strength='balanced')
    print(f"\nRefiner initialized: strength={refiner.refinement_strength}")

    # Post-CLAHE output (for visual display)
    refined_post = refiner.refine(denoised, noisy, model_type='X-GAN')
    print(f"\nPost-CLAHE output range: [{refined_post.min():.3f}, {refined_post.max():.3f}]")

    # Pre-CLAHE output (for clinical metrics)
    refined_post, refined_pre = refiner.refine(
        denoised, noisy, model_type='X-GAN', return_pre_clahe=True
    )
    print(f"Pre-CLAHE output range: [{refined_pre.min():.3f}, {refined_pre.max():.3f}]")

    # Demonstrate component toggling
    print("\nComponent Ablation Examples:")
    configs = [
        ("w/o DCT", {"use_dct_detail": False}),
        ("w/o Edge Protection", {"use_edge_protection": False}),
        ("w/o Adaptive Gains", {"use_adaptive_gains": False}),
    ]
    for name, kwargs in configs:
        r = VivekaRefiner(**kwargs)
        out = r.refine(denoised, noisy)
        diff = np.mean(np.abs(out - refined_post))
        print(f"  {name}: mean |diff| from Full Viveka = {diff:.6f}")

    print("\nDone.")


if __name__ == '__main__':
    main()