import numpy as np
from viveka import VivekaRefiner


def test_initialization():
    refiner = VivekaRefiner()
    assert refiner.Tg == 1.5
    assert refiner.refinement_strength == 'balanced'


def test_output_shape():
    noisy = np.random.rand(256, 256).astype(np.float32)
    denoised = np.clip(noisy + 0.05 * np.random.randn(256, 256), 0, 1)
    refiner = VivekaRefiner()
    output = refiner.refine(denoised, noisy)
    assert output.shape == (256, 256)


def test_pre_clahe():
    noisy = np.random.rand(256, 256).astype(np.float32)
    denoised = np.clip(noisy + 0.05 * np.random.randn(256, 256), 0, 1)
    refiner = VivekaRefiner()
    post, pre = refiner.refine(denoised, noisy, return_pre_clahe=True)
    assert post.shape == (256, 256)
    assert pre.shape == (256, 256)


def test_all_components_disabled():
    noisy = np.random.rand(256, 256).astype(np.float32)
    denoised = np.clip(noisy + 0.05 * np.random.randn(256, 256), 0, 1)
    refiner = VivekaRefiner(
        use_adaptive_gains=False,
        use_pathology_preservation=False,
        use_uncertainty_guidance=False,
        use_edge_protection=False,
        use_dct_detail=False
    )
    output = refiner.refine(denoised, noisy)
    assert output.shape == (256, 256)


def test_all_strengths():
    noisy = np.random.rand(256, 256).astype(np.float32)
    denoised = np.clip(noisy + 0.05 * np.random.randn(256, 256), 0, 1)
    for strength in ['gentle', 'balanced', 'strong']:
        refiner = VivekaRefiner(refinement_strength=strength)
        output = refiner.refine(denoised, noisy)
        assert output.shape == (256, 256)