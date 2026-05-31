"""
Clinical metric utilities for pre-CLAHE Viveka output evaluation.
"""

import numpy as np
import cv2
from skimage.feature import graycomatrix


def standardize_image(image):
    img = np.squeeze(image).astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return np.clip(img, 0, 1)


def compute_cnr(image, noisy_input=None):
    image_std = standardize_image(image)
    mask_source = standardize_image(noisy_input) if noisy_input is not None else image_std
    mask_uint8 = (mask_source * 255).astype(np.uint8)

    edges = cv2.Canny(mask_uint8, 30, 100)
    bone_mask = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1).astype(np.float32) / 255.0
    bone_mask = cv2.GaussianBlur(bone_mask, (5, 5), 2.0)

    background_mask = (mask_source < 0.05).astype(np.float32)
    background_mask = cv2.dilate(background_mask, np.ones((3, 3), np.uint8))

    bone_pixels = image_std[bone_mask > 0.3]
    bg_pixels = image_std[background_mask > 0.3]

    if len(bone_pixels) < 10 or len(bg_pixels) < 10:
        return 0.0

    mu_bone = np.mean(bone_pixels)
    mu_bg = np.mean(bg_pixels)
    sigma_bg = np.std(bg_pixels)

    return float(np.abs(mu_bone - mu_bg) / sigma_bg) if sigma_bg > 1e-7 else 0.0


def compute_fwhm(image, noisy_input, num_edges=5):
    image_std = standardize_image(image)
    noisy_std = standardize_image(noisy_input)

    image_uint8 = (image_std * 255).astype(np.uint8)
    noisy_uint8 = (noisy_std * 255).astype(np.uint8)

    noisy_edges = cv2.Canny(noisy_uint8, 50, 150)
    image_edges = cv2.Canny(image_uint8, 50, 150)
    valid_edges = noisy_edges & image_edges
    edge_y, edge_x = np.where(valid_edges > 0)

    if len(edge_x) < 10:
        return float('inf')

    fwhm_values = []
    h, w = image_std.shape

    for _ in range(num_edges * 5):
        if len(fwhm_values) >= num_edges:
            break
        idx = np.random.randint(0, len(edge_x))
        ex, ey = edge_x[idx], edge_y[idx]
        y_start, y_end = max(0, ey - 30), min(h, ey + 30)
        if y_end - y_start < 10:
            continue
        profile = image_std[y_start:y_end, ex].astype(np.float64)
        prof_min, prof_max = profile.min(), profile.max()
        if prof_max - prof_min < 0.05:
            continue
        profile_norm = (profile - prof_min) / (prof_max - prof_min)
        above_half = profile_norm > 0.5
        if np.all(above_half) or not np.any(above_half):
            continue
        transitions = np.diff(above_half.astype(int))
        rise_idx = np.where(transitions == 1)[0]
        fall_idx = np.where(transitions == -1)[0]
        if len(rise_idx) == 0 or len(fall_idx) == 0:
            continue
        rise = rise_idx[0]
        fall_candidates = fall_idx[fall_idx > rise]
        if len(fall_candidates) == 0:
            continue
        fwhm_pixels = fall_candidates[0] - rise
        if fwhm_pixels >= 1:
            fwhm_values.append(fwhm_pixels)

    return float(np.median(fwhm_values)) if fwhm_values else float('inf')


def compute_glcm_entropy(image, distances=(1, 2), levels=64):
    image_std = standardize_image(image)
    lung_mask = np.ones_like(image_std, dtype=np.float32)
    lung_mask[image_std < 0.04] = 0.0
    lung_mask[image_std > 0.96] = 0.0
    mask_pixels = image_std[lung_mask >= 0.3]
    if len(mask_pixels) < 100:
        return 0.0
    p_min, p_max = mask_pixels.min(), mask_pixels.max()
    if p_max - p_min < 1e-7:
        return 0.0
    quantized = np.zeros_like(image_std, dtype=np.uint8)
    quantized[lung_mask >= 0.3] = (
        (image_std[lung_mask >= 0.3] - p_min) / (p_max - p_min) * (levels - 1)
    ).astype(np.uint8)
    glcm = graycomatrix(
        quantized, distances=distances,
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=levels, symmetric=True, normed=True
    )
    glcm_flat = glcm.flatten()
    glcm_nonzero = glcm_flat[glcm_flat > 0]
    entropy = -np.sum(glcm_nonzero * np.log2(glcm_nonzero))
    max_entropy = np.log2(levels * levels)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def compute_ttpi(denoised_image, refined_image):
    denoised_std = standardize_image(denoised_image)
    refined_std = standardize_image(refined_image)
    denoised_uint8 = (denoised_std * 255).astype(np.uint8)
    refined_uint8 = (refined_std * 255).astype(np.uint8)
    lap_denoised = cv2.Laplacian(denoised_uint8, cv2.CV_64F)
    lap_refined = cv2.Laplacian(refined_uint8, cv2.CV_64F)
    var_denoised = np.var(lap_denoised)
    var_refined = np.var(lap_refined)
    if var_denoised < 0.01:
        return 999.0 if var_refined > 0.01 else 1.0
    return float(np.clip(var_refined / var_denoised, 0.01, 100.0))


def compute_scp(denoised_image, refined_image):
    denoised_std = standardize_image(denoised_image)
    refined_std = standardize_image(refined_image)
    denoised_uint8 = (denoised_std * 255).astype(np.uint8)
    refined_uint8 = (refined_std * 255).astype(np.uint8)
    edge_d = np.sqrt(cv2.Sobel(denoised_uint8, cv2.CV_64F, 1, 0, ksize=3)**2 +
                     cv2.Sobel(denoised_uint8, cv2.CV_64F, 0, 1, ksize=3)**2)
    edge_r = np.sqrt(cv2.Sobel(refined_uint8, cv2.CV_64F, 1, 0, ksize=3)**2 +
                     cv2.Sobel(refined_uint8, cv2.CV_64F, 0, 1, ksize=3)**2)
    edge_d_flat = edge_d.flatten()
    edge_r_flat = edge_r.flatten()
    edge_d_norm = (edge_d_flat - np.mean(edge_d_flat)) / (np.std(edge_d_flat) + 1e-7)
    edge_r_norm = (edge_r_flat - np.mean(edge_r_flat)) / (np.std(edge_r_flat) + 1e-7)
    return float(np.clip(np.mean(edge_d_norm * edge_r_norm), 0.0, 2.0))