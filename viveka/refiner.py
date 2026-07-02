"""
Viveka Refiner: Physics-based test-time refinement for chest X-ray denoising.
"""

import numpy as np
import cv2


class VivekaRefiner:
    """
    Physics-based test-time refinement for denoised chest X-ray images.

    Parameters:
        guide_thresh: DCT guide threshold for denoised output (default: 1.5)
        input_thresh: DCT input threshold for noisy input (default: 3.0)
        spins: Number of cycle spinning shifts (default: 2)
        use_adaptive_gains: Enable patient-specific gain modulation
        use_pathology_preservation: Enable pathology preservation constraint
        use_uncertainty_guidance: Enable uncertainty-weighted refinement
        use_edge_protection: Enable edge protection weight map
        use_dct_detail: Enable DCT-based detail extraction
        refinement_strength: 'gentle', 'balanced', or 'strong'
    """

    def __init__(
        self,
        guide_thresh=1.5,
        input_thresh=3.0,
        spins=2,
        use_adaptive_gains=True,
        use_pathology_preservation=True,
        use_uncertainty_guidance=True,
        use_edge_protection=True,
        use_dct_detail=True,
        refinement_strength='balanced'
    ):
        self.Tg = guide_thresh
        self.Ti = input_thresh
        self.spins = spins
        self.use_adaptive_gains = use_adaptive_gains
        self.use_pathology_preservation = use_pathology_preservation
        self.use_uncertainty_guidance = use_uncertainty_guidance
        self.use_edge_protection = use_edge_protection
        self.use_dct_detail = use_dct_detail
        self.refinement_strength = refinement_strength

        if refinement_strength == 'gentle':
            self.clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(8, 8))
        elif refinement_strength == 'strong':
            self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        else:
            self.clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))

        self.default_bone_gain = 1.0
        self.default_lung_gain = 0.3
        self.default_bg_gain = 0.0

    def anscombe_forward(self, img):
        return 2.0 * np.sqrt(img + (3.0 / 8.0))

    def anscombe_inverse(self, img):
        return np.maximum(0, (img / 2.0) ** 2 - (3.0 / 8.0))

    def compute_adaptive_gains(self, image, bone_mask, lung_mask, model_type='X-GAN',
                               baseline_quality=None, baseline_piqe=None):
        bone_pixels = image[bone_mask > 0.5]
        if len(bone_pixels) > 0:
            bone_density = np.percentile(bone_pixels, 95)
            bone_gain = 0.8 + 0.4 * bone_density
            bone_gain = np.clip(bone_gain, 0.8, 1.2)
        else:
            bone_gain = self.default_bone_gain

        lung_pixels = image[lung_mask > 0.5]
        if len(lung_pixels) > 0:
            lung_texture = np.std(lung_pixels)
            base_model = model_type.replace('+Viveka', '')

            if base_model == 'CGAN':
                gain_multiplier = 0.5
                max_gain = 0.35
                min_gain = 0.15
            elif base_model in ['RIDNet', 'CBDNet', 'X-ReCNN']:
                gain_multiplier = 1.5
                max_gain = 0.55
                min_gain = 0.25
            else:
                gain_multiplier = 0.5
                max_gain = 0.40
                min_gain = 0.18

            lung_gain = min_gain + gain_multiplier * 1.5 * lung_texture
            lung_gain = np.clip(lung_gain, min_gain, max_gain)
        else:
            lung_gain = self.default_lung_gain

        return bone_gain, lung_gain, self.default_bg_gain

    def compute_uncertainty_map(self, gan_output, noisy_input):
        kernel_size = 7
        local_mean = cv2.GaussianBlur(gan_output, (kernel_size, kernel_size), 1.0)
        local_var = cv2.GaussianBlur(gan_output ** 2, (kernel_size, kernel_size), 1.0) - local_mean ** 2

        global_var = np.var(gan_output)
        if global_var > 0:
            uncertainty = np.clip(local_var / global_var, 0, 1)
        else:
            uncertainty = np.zeros_like(gan_output)

        diff_from_noisy = np.abs(gan_output - noisy_input)
        uncertainty = 0.5 * uncertainty + 0.5 * np.clip(diff_from_noisy * 1.0, 0, 1)
        return uncertainty

    def pathology_preservation_loss(self, refined, original):
        refined_uint8 = (np.clip(refined, 0, 1) * 255).astype(np.uint8)
        original_uint8 = (np.clip(original, 0, 1) * 255).astype(np.uint8)

        sobel_refined = cv2.Sobel(refined_uint8, cv2.CV_64F, 1, 1, ksize=3)
        sobel_original = cv2.Sobel(original_uint8, cv2.CV_64F, 1, 1, ksize=3)
        edge_diff = np.mean(np.abs(sobel_refined - sobel_original)) / 255.0

        laplacian_refined = cv2.Laplacian(refined_uint8, cv2.CV_64F)
        laplacian_original = cv2.Laplacian(original_uint8, cv2.CV_64F)
        highfreq_diff = np.mean(np.abs(laplacian_refined - laplacian_original)) / 255.0

        return (edge_diff + highfreq_diff) / 2.0

    def run_oracle_dct(self, gan_vst, noisy_vst):
        if not self.use_dct_detail:
            return gan_vst

        h, w = gan_vst.shape
        output = np.zeros_like(gan_vst)
        weights = np.zeros_like(gan_vst)
        P = 8
        stride = 4

        for y in range(0, h - P + 1, stride):
            for x in range(0, w - P + 1, stride):
                patch_g = gan_vst[y:y + P, x:x + P]
                patch_n = noisy_vst[y:y + P, x:x + P]

                dct_g = cv2.dct(patch_g)
                dct_n = cv2.dct(patch_n)

                mask = np.logical_or(np.abs(dct_g) > self.Tg, np.abs(dct_n) > self.Ti)
                dct_filtered = dct_n * mask.astype(np.float32)

                output[y:y + P, x:x + P] += cv2.idct(dct_filtered)
                weights[y:y + P, x:x + P] += 1.0

        return output / (weights + 1e-5)

    def refine(self, gan_prediction, original_noisy_input, model_type='X-GAN',
               baseline_quality=None, baseline_piqe=None, return_pre_clahe=False):
        """
        Refine a denoised chest X-ray image.

        Args:
            gan_prediction: Denoiser output, shape (H, W) or (H, W, 1), range [0, 1]
            original_noisy_input: Noisy image, shape (H, W), range [0, 1]
            model_type: 'X-GAN', 'RIDNet', 'CBDNet', 'X-ReCNN', or 'CGAN'
            return_pre_clahe: If True, returns (post_clahe, pre_clahe)

        Returns:
            Refined image, or (post_clahe, pre_clahe) if return_pre_clahe=True
        """
        manas_clean = np.clip(np.squeeze(gan_prediction), 0, 1)

        gan_scaled = manas_clean * 255.0
        noisy_scaled = np.clip(original_noisy_input, 0, 1) * 255.0
        gan_vst = self.anscombe_forward(gan_scaled)
        noisy_vst = self.anscombe_forward(noisy_scaled)

        shifts = [(0, 0), (4, 4)]
        acc = np.zeros_like(gan_vst)
        for dy, dx in shifts:
            res = self.run_oracle_dct(
                np.roll(gan_vst, (dy, dx), (0, 1)),
                np.roll(noisy_vst, (dy, dx), (0, 1))
            )
            acc += np.roll(res, (-dy, -dx), (0, 1))
        viveka_sharp = self.anscombe_inverse(acc / len(shifts)) / 255.0
        viveka_sharp = np.clip(viveka_sharp, 0, 1)

        manas_uint8 = (manas_clean * 255).astype(np.uint8)
        edges = cv2.Canny(manas_uint8, 30, 100)
        bone_mask = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1).astype(np.float32) / 255.0
        bone_mask = cv2.GaussianBlur(bone_mask, (0, 0), 2.0)
        bg_mask = (manas_clean < 0.05).astype(np.float32)
        bg_mask = cv2.dilate(bg_mask, np.ones((3, 3), np.uint8))
        lung_mask = 1.0 - np.maximum(bone_mask, bg_mask)

        if self.use_adaptive_gains:
            bone_gain, lung_gain, bg_gain = self.compute_adaptive_gains(
                manas_clean, bone_mask, lung_mask, model_type,
                baseline_quality, baseline_piqe
            )
        else:
            bone_gain, lung_gain, bg_gain = (
                self.default_bone_gain, self.default_lung_gain, self.default_bg_gain
            )

        if self.use_uncertainty_guidance:
            uncertainty = self.compute_uncertainty_map(manas_clean, original_noisy_input)
            uncertainty_weight = 1.0 + uncertainty
        else:
            uncertainty_weight = np.ones_like(manas_clean)

        if self.use_edge_protection:
            sobel_mag = np.sqrt(
                cv2.Sobel(manas_clean, cv2.CV_64F, 1, 0, ksize=3) ** 2 +
                cv2.Sobel(manas_clean, cv2.CV_64F, 0, 1, ksize=3) ** 2
            )
            protection_map = np.exp(-20.0 * sobel_mag)
        else:
            protection_map = np.ones_like(manas_clean)

        weight_map = np.ones_like(manas_clean) * lung_gain * uncertainty_weight * protection_map
        weight_map = weight_map * (1.0 - bg_mask)
        weight_map = np.maximum(weight_map, bone_mask * bone_gain)

        detail_layer = viveka_sharp - manas_clean
        final_image = manas_clean + (detail_layer * weight_map)

        if self.use_pathology_preservation:
            pathology_loss = self.pathology_preservation_loss(final_image, manas_clean)
            if pathology_loss > 0.1:
                correction_factor = 1.0 - (pathology_loss - 0.1) * 2.0
                correction_factor = np.clip(correction_factor, 0.5, 1.0)
                final_image = manas_clean + (detail_layer * weight_map * correction_factor)

        pre_clahe = final_image.copy()

        final_uint8 = (np.clip(final_image, 0, 1) * 255).astype(np.uint8)
        final_pop = self.clahe.apply(final_uint8)
        final_output = final_pop.astype(np.float32) / 255.0

        if return_pre_clahe:
            return final_output, pre_clahe
        return final_output