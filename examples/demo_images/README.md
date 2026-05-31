# Demo Images

Sample chest X-ray images for testing Viveka.

## Contents

| File | Description |
|------|-------------|
| `noisy1.png` | Original noisy chest X-ray (256×256, 8-bit grayscale) |
| `denoised1.png` | X-GAN denoised output of `noisy1.png` (256×256, 8-bit grayscale) |

## Usage

### With the Live Demo

1. Go to [viveka-module.streamlit.app](https://viveka-module.streamlit.app/)
2. Download `noisy1.png` and `denoised1.png` from this folder
3. Upload them to the demo
4. Select model type: `X-GAN`
5. Choose refinement strength: `balanced`
6. Click "Refine" to see Viveka's output

### With Local Installation

```bash
git clone https://github.com/sijuswamyresearch/viveka.git
cd viveka
pip install -r requirements.txt
```
### Source
These images are from the Stanford CheXpert dataset (Irvin et al., 2019), resized to 256×256 pixels. The denoised output was generated using RIDNet (Siju et al., 2026).




