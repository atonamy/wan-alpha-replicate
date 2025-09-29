<div align="center">

  <h1>
    Wan-Alpha
  </h1>

  <h3>Wan-Alpha: High-Quality Text-to-Video Generation with Alpha Channel</h3>



[![arXiv](https://img.shields.io/badge/arXiv-xxxx-b31b1b)](https://arxiv.org/abs/)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://www.xxxx)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/htdong/Wan-Alpha)

</div>

<img src="assets/teaser.png" alt="Wan-Alpha Qualitative Results" style="max-width: 100%; height: auto;">

>Qualitative results of video generation using **Wan-Alpha**. Our model successfully generates various scenes with accurate and clearly rendered transparency. Notably, it can synthesize diverse semi-transparent objects, glowing effects, and fine-grained details such as hair.

---

## 🔥 News
* **[2025.09.30]** Released Wan-Alpha v1.0, the Wan2.1-14B-T2V–adapted weights and inference code are now open-sourced.

---

## 🌟 Showcase

### Text-to-Video Generation with Alpha Channel
## 🌟 Showcase

### Text-to-Video Generation with Alpha Channel

<!-- | Prompt | Generated Video | Alpha Video |
| :---: | :---: | :---: |
| "Medium shot. A little girl holds a bubble wand and blows out colorful bubbles that float and pop in the air. The background of this video is transparent. Realistic style." |
  <div style="display: flex; gap: 10px;">
    <img src="girl.gif" alt="..." style="flex: 1; min-width: 200px;">
  </div> |
  <div style="display: flex; gap: 10px;">
    <img src="girl_pha.gif" alt="..." style="flex: 1; min-width: 200px;">
  </div> | -->
| Prompt | Generated Video | Alpha Video |
| :---: | :---: | :---: |
| "Medium shot. A little girl holds a bubble wand and blows out colorful bubbles that float and pop in the air. The background of this video is transparent. Realistic style." | <img src="assets/girl.gif" width="320" height="180" style="object-fit:contain; display:block; margin:auto;"/> | <img src="assets/girl_pha.gif" width="320" height="180" style="object-fit:contain; display:block; margin:auto;"/> |

### For more results, please visit [https://Wan-Alpha.github.io/](https://www.xxx)

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the project repository
git clone https://github.com/WeChatCV/Wan-Alpha.git
cd Wan-Alpha

# Create and activate Conda environment
conda create -n Wan-Alpha python=3.11 -y
conda activate Wan-Alpha

# Install dependencies
pip install -r requirements.txt
```
### 2. Model Download
Download [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B)

Download [Lightx2v-T2V-14B](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors)

Download [Wan-Alpha VAE](https://huggingface.co/htdong/Wan-Alpha)
## 🧪 Usage
```bash
bash test_lightx2v_dora.sh
```
**Prompt Writing Tip:**  You need to specify that the background of the video is transparent, the visual style, the shot type (such as close-up, medium shot, wide shot, or extreme close-up), and a description of the main subject. Prompts support both Chinese and English input.

```bash
# An example of prompt.
This video has a transparent background. Close-up shot. A colorful parrot flying. Realistic style.
```
## 🤝 Acknowledgements

This project is built upon the following excellent open-source projects:
* [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (training/inference framework)
* [Wan2.1](https://github.com/Wan-Video/Wan2.1) (base video generation model)
* [LightX2V](https://github.com/ModelTC/LightX2V) (inference acceleration)
* [WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy) (inference acceleration)

We sincerely thank the authors and contributors of these projects.

---

## ✏ Citation

If you find our work helpful for your research, please consider citing our paper:

```bibtex
@article{
}
```

---

## 📬 Contact Us

If you have any questions or suggestions, feel free to reach out via [GitHub Issues](https://github.com/WeChatCV/Wan-Alpha/issues) . We look forward to your feedback!
