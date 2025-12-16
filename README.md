
# Moodscape: Emotion-Driven Terrain Synthesis for Virtual Reality Systems

This repository contains code for generating synthetic terrain heightmaps from a trained Dual-Head VAE-GAN model (DH-CVAE-GAN).

---

## Video Overview: Mood-Based Terrain Generation

In the video, we demonstrate the VR system where users input mood parameters — Valence (V) and Arousal (A) — via a UI panel using a controller. Based on the selected mood values, the system generates a terrain that corresponds to the emotional state.

The UI panel also offers:
- Direct selection of specific terrain types (Mountain, Desert, Plateau, Coastal)
- An option to play music (terrain-from-music mood is beyond the scope of this journal, but adds interactivity)

The video shows how varying Valence–Arousal values alter the generated environments in real time. Although the system currently supports four terrain types, it can be extended to up to eight, as discussed in the paper.

👉 **Video Link (YouTube):** https://youtu.be/GuNz8bGriLQ?si=RASaMvQOSGB-CYho

---

## Downloads

Please download the pretrained artifacts and place them in the correct folders:

- **Model checkpoints**  
  👉 [Click here to download the model checkpoints](https://drive.google.com/uc?export=download&id=1yo5E_LeI8bD2Zy40sLHRNjT1di6JLTX2)  
  Save the downloaded file(s) in the `checkpoints/` folder.

- **Latent bank**  
  👉 [Click here to download the latent codes](https://drive.google.com/uc?export=download&id=1QhH4bmfDysif83L9PIVoVHbkA9tOqcoc)  
  Save the downloaded file(s) in the `latent/` folder.

---

## Sampling from Valence–Arousal Inputs

To generate a terrain from Valence (V) and Arousal (A), both in the range `[-1, 1]`:

- Given (V, A), the script generates terrain samples and saves them in the `output/` folder.

### Usage

```bash
cd src

# Example 1: moderately positive valence and moderately aroused
python sample_from_va.py --valence 0.5 --arousal 0.2

# Example 2: negative valence, low arousal (more “desert-like”)
python sample_from_va.py --valence -0.8 --arousal -0.3
