"""
Sample one terrain heightmap given Valence (V) and Arousal (A) in [-1, 1].

Steps:
1. Map (V, A) to nearest terrain (Mountain / Desert / Coastal / Plateau)
   using Euclidean distance in V-A space and softmax similarity scores.
2. Print similarity scores (%) and the corresponding 2-bit code.
3. Use the chosen terrain's class in the latent bank to sample a latent
   vector and decode it using the trained DH-CVAE-GAN generator.
4. Save the generated image in 'samples/'.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from model import DualHeadVAEGenerator
from utils import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    TERRAIN_TO_CLASSNAME,
    pick_terrain_from_va,
    denorm_to_uint8,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Paths ----------------
CHECKPOINT = Path("checkpoints/dh_vae_gen.pth")
LATENT_BANK = Path("latent_bank/latent_bank_classwise.npz")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valence", type=float, required=True,
                        help="Valence in [-1, 1]")
    parser.add_argument("--arousal", type=float, required=True,
                        help="Arousal in [-1, 1]")
    args = parser.parse_args()

    v = args.valence
    a = args.arousal

    # 1) Map (V, A) -> terrain and code
    best_terrain, best_code, sims = pick_terrain_from_va(v, a)

    print(f"\nInput V={v:.2f}, A={a:.2f}")
    print("\nSimilarity Scores (%):")
    for name in sorted(sims, key=sims.get, reverse=True):
        print(f"  {name:8s}: {sims[name] * 100:5.1f}%")

    print(f"\nPredicted terrain : {best_terrain}")

    # Map to internal class name and index used by model & latent bank
    class_name = TERRAIN_TO_CLASSNAME[best_terrain]
    cls_idx = CLASS_TO_IDX[class_name]

    # 2) Load generator
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    gen = DualHeadVAEGenerator(num_classes=4, latent_dim=128).to(device)
    gen.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    gen.eval()

    # 3) Load latent bank and sample z for this class
    if not LATENT_BANK.exists():
        raise FileNotFoundError(f"Latent bank not found: {LATENT_BANK}")
    bank = np.load(LATENT_BANK)
    z_pool = bank[class_name]  # shape (N_c, latent_dim)
    if len(z_pool) == 0:
        raise RuntimeError(f"No latents found for class {class_name} in latent bank.")
    # random index
    idx = np.random.randint(len(z_pool))
    z = torch.from_numpy(z_pool[idx]).float().to(device)

    # 4) Prepare class embedding and decode
    y = torch.tensor([cls_idx], dtype=torch.long, device=device)
    c_emb = gen.class_emb(y)
    c_feat = gen.class_mlp(c_emb)

    with torch.no_grad():
        x_hat = gen.decode(z.unsqueeze(0), c_feat)

    img_arr = denorm_to_uint8(x_hat[0])
    out_path = OUT_DIR / f"{class_name}_V{v:+.2f}_A{a:+.2f}.png"
    Image.fromarray(img_arr).save(out_path)

    print(f"\n[OK] Saved generated heightmap to: {out_path}")

if __name__ == "__main__":
    main()
