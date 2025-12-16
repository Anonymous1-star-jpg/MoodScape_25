import numpy as np
from PIL import Image
import torch
import math

# ------------------------------------------------------------------
# Class names used by model,latent bank
# ------------------------------------------------------------------
CLASS_NAMES = [
    "Mood_Mountain_128",
    "Mood_Deseart_128",
    "Mood_Coastal_128",
    "Mood_Plateau_128",
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# ------------------------------------------------------------------
# Valence-Arousal Terrain mapping ( table)
# ------------------------------------------------------------------
# High-level terrain labels and their codes + (V, A) centres
TERRAINS_VA = {
    "Mountain": {"code": "00", "V": 0.50, "A": 0.60},   # Mountainous
    "Desert":   {"code": "01", "V": -0.75, "A": -0.21},
    "Coastal":  {"code": "10", "V": 0.80, "A": -0.55},
    "Plateau":  {"code": "11", "V": 0.10, "A": -0.30},
}

# Map from high-level terrain label -> internal class name in model
TERRAIN_TO_CLASSNAME = {
    "Mountain": "Mood_Mountain_128",
    "Desert":   "Mood_Deseart_128",
    "Coastal":  "Mood_Coastal_128",
    "Plateau":  "Mood_Plateau_128",
}

def denorm_to_uint8(x):
    """Convert [-1,1] tensor to uint8 [0,255] image."""
    x = x.squeeze().detach().cpu().numpy()  # [-1,1]
    v = (x + 1.0) * 0.5                     # [0,1]
    arr = np.clip(v * 255.0, 0, 255).astype(np.uint8)
    return arr

# ------------------------------------------------------------------
# Valence - Arousal helpers
# ------------------------------------------------------------------
def distances_from_input(v, a):
    """Return dict: terrain -> Euclidean distance from (v, a)."""
    dists = {}
    for name, info in TERRAINS_VA.items():
        dv = v - info["V"]
        da = a - info["A"]
        dists[name] = math.sqrt(dv * dv + da * da)
    return dists

def similarity_scores(dists):
    """
    Convert distances to similarity scores using softmax over -distance.
    Return dict: terrain -> similarity in [0, 1].
    """
    raw = {name: math.exp(-d) for name, d in dists.items()}  
    Z = sum(raw.values())
    if Z == 0:
        n = len(raw)
        return {name: 1.0 / n for name in raw}
    return {name: val / Z for name, val in raw.items()}

def pick_terrain_from_va(v, a):
    """
    Given V, A in [-1, 1], return:
      best_terrain, best_code, similarity_dict
    """
    dists = distances_from_input(v, a)
    sims = similarity_scores(dists)
    best_terrain = max(sims, key=sims.get)
    best_code = TERRAINS_VA[best_terrain]["code"]
    return best_terrain, best_code, sims
