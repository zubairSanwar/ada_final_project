
# Takes an xray PNG and saves an augmented version simulating a different machine/protocol.

import sys
import random
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def augment_xray(input_path: str, output_path: str):
    
    #grey scale
    img = Image.open(input_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0

    #adjust resolution
    scale = random.uniform(0.75, 1.0)
    if scale < 1.0:
        h, w = arr.shape
        small = Image.fromarray((arr * 255).astype(np.uint8)).resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        arr = np.array(small.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0

    # blur
    blur = random.uniform(0.5, 2.5)
    arr = gaussian_filter(arr, sigma=blur)

    # noise
    noise_std = random.uniform(0.01, 0.08)
    arr = arr + np.random.normal(0, noise_std, arr.shape).astype(np.float32)
    arr = np.clip(arr, 0, 1)

    # adjust contrast
    gamma = random.uniform(0.75, 1.3)
    arr = np.power(np.clip(arr, 1e-6, 1.0), gamma)

    lo = np.percentile(arr, random.uniform(1, 5))
    hi = np.percentile(arr, random.uniform(95, 99))
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)

    Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(output_path)
    print(f"Save image -> {output_path}")
    # print(f"scale={scale:.2f}, blur={blur:.2f}, noise={noise_std:.3f}, gamma={gamma:.2f}")


if __name__ == "__main__":

    # 2 args -> input image and output image (path/name)

    if len(sys.argv) != 3:
        print("Usage: python xray_augment.py input.png output.png")
        sys.exit(1)
    augment_xray(sys.argv[1], sys.argv[2])