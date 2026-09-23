from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
BRAND_DIR = ROOT / "public" / "theta-assets" / "brand"
MODEL_OUTPUT = BRAND_DIR / "theta-background-extraction-model-v2.png"
CAT_CUTOUT = BRAND_DIR / "theta-peeking-cat-v1.png"
OUTPUT = BRAND_DIR / "theta-cat-wordmark-transparent-v1.png"


def main() -> None:
    source = Image.open(MODEL_OUTPUT).convert("RGB")
    pixels = np.asarray(source).astype(np.float32)
    red, green, blue = np.moveaxis(pixels, -1, 0)

    # The image model preserved the checkerboard as neutral RGB pixels.  The
    # glass wordmark is blue/cyan, so blue chroma gives a clean letter mask
    # without carrying any of the neutral squares into the final PNG.
    neutral = (red + green) * 0.5
    word_alpha = np.clip((blue - neutral - 1.5) * 9.0, 0, 255).astype(np.uint8)

    height, width = word_alpha.shape
    region = np.zeros_like(word_alpha)
    region[int(height * 0.46) :, int(width * 0.08) :] = 255
    word_alpha = np.minimum(word_alpha, region)
    word_alpha = np.asarray(
        Image.fromarray(word_alpha, "L")
        .filter(ImageFilter.MaxFilter(5))
        .filter(ImageFilter.GaussianBlur(0.8))
    )

    logo = source.convert("RGBA")
    logo.putalpha(Image.fromarray(word_alpha, "L"))

    cat = Image.open(CAT_CUTOUT).convert("RGBA")
    cat_width = 548
    cat_height = round(cat_width * cat.height / cat.width)
    cat = cat.resize((cat_width, cat_height), Image.Resampling.LANCZOS)
    logo.alpha_composite(cat, (92, 130))

    alpha_box = logo.getchannel("A").getbbox()
    if alpha_box is None:
        raise RuntimeError("The composed logo has no visible pixels")

    left, top, right, bottom = alpha_box
    padding = 28
    crop_box = (
        max(0, left - padding),
        max(0, top - padding),
        min(logo.width, right + padding),
        min(logo.height, bottom + padding),
    )
    logo.crop(crop_box).save(OUTPUT, "PNG", optimize=True)

    result = Image.open(OUTPUT)
    print(f"saved={OUTPUT}")
    print(f"mode={result.mode} size={result.size} alpha={result.getchannel('A').getextrema()}")


if __name__ == "__main__":
    main()
