import os
import base64
import argparse

from dotenv import load_dotenv
load_dotenv()

from io import BytesIO
from openai import OpenAI
from PIL import Image, ImageFilter

# -----------------------------------------
# CONFIG
# -----------------------------------------

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing.\n"
        "Create a .env file with:\n"
        "OPENAI_API_KEY=sk-xxxxx\n"
    )

client = OpenAI(api_key=API_KEY)

PORTRAIT_FILE = "portrait_stage.png"   # raw generation
CUTOUT_FILE   = "portrait_cutout.png"  # background removed
FINAL_FILE    = "home_alone_final.png"
PORTRAIT_SIZE = "1024x1536"

# Reference sweater images (your existing ones)
REFERENCE_IMAGES = [
    "references/sweaterRef1.jpg",
    "references/sweaterRef2.jpg",
    "references/sweaterRef3.jpg",
]

# Static poster background you created
BACKGROUND_PATH = "templates/bg.jpg"


# -----------------------------------------
# HELPERS
# -----------------------------------------

def load_image_file(path: str):
    """Load the file directly for OpenAI API."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return open(path, "rb")


def save_output(b64_data: str, output_path: str):
    img_bytes = base64.b64decode(b64_data)
    img = Image.open(BytesIO(img_bytes))
    img.save(output_path)
    print(f"✔ Saved: {output_path}")


# -----------------------------------------
# STEP 1 — Generate safe portrait (OPTIONAL)
# -----------------------------------------

def generate_clean_portrait(user_image_path: str):
    print("🎨 Generating portrait with gpt-image-1...")

    image_files = [load_image_file(user_image_path)]
    for ref in REFERENCE_IMAGES:
        image_files.append(load_image_file(ref))

    SAFE_PROMPT = """
    Create a portrait of the person wearing the exact Home Alone “Le Tigre” sweater.
    Reproduce the specific sweater texture:

    - marled red thick yarn mixed with thin black yarn 
    - no cable braids
    - regular jersey sweater
    - fuzzy 1990s acrylic texture
    - loose fit
    - ribbed collar, cuffs, and hem in the same marled yarn

    Do NOT generate a cable knit pattern.
    Use the real marled pattern like the references.
    Use the provided sweater reference images for accuracy.
    

    Background:
    - deep solid blue
    - no text, no logos, no additional elements.

    Keep the person's face realistic and sharp.
    """

    result = client.images.edit(
        model="gpt-image-1",
        image=image_files,
        prompt=SAFE_PROMPT,
        size=PORTRAIT_SIZE,
    )

    b64 = result.data[0].b64_json
    save_output(b64, PORTRAIT_FILE)
    print("✅ Portrait generated.")


# -----------------------------------------
# STEP 2 — Remove blue background (LOCAL ONLY)
# -----------------------------------------

def remove_background():
    print("✂ Removing background from portrait_stage.png...")

    if not os.path.exists(PORTRAIT_FILE):
        raise RuntimeError("portrait_stage.png missing. Generate it first or remove --skip-gen flag.")

    img = Image.open(PORTRAIT_FILE).convert("RGBA")
    w, h = img.size
    pixels = img.load()

    # Sample BG color from top-center
    sample_x = w // 2
    sample_y = 10
    bg_r, bg_g, bg_b, bg_a = pixels[sample_x, sample_y]

    # Create mask: 255 = keep, 0 = remove
    mask = Image.new("L", (w, h), 0)
    mask_pixels = mask.load()

    def rgb_dist2(r, g, b, rr, gg, bb):
        dr = r - rr
        dg = g - gg
        db = b - bb
        return dr*dr + dg*dg + db*db

    # threshold controls how strict the BG detection is
    # start with 35; you can tweak to 25–50 if needed
    THR = 35
    THR2 = THR * THR

    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            d2 = rgb_dist2(r, g, b, bg_r, bg_g, bg_b)
            if d2 > THR2:   # far from BG color -> keep
                mask_pixels[x, y] = 255
            else:
                mask_pixels[x, y] = 0

    # Soften edges a little
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2))

    # Apply mask as alpha
    img.putalpha(mask)
    img.save(CUTOUT_FILE)
    print("✅ Background removed -> portrait_cutout.png")


# -----------------------------------------
# STEP 3 — Composite onto bg.jpg (LOCAL ONLY)
# -----------------------------------------

def compose_final():
    print("🧩 Compositing final poster...")

    if not os.path.exists(CUTOUT_FILE):
        raise RuntimeError("portrait_cutout.png missing. Run remove_background() first.")

    if not os.path.exists(BACKGROUND_PATH):
        raise RuntimeError("Background template missing. Expected at templates/bg.jpg")

    bg = Image.open(BACKGROUND_PATH).convert("RGBA")
    cutout = Image.open(CUTOUT_FILE).convert("RGBA")

    # Resize cutout to fit your window area nicely
    # Adjust target_w/target_h until it visually matches your design
    target_w = 950
    scale = target_w / cutout.width
    target_h = int(cutout.height * scale)
    cutout = cutout.resize((target_w, target_h), Image.LANCZOS)

    # Position (x, y) so face sits nicely in the center window.
    # Tweak y until it looks perfect with your bg.jpg
    x = (bg.width - cutout.width) // 2
    y = 330  # adjust this number to move portrait up/down

    bg.paste(cutout, (x, y), cutout)
    bg.save(FINAL_FILE)

    print(f"🎬 Final poster created: {FINAL_FILE}")


# -----------------------------------------
# CLI / RUN
# -----------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Home Alone poster generator")
    parser.add_argument(
        "--input",
        help="Path to user image for generation (used only if generation is enabled).",
    )
    parser.add_argument(
        "--skip-gen",
        action="store_true",
        help="Do NOT call OpenAI. Use existing portrait_stage.png.",
    )
    args = parser.parse_args()

    print("\n🎬 HOME ALONE POSTER GENERATOR\n")

    if not args.skip_gen:
        # Need an input image for generation
        if not args.input:
            user_img = input("👉 Enter path to your image for generation: ")
        else:
            user_img = args.input

        if not os.path.exists(user_img):
            print("❌ File not found. Exiting.")
            raise SystemExit(1)

        generate_clean_portrait(user_img)
    else:
        print("⏭ Skipping generation, using existing portrait_stage.png")

    # Local-only steps: free, no API cost
    remove_background()
    compose_final()
