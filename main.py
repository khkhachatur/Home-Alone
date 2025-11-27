import os
import base64
import argparse
from io import BytesIO

from openai import OpenAI
from PIL import Image

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DEFAULT_ENGINE = "gpt-image-1"  # recommended for best edits
# You can also try: "dall-e-3" (text-only generation, will NOT preserve exact face)

POSTER_SIZE = "1024x1536"  # vertical movie-poster style

# --------------------------------------------------
# PROMPT TEMPLATES
# --------------------------------------------------

def build_home_alone_prompt(user_name: str | None = None) -> str:
    name_part = f"{user_name}'s face" if user_name else "the person’s face"

    return f"""
Turn {name_part} into the star of the classic 1990 movie poster "Home Alone".

Requirements:
- Keep {name_part} clearly recognizable and realistic.
- Preserve facial expression and main facial structure.
- Change the clothes to the iconic thick red knitted sweater from the reference image:
  heavy knit, slightly loose, bright red with darker red/black fibers in the threads.
- Show the character in a centered portrait composition from chest up.

Background and layout:
- Replace the background with a flat deep blue, like the original Home Alone poster.
- At the top of the image, add the movie title text:
  At very top: small white caps text: "FROM JOHN HUGHES".
  Under it, big yellow text: "HOME" and "ALONe" with a red house icon between HOME and ALONe.
- The logo must be sharp, centered, and readable.
- Add very subtle winter / Christmas mood, soft film grain.

Overall style:
- High-quality 1990 movie poster.
- Bright, clean colors, no blur or distortion.
- Photorealistic face, cinematic lighting, sharp details.
    """.strip()


def build_midjourney_prompt():
    """
    This returns a text prompt that you can paste into Midjourney in Discord.
    The script can't call Midjourney directly (no official API).
    """
    return r"""
/imagine prompt: A modern recreation of the classic "Home Alone" movie poster.

Center frame: a close-up portrait of the user, hands on cheeks in a shocked expression,
wearing a thick bright red knitted sweater with a chunky texture, similar to 1990s winter knitwear.
The sweater has a deep red color with darker dyed fibers.

Background: solid deep blue poster background.

At the top, clean movie-style typography:
small white text "FROM JOHN HUGHES" above,
below it the title "HOME" [yellow], then a red house icon, then "ALONe" [yellow].
Crisp, readable text aligned and centered like a Hollywood movie poster.

Lighting: soft studio light, slightly frontal, subtle shadows, cinematic sharpness.
Style: photorealistic 8K, movie poster, high contrast, rich colors, winter holiday mood.
--ar 2:3 --v 6
    """.strip()


# --------------------------------------------------
# IMAGE HELPERS
# --------------------------------------------------

def load_and_optionally_resize(path: str, max_side: int = 1536) -> BytesIO:
    """
    Load an image from disk, convert to RGB, and optionally downscale so
    the longest side <= max_side. Returns a BytesIO buffer (PNG).
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def save_base64_image(b64_data: str, output_path: str):
    image_bytes = base64.b64decode(b64_data)
    img = Image.open(BytesIO(image_bytes))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"✅ Saved generated poster to: {output_path}")


# --------------------------------------------------
# OPENAI CALLS
# --------------------------------------------------

def generate_home_alone_poster_openai(
    client: OpenAI,
    user_image_path: str,
    sweater_ref_path: str | None,
    engine: str,
    output_path: str,
    user_name: str | None = None,
):
    """
    Uses OpenAI Images API to turn the user into the Home Alone poster star.
    With gpt-image-1 we can pass images (user + sweater reference) and a text prompt.
    With dall-e-3 we generate only from text (no image input).
    """

    prompt = build_home_alone_prompt(user_name)

    # gpt-image-1 path: image editing with references
    if engine == "gpt-image-1":
        user_buf = load_and_optionally_resize(user_image_path)
        image_files = [user_buf]

        if sweater_ref_path:
            sweater_buf = load_and_optionally_resize(sweater_ref_path)
            image_files.append(sweater_buf)

        # Convert BytesIO objects to "file-like" objects for images.edit
        user_buf.seek(0)
        files = [user_buf]
        if sweater_ref_path:
            sweater_buf.seek(0)
            files.append(sweater_buf)

        # GPT Image edit call
        result = client.images.edit(
            model="gpt-image-1",
            image=files,
            prompt=prompt,
            size=POSTER_SIZE,
        )
        b64 = result.data[0].b64_json
        save_base64_image(b64, output_path)

    # dall-e-3: text-only generation (will not preserve the real face 1:1)
    elif engine == "dall-e-3":
        result = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1536",
            quality="hd",
            style="natural",
        )
        # DALL-E 3 returns URL or base64 depending on client config.
        # Using base64 version for full control:
        data = result.data[0]

        if hasattr(data, "b64_json") and data.b64_json:
            b64 = data.b64_json
            save_base64_image(b64, output_path)
        else:
            # Fallback: URL download
            url = data.url
            print("Got URL from DALL·E 3:", url)
            print("Download this URL manually or add requests.get(...) code here.")
    else:
        raise ValueError("Unsupported engine. Use 'gpt-image-1' or 'dall-e-3'.")


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a Home Alone movie poster with the user as the star."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the user image (face visible).",
    )
    parser.add_argument(
        "--sweater",
        required=False,
        help="Path to the red sweater reference image (optional but recommended for gpt-image-1).",
    )
    parser.add_argument(
        "--output",
        default="output/home_alone_poster.png",
        help="Where to save the final poster image.",
    )
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        choices=["gpt-image-1", "dall-e-3"],
        help="Which OpenAI image model to use.",
    )
    parser.add_argument(
        "--user-name",
        default=None,
        help="Optional name used only inside the prompt (for personalization).",
    )
    parser.add_argument(
        "--show-midjourney-prompt",
        action="store_true",
        help="Print a ready Midjourney prompt you can paste into Discord.",
    )

    args = parser.parse_args()

    if args.show_midjourney_prompt:
        print("------ Midjourney Prompt (copy & paste into Discord) ------\n")
        print(build_midjourney_prompt())
        print("\n-----------------------------------------------------------\n")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    generate_home_alone_poster_openai(
        client=client,
        user_image_path=args.input,
        sweater_ref_path=args.sweater,
        engine=args.engine,
        output_path=args.output,
        user_name=args.user_name,
    )


if __name__ == "__main__":
    main()
