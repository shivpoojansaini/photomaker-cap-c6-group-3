from PIL import Image, ImageDraw, ImageFont

def add_watermark(image, text="© AI-Generated image by CAP-C6-Group_3", opacity=160):
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    W, H = image.size

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    font_size = max(24, W // 10)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding = font_size // 4
    pos = (W - text_w - padding, H - text_h - padding)

    draw.text(pos, text, fill=(255, 255, 255, opacity), font=font)

    return Image.alpha_composite(image, layer).convert("RGB")
