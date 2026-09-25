"""Optionally recreate the synthetic scene with Pillow without downloading images."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    image = Image.new("RGB", (640, 400), "#f3f0e8")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)
    draw.text((20, 20), "Synthetic scene: cup left of laptop", fill="#333333", font=font)
    draw.line((40, 320, 600, 320), fill="#b4a897", width=3)
    # The bounding box enclosing the cup body and handle is [100, 170, 200, 290].
    draw.ellipse((158, 190, 200, 250), fill="#438fb2")
    draw.ellipse((171, 202, 189, 238), fill="#f3f0e8")
    draw.rounded_rectangle((100, 170, 177, 290), radius=12, fill="#438fb2")
    draw.ellipse((100, 170, 177, 188), fill="#204b5e")
    draw.text((117, 330), "Cup", fill="#333333", font=font)
    draw.rounded_rectangle((330, 115, 535, 270), radius=9, fill="#454d5a")
    draw.rectangle((342, 127, 523, 255), fill="#9ec5d4")
    draw.polygon([(330, 270), (535, 270), (565, 300), (300, 300)], fill="#7b8491")
    draw.text((385, 330), "Laptop", fill="#333333", font=font)
    image.save(Path(__file__).resolve().parent / "scene.png")


if __name__ == "__main__":
    main()
