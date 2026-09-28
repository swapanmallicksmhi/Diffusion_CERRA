#!/usr/bin/env python3

import os
import re
import argparse
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-file", default="combined_daily_panels.png")

    parser.add_argument("--crop-left", type=int, default=0)
    parser.add_argument("--crop-right", type=int, default=0)
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--crop-bottom", type=int, default=0)

    parser.add_argument("--date-font-size", type=int, default=150)
    parser.add_argument("--title-font-size", type=int, default=70)
    parser.add_argument("--date-width", type=int, default=120)
    parser.add_argument("--date-scale", type=int, default=5)

    parser.add_argument("--colorbar-panel", default="CERRA")
    parser.add_argument("--colorbar-crop-left", type=int, default=900)
    parser.add_argument("--colorbar-crop-right", type=int, default=20)
    parser.add_argument("--colorbar-crop-top", type=int, default=100)
    parser.add_argument("--colorbar-crop-bottom", type=int, default=100)
    parser.add_argument("--colorbar-scale", type=float, default=2.5)

    return parser.parse_args()


def crop_image(img, crop_left, crop_right, crop_top, crop_bottom):
    width, height = img.size

    left = crop_left
    top = crop_top
    right = width - crop_right
    bottom = height - crop_bottom

    if right <= left or bottom <= top:
        raise ValueError(
            f"Crop values are too large for image size {width}x{height}."
        )

    return img.crop((left, top, right, bottom))


def get_dates(input_dir):
    pattern = re.compile(
        r"^(ERA|CERRA|DIFFUSION)_std_(\d{8})_120000_UTC\.png$"
    )

    dates = set()

    for fname in os.listdir(input_dir):
        match = pattern.match(fname)
        if match:
            dates.add(match.group(2))

    return sorted(dates)


def load_font(size):
    possible_fonts = [
        "DejaVuSans-Bold.ttf",
        "LiberationSans-Bold.ttf",
        "Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    ]

    for font_path in possible_fonts:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass

    return ImageFont.load_default()


def create_vertical_date_label(date_label, font, scale_factor=5):
    tmp_img = Image.new("RGBA", (3000, 1000), (255, 255, 255, 0))
    tmp_draw = ImageDraw.Draw(tmp_img)

    bbox = tmp_draw.textbbox((0, 0), date_label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    tmp_img = Image.new(
        "RGBA",
        (text_width + 100, text_height + 100),
        (255, 255, 255, 0),
    )

    tmp_draw = ImageDraw.Draw(tmp_img)

    tmp_draw.text(
        (50, 50),
        date_label,
        fill=(0, 70, 180, 255),
        font=font,
    )

    rotated = tmp_img.rotate(90, expand=True)

    if scale_factor > 1:
        rotated = rotated.resize(
            (
                rotated.size[0] * scale_factor,
                rotated.size[1] * scale_factor,
            ),
            Image.Resampling.LANCZOS,
        )

    return rotated


def main():
    args = parse_args()

    #panel_names = ["ERA5-EDA", "CERRA-EDA", "CONTROL"]
    panel_names = ["ERA", "CERRA", "DIFFUSION"]
    panel_titles = ["ERA5-EDA", "CERRA-EDA", "Control-Exp"]
    dates = get_dates(args.input_dir)

    if len(dates) == 0:
        raise ValueError("No matching image files found.")

    font_title = load_font(args.title_font_size)
    font_date = load_font(args.date_font_size)

    first_img_path = os.path.join(
        args.input_dir,
        f"{panel_names[0]}_std_{dates[0]}_120000_UTC.png",
    )

    first_img = Image.open(first_img_path).convert("RGB")
    first_img = crop_image(
        first_img,
        args.crop_left,
        args.crop_right,
        args.crop_top,
        args.crop_bottom,
    )

    panel_width, panel_height = first_img.size

    title_height = args.title_font_size + 70
    date_width = args.date_width
    pad = 15

    total_width = (
        date_width
        + len(panel_names) * panel_width
        + (len(panel_names) + 1) * pad
    )

    total_height = (
        title_height
        + len(dates) * panel_height
        + (len(dates) + 1) * pad
    )

    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)

    for i, panel in enumerate(panel_names):
        x = date_width + pad + i * (panel_width + pad)
        draw.text( (x + panel_width // 2, 30), panel_titles[i], fill="black", font=font_title, anchor="ma",)
    #

    for row_idx, date in enumerate(dates):
        y = title_height + pad + row_idx * (panel_height + pad)

        date_label = f"{date[:4]}-{date[4:6]}-{date[6:8]}"

        date_img = create_vertical_date_label(
            date_label,
            font_date,
            scale_factor=args.date_scale,
        )

        max_date_height = int(panel_height * 0.95)

        if date_img.size[1] > max_date_height:
            scale = max_date_height / date_img.size[1]

            date_img = date_img.resize(
                (
                    int(date_img.size[0] * scale),
                    int(date_img.size[1] * scale),
                ),
                Image.Resampling.LANCZOS,
            )

        if date_img.size[0] > date_width:
            scale = date_width / date_img.size[0]

            date_img = date_img.resize(
                (
                    int(date_img.size[0] * scale),
                    int(date_img.size[1] * scale),
                ),
                Image.Resampling.LANCZOS,
            )

        date_x = date_width - date_img.size[0] - 5
        date_y = y + (panel_height - date_img.size[1]) // 2

        canvas.paste(date_img, (date_x, date_y), date_img)

        for col_idx, panel in enumerate(panel_names):
            fname = f"{panel}_std_{date}_120000_UTC.png"
            fpath = os.path.join(args.input_dir, fname)

            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing file: {fpath}")

            img = Image.open(fpath).convert("RGB")

            img = crop_image(
                img,
                args.crop_left,
                args.crop_right,
                args.crop_top,
                args.crop_bottom,
            )

            img = img.resize(
                (panel_width, panel_height),
                Image.Resampling.LANCZOS,
            )

            x = date_width + pad + col_idx * (panel_width + pad)
            canvas.paste(img, (x, y))

    colorbar_file = os.path.join(
        args.input_dir,
        f"{args.colorbar_panel}_std_{dates[0]}_120000_UTC.png",
    )

    if not os.path.exists(colorbar_file):
        raise FileNotFoundError(f"Missing colorbar source file: {colorbar_file}")

    colorbar_img = Image.open(colorbar_file).convert("RGB")

    colorbar_img = crop_image(
        colorbar_img,
        args.colorbar_crop_left,
        args.colorbar_crop_right,
        args.colorbar_crop_top,
        args.colorbar_crop_bottom,
    )

    colorbar_img = colorbar_img.rotate(270, expand=True)

    colorbar_img = colorbar_img.resize(
        (
            int(colorbar_img.size[0] * args.colorbar_scale),
            int(colorbar_img.size[1] * args.colorbar_scale),
        ),
        Image.Resampling.LANCZOS,
    )

    output_dir = os.path.dirname(args.output_file)

    if output_dir == "":
        output_dir = "."

    os.makedirs(output_dir, exist_ok=True)

    debug_colorbar_file = os.path.join(
        output_dir,
        "DEBUG_colorbar_crop_rotated_scaled.png",
    )

    colorbar_img.save(debug_colorbar_file)

    final_height = total_height + colorbar_img.size[1] + 40

    final_canvas = Image.new("RGB", (total_width, final_height), "white")
    final_canvas.paste(canvas, (0, 0))

    colorbar_x = (total_width - colorbar_img.size[0]) // 2
    colorbar_y = total_height + 20

    final_canvas.paste(colorbar_img, (colorbar_x, colorbar_y))

    final_canvas.save(args.output_file, dpi=(100, 100))

    print(f"Saved cropped rotated scaled colorbar: {debug_colorbar_file}")
    print(f"Colorbar size: {colorbar_img.size}")
    print(f"Saved final image: {args.output_file}")


if __name__ == "__main__":
    main()
