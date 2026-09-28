#!/usr/bin/env python3

import os
import argparse
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop difference images from experiments and combine into one panel."
    )

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-file", required=True)

    parser.add_argument(
        "--experiments",
        default="CNT-EXP,EXP-1,EXP-2,EXP-3,EXP-4,EXP-5",
    )

    parser.add_argument(
        "--dates",
        required=True,
        help="Comma-separated dates in YYYYMMDD format",
    )

    parser.add_argument("--variable", default="t2m_cerra_mean")

    parser.add_argument("--crop-left", type=int, default=None)
    parser.add_argument("--crop-upper", type=int, default=None)
    parser.add_argument("--crop-right", type=int, default=None)
    parser.add_argument("--crop-lower", type=int, default=None)

    parser.add_argument("--no-crop", action="store_true")

    parser.add_argument("--padding", type=int, default=30)
    parser.add_argument("--title-height", type=int, default=60)
    parser.add_argument("--label-width", type=int, default=120)

    return parser.parse_args()


def find_image(input_dir, exp_name, variable, date):
    time_str = f"{date}_1200"
    filename = f"DIFF_MEAN_CERRA_minus_REGRIDDED_{exp_name}_{variable}_{time_str}.png"
    #filename = f"DIFF_STD_CERRA_minus_REGRIDDED_{exp_name}_{variable}_{time_str}.png"
    image_path = os.path.join(input_dir, exp_name, filename)
    print(image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Missing image:\n{image_path}")

    return image_path


def load_font(size):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)

    return ImageFont.load_default()


def check_crop_box(img, crop_box, image_path):
    w, h = img.size
    left, upper, right, lower = crop_box

    print(f"Image: {image_path}")
    print(f"Image size: width={w}, height={h}")
    print(f"Crop box: left={left}, upper={upper}, right={right}, lower={lower}")

    if left < 0 or upper < 0:
        raise ValueError("Crop left/upper cannot be negative.")

    if right > w or lower > h:
        raise ValueError(
            f"Crop box is outside the image size.\n"
            f"Image size: width={w}, height={h}\n"
            f"Crop box: {crop_box}"
        )

    if right <= left or lower <= upper:
        raise ValueError(
            f"Invalid crop box. right must be > left and lower must be > upper.\n"
            f"Crop box: {crop_box}"
        )


def main():
    args = parse_args()

    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]

    use_crop = not args.no_crop

    if use_crop:
        crop_values = [
            args.crop_left,
            args.crop_upper,
            args.crop_right,
            args.crop_lower,
        ]

        if any(v is None for v in crop_values):
            raise ValueError(
                "You must give --crop-left --crop-upper --crop-right --crop-lower, "
                "or use --no-crop."
            )

        crop_box = (
            args.crop_left,
            args.crop_upper,
            args.crop_right,
            args.crop_lower,
        )
    else:
        crop_box = None

    cropped_images = {}

    for exp_name in experiments:
        cropped_images[exp_name] = {}

        for date in dates:
            image_path = find_image(
                args.input_dir,
                exp_name,
                args.variable,
                date,
            )

            img = Image.open(image_path).convert("RGB")

            if use_crop:
                check_crop_box(img, crop_box, image_path)
                img = img.crop(crop_box)

            cropped_images[exp_name][date] = img

    sample_img = cropped_images[experiments[0]][dates[0]]
    panel_w, panel_h = sample_img.size

    n_rows = len(experiments)
    n_cols = len(dates)

    canvas_w = (
        args.label_width
        + n_cols * panel_w
        + (n_cols + 1) * args.padding
    )

    canvas_h = (
        args.title_height
        + n_rows * panel_h
        + (n_rows + 1) * args.padding
    )

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    font_title = load_font(28)
    font_label = load_font(26)

    for col, date in enumerate(dates):
        x = args.label_width + args.padding + col * (panel_w + args.padding)
        y = 15

        date_label = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        draw.text((x, y), date_label, fill="black", font=font_title)

    for row, exp_name in enumerate(experiments):
        x_label = 10
        y_label = (
            args.title_height
            + args.padding
            + row * (panel_h + args.padding)
            + panel_h // 2
            - 15
        )

        draw.text((x_label, y_label), exp_name, fill="black", font=font_label)

        for col, date in enumerate(dates):
            x = args.label_width + args.padding + col * (panel_w + args.padding)
            y = args.title_height + args.padding + row * (panel_h + args.padding)

            canvas.paste(cropped_images[exp_name][date], (x, y))

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    canvas.save(args.output_file, dpi=(150, 150))

    print(f"Saved combined image:\n{args.output_file}")


if __name__ == "__main__":
    main()
