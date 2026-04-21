import qrcode
from PIL import Image, ImageDraw
import os


def generate_qr_pattern(
    data: str,
    style: str = "square",
    output_path: str = "qr_base.png",
    box_size: int = 16,
    border: int = 2
):
    """
    Generate artistic QR codes with creative module patterns (scannable).

    Styles:
    - square      (default - classic solid)
    - wave        (wavy/undulating shapes)
    - corner      (corner-filled gradient effect)
    - glitch      (digital glitch art style)
    - organic     (soft organic blob shapes)
    - stipple     (pointillist dot technique)
    """

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=box_size,
        border=border
    )

    qr.add_data(data)
    qr.make(fit=True)

    matrix = qr.get_matrix()
    size = len(matrix)

    img_size = size * box_size
    img = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)

    for y in range(size):
        for x in range(size):
            if not matrix[y][x]:
                continue

            x1 = x * box_size
            y1 = y * box_size
            x2 = x1 + box_size
            y2 = y1 + box_size

            # 🎨 Artistic pattern variations (SAFE for scanning)
            if style == "wave":
                # Wavy undulating pattern
                draw.rectangle([x1, y1, x2, y2], fill="black")
                wave_height = box_size // 8
                for i in range(x1, x2, 2):
                    offset = int(wave_height * abs((i - x1) / box_size - 0.5))
                    draw.line([i, y1 + offset, i, y2 - offset], fill="white", width=1)

            elif style == "corner":
                # Gradient corner fill - fills from corners inward
                draw.rectangle([x1, y1, x2, y2], fill="black")
                margin = box_size // 4
                # Draw white corners to create gradient effect
                draw.polygon([x1, y1, x1 + margin, y1, x1, y1 + margin], fill="white")
                draw.polygon([x2, y1, x2 - margin, y1, x2, y1 + margin], fill="white")
                draw.polygon([x1, y2, x1 + margin, y2, x1, y2 - margin], fill="white")
                draw.polygon([x2, y2, x2 - margin, y2, x2, y2 - margin], fill="white")

            elif style == "glitch":
                # Digital glitch art effect
                draw.rectangle([x1, y1, x2, y2], fill="black")
                offset = box_size // 3
                if (x + y) % 2 == 0:
                    draw.rectangle([x1 + offset, y1, x2, y2 - offset], fill="white")
                else:
                    draw.rectangle([x1, y1 + offset, x2 - offset, y2], fill="white")

            elif style == "organic":
                # Soft organic blob shapes
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                size_offset = box_size // 2 - 2
                draw.ellipse([center_x - size_offset, center_y - size_offset, center_x + size_offset, center_y + size_offset], fill="black")

            elif style == "stipple":
                # Pointillist dot/stipple pattern
                draw.rectangle([x1, y1, x2, y2], fill="black")
                dot_size = 2
                for dy in range(y1 + 2, y2 - 2, 3):
                    for dx in range(x1 + 2, x2 - 2, 3):
                        if (dx + dy) % 5 == 0:
                            draw.ellipse([dx - dot_size, dy - dot_size, dx + dot_size, dy + dot_size], fill="white")

            else:  # square (default)
                draw.rectangle([x1, y1, x2, y2], fill="black")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    img.save(output_path)

    return img

styles = ["square", "wave", "corner", "glitch", "organic", "stipple"]

for style in styles:
    generate_qr_pattern(
        data="https://example.com",
        style=style,
        output_path=f"Qr patterns style/qr_{style}.png"
    )