from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


SECTION_RE = re.compile(r"^##\s+(?P<name>.+?)\s*$")
IMAGE_RE = re.compile(r"!\[(?P<alt>.*?)\]\((?P<path>.+?)\)\s*(?P<meta>\(.*\))?")


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Prefer a Windows system font for full Unicode coverage.
    win_arial = Path(r"C:\Windows\Fonts\arial.ttf")
    if win_arial.exists():
        return ImageFont.truetype(str(win_arial), size=size)
    return ImageFont.load_default()


def _wrap_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split(" ")
        current = ""
        for w in words:
            candidate = (current + " " + w).strip()
            if draw.textlength(candidate, font=font) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = w
        if current:
            lines.append(current)
    return lines


def _new_page(size: tuple[int, int]) -> Image.Image:
    return Image.new("RGB", size, "white")


def _draw_text_block(
    pages: list[Image.Image],
    text: str,
    font: ImageFont.ImageFont,
    x: int,
    y: int,
    max_width: int,
    max_height: int,
    line_gap: int,
) -> tuple[int, int]:
    page = pages[-1]
    draw = ImageDraw.Draw(page)
    lines = _wrap_lines(draw, text, font, max_width)
    line_h = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]

    for line in lines:
        if y + line_h > max_height:
            pages.append(_new_page(page.size))
            page = pages[-1]
            draw = ImageDraw.Draw(page)
            y = 60
        draw.text((x, y), line, fill="black", font=font)
        y += line_h + line_gap
    return x, y


def _paste_image_block(
    pages: list[Image.Image],
    image: Image.Image,
    caption: str,
    font: ImageFont.ImageFont,
    x: int,
    y: int,
    max_width: int,
    max_height: int,
    line_gap: int,
) -> tuple[int, int]:
    page = pages[-1]
    draw = ImageDraw.Draw(page)
    caption_lines = _wrap_lines(draw, caption, font, max_width)
    caption_h = (font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + line_gap) * len(caption_lines)

    img = image.convert("RGB")
    scale = min(max_width / img.width, (max_height - y - caption_h) / img.height, 1.0)
    if scale <= 0:
        pages.append(_new_page(page.size))
        page = pages[-1]
        draw = ImageDraw.Draw(page)
        y = 60
        scale = min(max_width / img.width, (max_height - y - caption_h) / img.height, 1.0)

    new_size = (int(img.width * scale), int(img.height * scale))
    img_resized = img.resize(new_size, Image.LANCZOS)

    if y + new_size[1] + caption_h > max_height:
        pages.append(_new_page(page.size))
        page = pages[-1]
        draw = ImageDraw.Draw(page)
        y = 60

    page.paste(img_resized, (x, y))
    y += new_size[1] + line_gap

    for line in caption_lines:
        draw.text((x, y), line, fill="black", font=font)
        y += (font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) + line_gap
    return x, y


def parse_report(md_text: str) -> dict:
    sections: dict[str, list[str]] = {}
    current = None
    for line in md_text.splitlines():
        m = SECTION_RE.match(line)
        if m:
            current = m.group("name").strip()
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)

    return {
        "Query": "\n".join(sections.get("Query", [])).strip(),
        "Answer": "\n".join(sections.get("Answer", [])).strip(),
        "Sources": [l.strip() for l in sections.get("Sources", []) if l.strip()],
        "Images": [l.strip() for l in sections.get("Images", []) if l.strip()],
    }


def extract_images(image_lines: Iterable[str]) -> list[dict]:
    out: list[dict] = []
    for line in image_lines:
        m = IMAGE_RE.search(line)
        if not m:
            continue
        path = m.group("path")
        meta = (m.group("meta") or "").strip()
        out.append({"path": path, "meta": meta})
    return out


def md_report_to_pdf(md_path: str | Path, pdf_path: str | Path | None = None) -> str:
    md_path = Path(md_path)
    if pdf_path is None:
        pdf_path = md_path.with_suffix(".pdf")
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    text = md_path.read_text(encoding="utf-8")
    data = parse_report(text)
    images = extract_images(data["Images"])

    # A4 at ~150 DPI
    page_size = (1240, 1754)
    margin = 60
    max_width = page_size[0] - margin * 2
    max_height = page_size[1] - margin

    title_font = _load_font(28)
    header_font = _load_font(20)
    body_font = _load_font(16)

    pages: list[Image.Image] = [_new_page(page_size)]
    x, y = margin, margin

    # Title
    pages[-1].paste(_new_page(page_size), (0, 0))
    draw = ImageDraw.Draw(pages[-1])
    draw.text((x, y), "RAG Answer", fill="black", font=title_font)
    y += (title_font.getbbox("Ag")[3] - title_font.getbbox("Ag")[1]) + 18

    # Query
    draw.text((x, y), "Query", fill="black", font=header_font)
    y += (header_font.getbbox("Ag")[3] - header_font.getbbox("Ag")[1]) + 8
    _, y = _draw_text_block(pages, data["Query"], body_font, x, y, max_width, max_height, 6)
    y += 10

    # Answer
    pages.append(pages.pop())  # keep same page reference
    draw = ImageDraw.Draw(pages[-1])
    if y + 60 > max_height:
        pages.append(_new_page(page_size))
        y = margin
        draw = ImageDraw.Draw(pages[-1])
    draw.text((x, y), "Answer", fill="black", font=header_font)
    y += (header_font.getbbox("Ag")[3] - header_font.getbbox("Ag")[1]) + 8
    _, y = _draw_text_block(pages, data["Answer"], body_font, x, y, max_width, max_height, 6)
    y += 10

    # Sources
    if y + 60 > max_height:
        pages.append(_new_page(page_size))
        y = margin
    draw = ImageDraw.Draw(pages[-1])
    draw.text((x, y), "Sources", fill="black", font=header_font)
    y += (header_font.getbbox("Ag")[3] - header_font.getbbox("Ag")[1]) + 8
    src_text = "\n".join(data["Sources"])
    _, y = _draw_text_block(pages, src_text, body_font, x, y, max_width, max_height, 4)
    y += 10

    # Images
    if y + 60 > max_height:
        pages.append(_new_page(page_size))
        y = margin
    draw = ImageDraw.Draw(pages[-1])
    draw.text((x, y), "Images", fill="black", font=header_font)
    y += (header_font.getbbox("Ag")[3] - header_font.getbbox("Ag")[1]) + 8

    for img in images:
        img_path = Path(img["path"])
        if not img_path.is_absolute():
            img_path = (md_path.parent / img_path).resolve()
        caption = img.get("meta") or img_path.name
        try:
            with Image.open(img_path) as im:
                x, y = _paste_image_block(
                    pages,
                    im,
                    caption,
                    body_font,
                    x,
                    y,
                    max_width,
                    max_height,
                    6,
                )
                y += 8
        except Exception:
            # If an image fails to open, just note it.
            _, y = _draw_text_block(
                pages,
                f"[missing image] {img_path}",
                body_font,
                x,
                y,
                max_width,
                max_height,
                4,
            )

    pages[0].save(str(pdf_path), save_all=True, append_images=pages[1:])
    return str(pdf_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert a RAG report markdown into a PDF with images.")
    parser.add_argument("md_path", help="Path to the .md report")
    parser.add_argument("--out", dest="out_path", help="Output PDF path", default=None)
    args = parser.parse_args()

    pdf_path = md_report_to_pdf(args.md_path, args.out_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
