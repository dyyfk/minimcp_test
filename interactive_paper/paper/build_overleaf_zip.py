#!/usr/bin/env python3
"""Package the current paper sources for Overleaf (requires Pillow and pypdf).

Only reachable TeX inputs and graphics are included. Opaque PNG plots are
embedded in PDF losslessly, preserving their pixels and native dimensions.
The source tree is never modified. Run from any directory:
  python build_overleaf_zip.py /path/to/paper_overleaf.zip
"""

import argparse
import io
import re
import struct
import zipfile
from pathlib import Path

from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject as Dict,
    EncodedStreamObject,
    NameObject as Name,
    NumberObject as Int,
)


ROOT = Path(__file__).resolve().parent


def required_files():
    """Resolve this paper's input tree and extension-free graphic references."""
    required = {Path("refs.bib"), Path("neurips_2026.sty")}
    pending = [Path("main.tex")]
    while pending:
        rel = pending.pop()
        if rel in required:
            continue
        required.add(rel)
        text = re.sub(r"(?<!\\)%[^\n]*", "", (ROOT / rel).read_text())
        for target in re.findall(r"\\(?:input|include)\s*\{([^}]+)\}", text):
            pending.append(Path(target if target.endswith(".tex") else target + ".tex"))
        for target in re.findall(r"\\includegraphics\*?(?:\[[^]]*\])?\s*\{([^}]+)\}", text):
            extensions = [""] if Path(target).suffix else [".pdf", ".png", ".jpg", ".jpeg"]
            candidates = [folder / (target + ext) for ext in extensions
                          for folder in (Path("."), Path("figures"))]
            image = next((p for p in candidates if (ROOT / p).is_file()), None)
            if image is None:
                raise FileNotFoundError(f"Unresolved graphic: {target} in {rel}")
            required.add(image)
    return sorted(required)


def lossless_pdf(path):
    with Image.open(path) as image:
        if image.mode == "RGBA":
            if image.getextrema()[3] != (255, 255):
                raise ValueError(f"Refusing to discard transparency: {path}")
        elif image.mode != "RGB":
            raise ValueError(f"Unsupported image mode {image.mode}: {path}")
        rgb = image.convert("RGB")
        width, height = rgb.size
        dpi = image.info.get("dpi", (72, 72))
        page_width, page_height = width * 72 / dpi[0], height * 72 / dpi[1]
        png_buffer = io.BytesIO()
        rgb.save(png_buffer, format="PNG", optimize=True, compress_level=9)
        png = png_buffer.getvalue()
        position, idat = 8, []
        while position < len(png):
            length = struct.unpack(">I", png[position:position + 4])[0]
            if png[position + 4:position + 8] == b"IDAT":
                idat.append(png[position + 8:position + 8 + length])
            position += length + 12
        stream = EncodedStreamObject()
        stream._data = b"".join(idat)
        stream.update({
            Name("/Type"): Name("/XObject"), Name("/Subtype"): Name("/Image"),
            Name("/Width"): Int(width), Name("/Height"): Int(height),
            Name("/ColorSpace"): Name("/DeviceRGB"), Name("/BitsPerComponent"): Int(8),
            Name("/Filter"): Name("/FlateDecode"),
            Name("/DecodeParms"): Dict({Name("/Predictor"): Int(15),
                                       Name("/Colors"): Int(3),
                                       Name("/BitsPerComponent"): Int(8),
                                       Name("/Columns"): Int(width)}),
        })
        writer = PdfWriter()
        page = writer.add_blank_page(width=page_width, height=page_height)
        page[Name("/Resources")] = Dict({
            Name("/XObject"): Dict({Name("/Im0"): writer._add_object(stream)})})
        content = DecodedStreamObject()
        content.set_data(f"q {page_width:.9f} 0 0 {page_height:.9f} 0 0 cm /Im0 Do Q\n".encode())
        page[Name("/Contents")] = writer._add_object(content)
        result = io.BytesIO()
        writer.write(result)
        result.seek(0)
        decoded = PdfReader(result).pages[0]["/Resources"]["/XObject"]["/Im0"].get_object().get_data()
        if decoded != rgb.tobytes():
            raise ValueError(f"Pixel verification failed: {path}")
        return result.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output ZIP path")
    args = parser.parse_args()
    entries, converted = {}, 0
    tex = "\n".join((ROOT / p).read_text() for p in required_files() if p.suffix == ".tex")
    for path in required_files():
        # An explicit .png reference must retain that filename.
        if path.suffix == ".png" and path.name not in tex:
            entries[str(path.with_suffix(".pdf"))] = lossless_pdf(ROOT / path)
            converted += 1
        else:
            entries[str(path)] = (ROOT / path).read_bytes()
    entries["README_OVERLEAF.md"] = (
        "# MiniMCP RTCA 2026 - lean Overleaf project\n\n"
        "Main document: main.tex. Compiler: pdfLaTeX; TeX Live 2026; Normal mode.\n\n"
        "All TeX, bibliography and style files match the Git checkout used to build\n"
        "this package. Complete paper text and appendix are retained. Unused files,\n"
        "scripts and the compiled reference PDF are omitted. Included PNG plots\n"
        "with extension-free references use lossless PDF containers: original\n"
        "pixels, no resampling, no JPEG compression. Existing PDF figures and\n"
        "font repairs are unchanged.\n\n"
        "Rebuild after source edits with build_overleaf_zip.py in the repository.\n"
        "Tectonic and pdfLaTeX can produce different pagination; compare versions\n"
        "using the same engine. The smaller package does not guarantee that every\n"
        "compile stays below Overleaf's free-plan timeout.\n"
    ).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(entries.items()):
            archive.writestr(name, data)
    print(f"Wrote {args.output}: {len(entries)} files, {converted} lossless PNG conversions, "
          f"{args.output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
