"""Embed matching TrueType fonts in the six legacy Matplotlib PDF figures.

This repairs font resources only: page geometry, content streams, character
codes, widths, Unicode maps, plotted paths, and numerical observations stay
unchanged. It does not regenerate data or rerun experiments. The archived
Type 3 fonts must be Matplotlib's DejaVu Sans fonts with matching glyph metrics.

Usage: python fix_legacy_figure_fonts.py [--output-dir DIR]
Dependencies: matplotlib (bundled DejaVu fonts), fonttools, pypdf.
Already-repaired files are left byte-for-byte unchanged.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import matplotlib
from fontTools import subset
from fontTools.ttLib import TTFont
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject, NumberObject

HERE = Path(__file__).resolve().parent
FIGURES = (
    "roc", "nvda_probe_reads", "fork_pareto", "timeline_live",
    "sllama_latency_decomp", "noise_audit",
)
FONT_FILES = {
    "DejaVuSans": "DejaVuSans.ttf",
    "DejaVuSans-Oblique": "DejaVuSans-Oblique.ttf",
}


def repair(source, destination):
    before = source.read_bytes()
    reader = PdfReader(io.BytesIO(before))
    writer = PdfWriter(clone_from=reader)
    converted = []
    seen = set()

    def visit(resources):
        if resources is None:
            return
        resources = resources.get_object()
        for ref in resources.get("/Font", {}).values():
            font = ref.get_object()
            if id(font) in seen:
                continue
            seen.add(id(font))
            if font.get("/Subtype") != "/Type3":
                continue

            family = str(font["/BaseFont"]).split("+")[-1].lstrip("/")
            if family not in FONT_FILES:
                raise ValueError(f"Unsupported Type 3 font: {font['/BaseFont']}")
            if [float(v) for v in font["/FontMatrix"]] != [.001, 0, 0, .001, 0, 0]:
                raise ValueError("Unexpected font matrix; refusing to alter glyph scale")

            path = Path(matplotlib.get_data_path()) / "fonts/ttf" / FONT_FILES[family]
            ttf = TTFont(path, recalcTimestamp=False)
            glyphs = {str(k).lstrip("/") for k in font["/CharProcs"]}
            if not glyphs <= set(ttf.getGlyphOrder()):
                raise ValueError(f"Missing glyphs in {path.name}")

            # Verify that the embedded font is metrically the original font.
            code = None
            units = ttf["head"].unitsPerEm
            for item in font["/Encoding"]["/Differences"]:
                if isinstance(item, int):
                    code = item
                    continue
                glyph = str(item).lstrip("/")
                expected = ttf["hmtx"][glyph][0] * 1000 / units
                recorded = float(font["/Widths"][code - font["/FirstChar"]])
                if abs(expected - recorded) > 1:
                    raise ValueError(f"Glyph width mismatch: {family}/{glyph}")
                code += 1

            options = subset.Options()
            options.glyph_names = True
            options.notdef_outline = True
            options.drop_tables += ["FFTM"]
            sub = subset.Subsetter(options=options)
            sub.populate(glyphs=glyphs)
            sub.subset(ttf)
            program = io.BytesIO()
            ttf.save(program)
            ttf.close()

            stream = DecodedStreamObject()
            stream.set_data(program.getvalue())
            stream[NameObject("/Length1")] = NumberObject(len(program.getvalue()))
            descriptor = font["/FontDescriptor"]
            descriptor[NameObject("/FontFile2")] = writer._add_object(stream.flate_encode())
            font[NameObject("/Subtype")] = NameObject("/TrueType")
            for key in ("/CharProcs", "/FontMatrix", "/FontBBox", "/Resources", "/Name"):
                font.pop(key, None)
            converted.append({"font": family, "glyphs": len(glyphs),
                              "source_font_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

        for ref in resources.get("/XObject", {}).values():
            visit(ref.get_object().get("/Resources"))

    for page in writer.pages:
        visit(page.get("/Resources"))
    if not converted:
        if destination != source:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(before)
        return {"figure": source.name, "converted": []}

    result = io.BytesIO()
    writer.write(result)
    after = result.getvalue()
    check = PdfReader(io.BytesIO(after))
    assert len(reader.pages) == len(check.pages)
    for old, new in zip(reader.pages, check.pages):
        # pypdf serializes coordinates with eight significant decimal digits.
        assert all(abs(float(a) - float(b)) < .0001
                   for a, b in zip(old.mediabox, new.mediabox))
        assert old.get_contents().get_data() == new.get_contents().get_data()
        assert old.extract_text() == new.extract_text()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pdf.tmp")
    temporary.write_bytes(after)
    temporary.replace(destination)
    return {"figure": source.name, "converted": converted,
            "before_sha256": hashlib.sha256(before).hexdigest(),
            "after_sha256": hashlib.sha256(after).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    results = [repair(HERE / "figures" / f"{name}.pdf",
                      args.output_dir / f"{name}.pdf") for name in FIGURES]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
