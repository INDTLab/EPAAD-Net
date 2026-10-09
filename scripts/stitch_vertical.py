"""
Vertically stitch the input and reconstructed-output panels of a single
dimension into one figure.

Usage (from the repository root):
    python scripts/stitch_vertical.py [dim]

Writes docs/figures/qualitative_smd_epaad/stitched/d<dim>.png
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
from PIL import Image, ImageOps, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'docs', 'figures', 'qualitative_smd_epaad')
DST = os.path.join(SRC, 'stitched')

BORDER = 1


def stitch(dim):
    image1 = Image.open(os.path.join(SRC, 'input_%02d.png' % dim))
    image2 = Image.open(os.path.join(SRC, 'output_%02d.png' % dim))

    # make both panels the same width
    width = max(image1.width, image2.width)
    image1 = image1.resize((width, image1.height))
    image2 = image2.resize((width, image2.height))

    # thin black separators
    image1 = ImageOps.expand(image1, border=BORDER, fill='black')
    image2 = ImageOps.expand(image2, border=BORDER, fill='black')

    combined_height = image1.height + image2.height + BORDER
    combined = Image.new('RGB', (width, combined_height), color='black')
    combined.paste(image1, (0, 0))
    combined.paste(image2, (0, image1.height + BORDER))

    draw = ImageDraw.Draw(combined)
    draw.line([(0, image1.height), (width, image1.height)], fill='black', width=BORDER)
    draw.rectangle([(0, combined_height - BORDER), (width, combined_height)], fill='black')

    os.makedirs(DST, exist_ok=True)
    out = os.path.join(DST, 'd%02d.png' % dim)
    combined.save(out)
    print('wrote', os.path.relpath(out, ROOT))


def main():
    dim = int(_sys.argv[1]) if len(_sys.argv) > 1 else 20
    stitch(dim)


if __name__ == '__main__':
    main()
