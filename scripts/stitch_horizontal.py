"""
Horizontally stitch the per-dimension prediction-comparison panels of TranAD
and EPAAD-Net into a single side-by-side figure.

Usage (from the repository root):
    python scripts/stitch_horizontal.py

Reads  docs/figures/qualitative_smd_{tranad,epaad}/stitched/dim_comparison_11-20.png
Writes docs/figures/qualitative_smd_epaad/stitched/dim_comparison_tranad_vs_epaad.png
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, 'docs', 'figures')

PANEL = 'dim_comparison_11-20.png'
SRC_TRANAD = os.path.join(FIG, 'qualitative_smd_tranad', 'stitched', PANEL)
SRC_EPAAD = os.path.join(FIG, 'qualitative_smd_epaad', 'stitched', PANEL)
DST = os.path.join(FIG, 'qualitative_smd_epaad', 'stitched',
                   'dim_comparison_tranad_vs_epaad.png')


def stitch():
    image1 = Image.open(SRC_TRANAD)
    image2 = Image.open(SRC_EPAAD)

    # make both panels the same height
    height = max(image1.height, image2.height)
    image1 = image1.resize((image1.width, height))
    image2 = image2.resize((image2.width, height))

    combined = Image.new('RGB', (image1.width + image2.width, height), color='white')
    combined.paste(image1, (0, 0))
    combined.paste(image2, (image1.width, 0))

    combined.save(DST)
    print('wrote', os.path.relpath(DST, ROOT))


if __name__ == '__main__':
    stitch()
