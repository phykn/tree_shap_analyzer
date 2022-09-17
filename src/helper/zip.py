import zipfile
from typing import List
from io import BytesIO
from matplotlib.figure import Figure


def convert_figs2zip(
    figs: List[Figure]
) -> BytesIO:
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
        for i, fig in enumerate(figs):
            buf = BytesIO()
            fig.savefig(buf, bbox_inches="tight")
            name = f"fig_{i:03d}.png"
            zf.writestr(name, buf.getvalue())
    return zip_buffer
