"""Compose four existing winner-map PNGs with compact saved CSV annotations.

This is a figure-layout utility, not an analysis: no medians or winners are
recomputed. PNG cell colors and the existing source labels supply the content.
Requires Pillow. Run from any directory with python combine_winner_maps.py.
"""
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "results/winner_maps"
OUTPUT = BASE.parent / "paper_sparse_gradient/figures/winner_maps_four_panel.png"
PANELS = [
    ("phase2_winner_map_local.png", "(a) Phase 2 — Local GTX1650Ti"),
    ("phase2_winner_map_t4.png", "(b) Phase 2 — T4"),
    ("phase4_winner_map_local_proxy.png", "(c) Phase 4 — Local GTX1650Ti — Proxy step"),
    ("phase4_winner_map_t4_proxy.png", "(d) Phase 4 — Colab T4 — Proxy step"),
]
DISTS = ("uniform", "zipf", "hotspot")
BUCKETS = ("low_reference_pressure", "balanced_reference_pressure", "high_reference_pressure")
COLORS = {"index_add": "#0072B2", "scatter_add": "#E69F00", "embedding": "#009E73", "tie": "#D9D9D9"}
SHORT = {"index_add": "idx", "scatter_add": "scat", "embedding": "emb", "tie": "tie"}


def font(size, bold=False):
    path = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    if not path.exists():
        path = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(path), size)


def main():
    with (SOURCE / "winner_map_source.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    canvas = Image.new("RGB", (3760, 2120), "white")
    draw = ImageDraw.Draw(canvas)
    # At the paper's 6.27-inch text width, 60 px is approximately 7.2 pt.
    body, small, title = font(60), font(53), font(61, True)
    for panel, (filename, heading) in enumerate(PANELS):
        records = [r for r in rows if r["figure_file"] == filename]
        dims = sorted({int(r["dim"]) for r in records})
        buckets = [b for b in BUCKETS if any(r["ratio_bucket"] == b for r in records)]
        columns = [(d, b) for d in DISTS for b in buckets]
        lookup = {(int(r["dim"]), r["distribution"], r["ratio_bucket"]): r for r in records}
        if len(records) != len(dims) * len(columns):
            raise ValueError(f"Incomplete source panel: {filename}")
        with Image.open(SOURCE / filename) as opened:
            original = opened.convert("RGB")
        # The saved plotting utility positions its data grid in these normalized
        # bounds. Extract its corner colors, leaving original text unscaled; then
        # re-typeset verbatim categories/counts for readable full-width composition.
        grid = original.crop((round(.07*original.width), round(.30*original.height),
                              round(.99*original.width), round(.75*original.height)))
        ox, oy = (40 if panel % 2 == 0 else 1910), (165 if panel < 2 else 1080)
        draw.text((ox+915, oy), heading, font=title, fill="black", anchor="mt")
        gx, gy, gw, gh = ox+130, oy+120, 1660, (540 if panel < 2 else 400)
        cw, ch = gw/len(columns), gh/len(dims)
        draw.text((ox+45, gy-56), "dim", font=small, fill="black", anchor="mm")
        for i, dim in enumerate(dims):
            draw.text((gx-25, gy+(i+.5)*ch), str(dim), font=body, fill="black", anchor="rm")
            for j, (dist, bucket) in enumerate(columns):
                r = lookup[(dim, dist, bucket)]
                cats = r["dominant_category"].split(";")
                x0, y0 = round(gx+j*cw), round(gy+i*ch)
                x1, y1 = round(gx+(j+1)*cw), round(gy+(i+1)*ch)
                color = grid.getpixel((round((j+.08)*grid.width/len(columns)),
                                       round((i+.10)*grid.height/len(dims))))
                if len(cats) == 1:
                    expected = tuple(bytes.fromhex(COLORS[cats[0]][1:]))
                    if color != expected:
                        raise ValueError(f"PNG/source color mismatch: {filename}/{dim}/{dist}/{bucket}")
                    draw.rectangle((x0,y0,x1,y1), fill=color)
                else:
                    tile = Image.new("RGB", (x1-x0, y1-y0), "white")
                    td = ImageDraw.Draw(tile)
                    for start in range(-tile.height, tile.width, 38):
                        td.line((start, tile.height, start+tile.height, 0), fill="#D3D3D3", width=3)
                    canvas.paste(tile,(x0,y0))
                draw.rectangle((x0,y0,x1,y1), outline="white", width=4)
                ink = "white" if cats == ["index_add"] else "black"
                label = r["cell_label"]
                if len(cats) > 1:
                    # Three compact lines preserve both leaders and shared support.
                    label = "/".join(SHORT[c] for c in cats) + f"\n{r['dominant_count']}/{r['n_observations']}\neach"
                draw.multiline_text(((x0+x1)/2,(y0+y1)/2), label, font=small if len(cats)>1 else body,
                                    fill=ink, anchor="mm", align="center", spacing=8)
        for d, dist in enumerate(DISTS):
            center = gx+(d+.5)*len(buckets)*cw
            draw.text((center,gy+gh+62),dist.title(),font=body,fill="black",anchor="mm")
            for b, bucket in enumerate(buckets):
                draw.text((gx+(d*len(buckets)+b+.5)*cw,gy+gh+125),
                          {BUCKETS[0]:"Low",BUCKETS[1]:"Bal.",BUCKETS[2]:"High"}[bucket],
                          font=small,fill="black",anchor="mm")
            if d:
                draw.line((gx+d*len(buckets)*cw,gy,gx+d*len(buckets)*cw,gy+gh),fill="#555555",width=4)
    # One legend and one pressure definition replace repeated PNG furniture.
    x = 125
    for cat, color in COLORS.items():
        draw.rectangle((x,40,x+65,95),fill=color)
        draw.text((x+85,68),f"{SHORT[cat]}: {cat}",font=body,fill="black",anchor="lm")
        x += 685
    draw.text((x,68),"Hatching: equal category counts",font=small,fill="black",anchor="lm")
    draw.text((1880,1900),"Reference pressure (indices / rows): Low < 1; Bal. 1–<4; High ≥4",
              font=body,fill="black",anchor="mm")
    draw.text((1880,1980),"Cell: most frequent category and support / observations; all winners use strict >5% margin",
              font=small,fill="black",anchor="mm")
    draw.text((1880,2050),"Phase 2 has no low-pressure cells. Support counts include repeated draws or seeds.",
              font=small,fill="black",anchor="mm")
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(OUTPUT,dpi=(600,600))
    print(f"Created {OUTPUT}: {canvas.width} x {canvas.height}, 600 dpi metadata; no winner recomputation.")


if __name__ == "__main__":
    main()
