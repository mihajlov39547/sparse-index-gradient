# Wiley / CPE manuscript

Independent manuscript for **Concurrency and Computation: Practice and Experience**.
The final PDF is **16 pages**, with the Wiley two-column body, six full-width
figures, seven full-width tables, and **19 numeric references**.

## Build and deliverables

```powershell
.\build_wiley_cpe.ps1 -Clean
```

If PowerShell execution policy requires an explicit invocation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_wiley_cpe.ps1 -Clean
```

`-Clean` is an alias for `-FullClean`: remove explicitly named auxiliary files
before building. `-CleanOnly` removes only auxiliary files. Cleanup does not
remove the PDF, sources, figures, bibliography, class, styles, or dependencies.
The script sets its working directory to its own folder and builds only this
manuscript. Logs are retained after the build. No Python dependency is needed.

- Main source: `main_wiley_cpe.tex`
- Peer-review PDF: `main_wiley_cpe.pdf`
- Build script: `build_wiley_cpe.ps1`
- Content: `sections/`, `tables/`, `references.bib`, and `figures/`
- Template: unchanged `USG.cls`, `NJDnatbib.sty`, `lettersp.sty`, and `images/`
- Local adaptations: `cpe-template-fixes.sty`, `cpe-AMA.bst`
- Unmodified AMA style retained for provenance: `WileyNJD-AMA.bst`
- Diagnostics: `build-output.txt`, `main_wiley_cpe.log`, `main_wiley_cpe.blg`
- Validation: `inspection/validation.json`; current page previews are in
  `inspection/twocolumn/` (older previews directly under `inspection/` are historical).

The engine remains **pdfLaTeX + BibTeX**, tested with TeX Live 2025. USG actively
loads the Type 1 STIX2 fonts; its fontspec/OpenType setup is commented out, so
XeLaTeX is unnecessary. A normal TeX installation with the packages required by
USG, plus `enumitem` and `xurl`, is required. The script stops on nonzero tool exit
codes, BibTeX warnings/errors, unresolved references/citations, duplicate labels,
or outstanding rerun requests, and reports the output PDF path and layout warnings.

## Two-column layout and content preservation

The former single-column output came from the explicit `onecolumn` option.
The main document now uses **`\documentclass[AMA,twocolumn]{USG}`**. `AMA` retains
the numeric reference setup; `twocolumn` is the native template layout option.
No custom geometry or manuscript-level `\twocolumn`/`multicols` workaround was
added. USG itself creates the full-width title and abstract above the two-column
body. The layout is retained through the references; the final page has only a
short reference-list continuation in its left column, which is normal two-column
pagination, not a single-column fallback.

The title, running-title metadata, author, affiliation, email, linked ORCID,
seven keywords, strengthened abstract and conclusion, and four required statements
remain present. The abstract fits comfortably at full width and was not shortened.
Only discretionary breaks were added inside long literal API names; their wording
and the 145.0% regret result are unchanged. `Reproducibility.` is a bold unnumbered
label inside Section 3.9, eliminating the former `3.9.0.1` and doubled punctuation.

All six figures use `figure*`; all seven active tables use `table*`. At readable
9-point table type, even the narrower tables exceed a single column's width.
Tables 1-3 use `tabularx`; Tables 4-7 use `tabular*{\textwidth}` with flexible
intercolumn spacing. There is no scaling of active tables. Ordinary `[t]` placement
respects the float count/area limits, preserving body text on float-heavy pages.
The final PDF has no float-only pages or split tables.

Table 1 uses the shorter GPU label **GeForce GTX 1650 Ti**, a 3.4 cm GPU column,
and a 3.6 cm OS/runtime column. Both GPU and OS/runtime fields now fit cleanly.
Its stack/role display text is `GTX1650Ti`, `Colab T4`, `Local workstation`, and
`Hosted stack`; all software/version values are unchanged. Other table rows,
all numerical results, captions, and analysis content are unchanged.

Figure 6 remains full-width at `width=\linewidth`, top of page 13, following its
introduction in Practical Implications on page 12. Its four panels, labels, and
caption remain readable. The final targeted polish leaves its asset, size,
caption, and placement unchanged; no manual spacing or forced page break was added.

The local compatibility style fixes sample-specific branding/metadata, the
hard-coded affiliation separator count, the first-page header, the running-footer
width in two-column mode, and short-caption horizontal fill. The intentional
`CPE manuscript for peer review` first-page footer is retained. No sample journal
logo, open-access claim, editor/special-issue text, placeholder title, or invented
publication dates/DOI/volume are displayed. The copied USG class is unchanged.

## References and arXiv/DLRM removal

The arXiv-only **`naumov2019dlrm` entry and every citation to it were removed**
from this Wiley version. The DLRM-specific architecture sentence was removed from
the related-work paragraph; the general recommendation-model motivation remains
supported by Gupta (HPCA 2020), Acun (HPCA 2021), Neo (ISCA 2022), HugeCTR (RecSys
2022), and EMBark (RecSys 2024). No new scientific claim was introduced.

Searches covered manuscript `.tex`, `.bib`, `.bbl`, and PDF text. No arXiv-only
DLRM reference remains. `Naumov` still legitimately occurs in full BibTeX author
lists for the retained Gupta and Neo proceedings papers. The phrase "Deep Learning
Recommendation Models" remains in the legitimate Acun and Neo paper titles.
Those unrelated substring matches are intentionally retained. Historical audit
snapshots are not build inputs.

The bibliography uses **`cpe-AMA.bst`**, a local repair of Wiley's official AMA
style, with superscript numeric citations in first-citation order. More than six
authors are truncated to the first three plus "et al."; all six are retained for
six-author entries. The 19 rendered entries are unique, and their order matches
the first occurrence of citations. No duplicate PyTorch documentation references
were added. The original class distribution supplied only alphabetically sorted
Chicago styles; the missing AMA style was previously obtained from Wiley:

- [Official Wiley template page](https://onlinelibrary.wiley.com/page/journal/25780727/homepage/latexclassfile)
- [Official AMA template package](https://onlinelibrary.wiley.com/pb-assets/assets/25780727/AMA-stix_New-1659369997507.zip)

The local derivative fixes missing optional fields, omitted conference titles,
BibTeX's `others` sentinel, conference/manual punctuation, and consistent DOI/URL
output. The earlier EMBark author-list expansion was verified against
[Crossref publisher metadata](https://api.crossref.org/works/10.1145/3640457.3688111).

During two-column conversion, the DOI display was wrapped in `\nolinkurl` inside
its existing HTTPS DOI hyperlink. The final targeted polish does not further
rewrite the BST: `xurl`, `\urlstyle{same}`, and `\Urlmuskip=0mu plus 1mu` provide
low-risk line-breaking flexibility. DOI fields remain clean, without inserted
spaces or manual breaks. Reference entries are kept together, so a DOI fragment
cannot be stranded on the next page. Long DOI/URL strings still wrap within the
narrow columns, sometimes within identifiers; this is acceptable two-column
bibliography behavior. They remain readable and retain their original clickable
link targets. Exact publisher punctuation and journal-title abbreviation details
remain an editorial check; this is an AMA-derived style, not publisher certification.

## Figure readiness for separate upload

No figures were regenerated or modified. All seven files in `figures/` remain
byte-identical to the baseline assets. DPI below is **effective resolution at the
size displayed in this PDF**, not merely the PNG's embedded DPI tag.

| Figure / file in `figures/` | Format and pixels | Embedded DPI | Effective DPI / vector status | Revision-stage readiness |
| --- | --- | --- | --- | --- |
| 1: `phase3_median_ms_by_case_operation.png` | PNG, 2800 x 1000 | 200 | About 421 dpi; below 600 | Separate TIFF/EPS export likely needed |
| 2: `phase3_gpu_busy_fraction_by_case_operation.png` | PNG, 2800 x 1000 | 200 | About 421 dpi; below 600 | Separate TIFF/EPS export likely needed |
| 3: `phase3_kernel_category_share.png` | PNG, 2200 x 1200 | 200 | About 330 dpi; below 600 | Separate TIFF/EPS export likely needed |
| 4: `phase3_case_winner_comparison.png` | PNG, 2000 x 1000 | 200 | About 300 dpi; below 600 | Separate TIFF/EPS export likely needed |
| 5: `phase4_winner_counts_by_environment.pdf` | PDF | Not applicable | Vector paths/text; no raster images | EPS export may be needed if PDF is not accepted |
| 6: `winner_maps_four_panel.png` | PNG, 3760 x 2120 | 600 | About 537 dpi at full width; below 600 | Separate TIFF/EPS export may be needed |
| Unused alternate: `phase4_winner_counts_by_environment.png` | PNG, 1065 x 900 | 300 | Not included in manuscript; 300 dpi tag | Prefer the vector PDF source for a later EPS export |

Under the supplied CPE guidance, graph/drawing exports should meet at least
600 dpi, preferably 800 dpi, or use a suitable vector format. Existing images
are readable for the current peer-review PDF, but the active PNGs do not reach
600 effective dpi at their present displayed sizes. Merely changing their DPI
metadata or upsampling would not add detail. Separate native-resolution/vector
exports can be prepared if requested later. **No TIFF/EPS submission exports or
submission ZIP were generated.** Existing Wiley/ORCID EPS assets are template
artwork, not newly generated figure exports.

## Build validation, warnings, and isolation

The final clean build passed and the 16-page PDF was visually checked page by
page. There are no LaTeX/BibTeX errors, unresolved references/citations, duplicate
labels, missing figures, overfull horizontal/vertical boxes, or float-only pages.
Front matter, Reproducibility, all figures/tables, references, and the Data
Availability, Conflict of Interest, Funding, and Acknowledgments statements were
checked. The public repository URL and ORCID link are preserved.

Remaining nonfatal warnings are documented rather than suppressed:

- **`multicol`**: USG line 263 unconditionally loads `graphicx,multicol`. The
  installed package warns about the global `twocolumn` option. No manuscript
  `multicols` environment exists; native USG two-column output is visually correct.
  No risky class change was made to silence this warning.
- Legacy `etex`, `fixltx2e`, and redundant `amssymb` warnings from the template.
- USG's embedded appendix code reprocesses options and reports unused global
  `AMA,twocolumn`. Both options are active, as verified by the numeric bibliography
  and rendered two-column body.
- **15 underfull horizontal boxes**, arising from narrow-column prose and one
  bibliography passage. All affected material is present and readable; no
  overfull boxes remain and warning thresholds were not raised to hide messages.

SHA-256 comparisons confirmed all **32 baseline manuscript files** present at the
start of polishing remain unchanged. All edits are confined to this Wiley/CPE
folder. LaTeX recorder checks confirm no input is read from `paper_sparse_gradient`
or the original `Optimal-Design-layout` folder. Table rows, source content, and
figure bytes were compared, allowing only the explicitly authorized display-text,
reference-removal, and LaTeX formatting changes. Benchmark scripts and analysis
outputs were not touched.

Changed during two-column polishing: `main_wiley_cpe.tex`, eight section files
(`01_abstract` through `07_conclusion`, plus `phase4_training_proxy_text`), seven
active table files, `references.bib`, `cpe-template-fixes.sty`, `cpe-AMA.bst`,
`build_wiley_cpe.ps1`, this README, local diagnostic ignore rules, and regenerated
PDF/BBL/diagnostics. The final targeted pass additionally refined Table 1 and
URL spacing; it did not alter Figure 6, the class, or scientific results.

Before portal submission: confirm current CPE abstract/length and file-designation
requirements, review author/declaration metadata, check bibliography abbreviations
and completeness, confirm public repository contents, and arrange compliant
separate figure exports if requested. No submission, upload, or submission archive
has been performed.
