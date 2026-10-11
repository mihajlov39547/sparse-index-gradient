# Wiley / CPE manuscript

Independent manuscript for **Concurrency and Computation: Practice and Experience**.
The original `paper_sparse_gradient` and `WileyDesign/Optimal-Design-layout`
directories were left unchanged. SHA-256 comparisons verified all 78 files that
were present in those directories before this work.

## Deliverables and build

- Main source: `main_wiley_cpe.tex`
- Peer-review PDF: `main_wiley_cpe.pdf` (19 pages)
- Build script: `build_wiley_cpe.ps1`
- Copied/adapted content: `sections/`, `tables/`, `figures/`, `references.bib`
- Template dependencies: unchanged `USG.cls`, `NJDnatbib.sty`, and `lettersp.sty`
  (the latter copied from `LETTERSP.STY` with a portable lowercase filename).
- Template artwork: Wiley and ORCID EPS/PDF assets in `images/`.
- Local adaptations: `cpe-template-fixes.sty` and `cpe-AMA.bst`.
- Reference style provenance: original, unmodified `WileyNJD-AMA.bst` retained.
- Diagnostics: `build-output.txt`, `main_wiley_cpe.log`, `main_wiley_cpe.blg`,
  and `inspection/validation.json`. Page previews are under `inspection/`.

From this directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_wiley_cpe.ps1 -FullClean
```

The script also works when invoked by full path from another directory: it sets
its working directory to its own folder. Without switches it rebuilds using
existing auxiliary files. `-FullClean` removes only explicitly named auxiliary
files before building; `-CleanOnly` performs only that cleanup. Source files,
figures, styles, bibliography, and PDF are never deleted by cleanup. Logs remain
after a successful full build so warnings can be inspected.

The build uses **pdfLaTeX + BibTeX**, followed by at least two pdfLaTeX passes and
additional passes when needed. It stops on nonzero tool exit codes, BibTeX
warnings, unresolved references/citations, duplicate labels, or outstanding
rerun requests. It reports the PDF path and remaining layout warnings.

A normal TeX Live/MiKTeX installation with the packages loaded by USG is required.
Tested with TeX Live 2025. The downloaded class actively loads the Type 1 `stix2`
package; its OpenType/fontspec configuration is commented out. Therefore neither
XeLaTeX/LuaLaTeX nor the unused downloaded OTF font directory is required. The
LaTeX recorder confirms that no build input comes from the baseline manuscript
or the original template directory. Python is not required to build the paper.

## Content and layout

The requested title, running-title metadata, author, affiliation, email, linked
ORCID, seven keywords, abstract, conclusion, and four end statements are present.
The baseline abstract/conclusion already matched the requested wording; only the
abstract wrapper was changed. Wiley's `\bmsubsection*` formats the end statements.
The main body follows Introduction, Background and Related Work, Methodology,
Results and Analysis, General Discussion, and Conclusion.

The class's single-column layout preserves readable full-width tables and
multi-panel figures. Seven tables and six figures appear inline with Wiley
captions. All numerical table contents and figure/caption semantics are retained.
The existing wrapped columns are retained, table text is explicitly 9 pt instead
of the class's forced 8 pt, and no active table is scaled. All copied figure
files are byte-identical to the originals: no regeneration, cropping, distortion,
or rasterization of the vector figure was performed. Separate figure assets are
ready in `figures/`. The copied content directories also retain the baseline's
unused diagnostic tables and alternate PNG of the vector proxy figure; only the
six figures/seven tables referenced by the main source enter the PDF.

The unmodified class has several sample-specific behaviors. The local style
removes its hard-coded Allergy logo and open-access badge, corrects its
hard-coded four-affiliation separator for the actual address count, and replaces
the first-page publication footer with a peer-review footer. It omits invented
dates/DOI/volume/year, removes the empty-year footer comma, and corrects the
zero-height first-page header overflow. Wiley typography, title/author/abstract
macros, section headings, and caption formatting remain in use. All compatibility
patches fail clearly if a future class version no longer matches.

## Bibliography

The downloaded Chicago styles were inspected; they contain alphabetical sorting
and do not implement the requested AMA bibliography. USG supports `AMA` but its
required `WileyNJD-AMA.bst` was absent from the download and local TeX installation.
The original AMA style was obtained from this official Wiley distribution:

- [Wiley template download page](https://onlinelibrary.wiley.com/page/journal/25780727/homepage/latexclassfile)
- [AMA-stix_New package](https://onlinelibrary.wiley.com/pb-assets/assets/25780727/AMA-stix_New-1659369997507.zip)

Only the bibliography style was extracted from that package; the manuscript still
uses the requested downloaded USG class. No submission ZIP was created.

The older AMA style had errors on missing optional organization/address fields,
omitted conference titles without editors, mishandled BibTeX's `others` sentinel,
and did not consistently output DOI/URL fields. `cpe-AMA.bst` is a clearly named
local derivative repairing these issues and related conference/manual punctuation.
The main source redirects the class's bibliography-style selection to this file.

The final PDF has **20 superscript numeric references**, listed in first-citation
order. More than six authors are truncated to the first three followed by
“et al.”; six-author entries retain all six. The cleaned baseline bibliography
was the source, and no duplicate documentation entries were added. The only
bibliographic data change is expansion of `liu2024embark` from one author plus
`others` to all nine authors, so the required first-three rule can work. Names
were verified against [Crossref's publisher-deposited DOI metadata](https://api.crossref.org/works/10.1145/3640457.3688111).

This meets the requested numeric ordering and author-count rules. It is a locally
repaired Wiley AMA style, not a claim of publisher certification of every AMA
punctuation/abbreviation detail. Journal/conference abbreviations and complete
reference metadata remain part of the author's final editorial check.

## Validation and remaining checks

The clean build passed. All pages were rendered and inspected, including the
title/author/abstract/keywords, every figure/table, end statements, and references.
No figure-only pages, clipping, missing assets, duplicate labels, unresolved
citations/references, wrong build paths, or overfull boxes remain. The ORCID link
and citation-to-bibliography ordering were checked programmatically. Table text
was compared with the baseline after removing only font declarations; figures
were compared byte-for-byte.

Remaining nonfatal log warnings:

- Legacy `etex`, `fixltx2e`, and redundant `amssymb` package warnings.
- USG reports unused global `AMA,onecolumn` options after its embedded appendix
  code processes options again. The earlier AMA branch and single-column layout
  are active, as verified in the auxiliary file and PDF.
- Five underfull horizontal boxes: two short table captions, two prose passages,
  and one bibliography passage. These are spacing warnings; inspected content is
  complete and legible. No warnings have been hidden.

Before submission, the author should review the final wording and metadata,
confirm the current CPE abstract/length and reference-detail requirements, inspect
the smallest figure legends at intended viewing/print size, confirm the public
repository contains the intended supporting artifacts, and check the submission
portal's requested file designations and separate-figure formats. The supplied
long abstract was preserved rather than shortened without authorization. Standard
PNG plot assets can be converted/exported to a requested revision format later.
No submission, upload, or final submission archive was performed.
