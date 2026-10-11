# Isolated Wiley/CPE build. Requires TeX Live or MiKTeX with STIX2.
param([switch]$CleanOnly, [Alias('Clean')][switch]$FullClean)

$ErrorActionPreference = 'Stop'
$PaperDirectory = $PSScriptRoot
$MainFile = 'main_wiley_cpe'
$ReportFile = Join-Path $PaperDirectory 'build-output.txt'

function Remove-AuxiliaryFiles {
    # Exact local generated filenames only; no recursive deletion or wildcards.
    foreach ($extension in @('aux','bbl','blg','log','out','toc','lof','lot',
        'synctex.gz','synctex','fls','fdb_latexmk','pag')) {
        $target = Join-Path $PaperDirectory "$MainFile.$extension"
        if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Force }
    }
}

function Invoke-BuildCommand {
    param([string]$Command, [string[]]$Arguments)
    "Running $Command $Arguments" | Tee-Object -FilePath $ReportFile -Append
    & $Command @Arguments 2>&1 | Tee-Object -FilePath $ReportFile -Append
    if ($LASTEXITCODE -ne 0) { throw "$Command failed with exit code $LASTEXITCODE. See $ReportFile" }
}

Push-Location -LiteralPath $PaperDirectory
try {
    if ($CleanOnly) { Remove-AuxiliaryFiles; Write-Host 'Auxiliary files cleaned.'; exit 0 }
    foreach ($command in @('pdflatex','bibtex')) {
        if (-not (Get-Command $command -ErrorAction SilentlyContinue)) { throw "$command is not in PATH." }
    }
    if ($FullClean) { Remove-AuxiliaryFiles }
    Set-Content -LiteralPath $ReportFile -Value 'Wiley/CPE: pdfLaTeX + BibTeX'
    $latexArguments = @('-interaction=nonstopmode','-halt-on-error','-file-line-error','-recorder',"$MainFile.tex")
    Invoke-BuildCommand 'pdflatex' $latexArguments
    Invoke-BuildCommand 'bibtex' @($MainFile)
    # Always run twice after BibTeX, even when the previous log requests no rerun.
    Invoke-BuildCommand 'pdflatex' $latexArguments
    Invoke-BuildCommand 'pdflatex' $latexArguments
    $rerun = 'Rerun to get|Label\(s\) may have changed|rerunfilecheck.*Warning'
    for ($pass = 4; $pass -le 5; $pass++) {
        if (-not (Select-String -LiteralPath "$MainFile.log" -Pattern $rerun -Quiet)) { break }
        Invoke-BuildCommand 'pdflatex' $latexArguments
    }
    $fatal = 'undefined references|undefined citations|Citation .* undefined|Reference .* undefined|multiply defined|multiply-defined|LaTeX Error|Emergency stop|Fatal error|Undefined control sequence'
    if (Select-String -LiteralPath "$MainFile.log" -Pattern "$fatal|$rerun" -Quiet) {
        throw 'Final validation failed: errors, unresolved references, duplicate labels, or reruns remain.'
    }
    if (Select-String -LiteralPath "$MainFile.blg" -Pattern 'Warning--|error message' -Quiet) {
        throw 'BibTeX reported warnings or errors; inspect the .blg file.'
    }
    if (-not (Test-Path -LiteralPath "$MainFile.pdf")) { throw 'Output PDF was not created.' }
    $warnings = Select-String -LiteralPath "$MainFile.log" -Pattern 'Warning:|Overfull|Underfull'
    if ($warnings) { $warnings.Line | Tee-Object -FilePath $ReportFile -Append }
    else { 'Final log: no warnings or overfull/underfull boxes.' | Tee-Object -FilePath $ReportFile -Append }
    "Build passed. PDF: $(Join-Path $PaperDirectory "$MainFile.pdf")" | Tee-Object -FilePath $ReportFile -Append
}
catch { Write-Host $_.Exception.Message -ForegroundColor Red; exit 1 }
finally { Pop-Location }
