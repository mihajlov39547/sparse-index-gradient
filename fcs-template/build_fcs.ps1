param(
    [switch]$Clean,
    [switch]$FullClean
)

$ErrorActionPreference = "Stop"
$MainFile = "manuscript"
$TexFile = "$MainFile.tex"
$AuxiliaryExtensions = @(
    "aux", "bbl", "blg", "log", "out", "toc", "lof", "lot",
    "fls", "fdb_latexmk", "synctex", "synctex.gz"
)

function Remove-AuxiliaryFiles {
    foreach ($extension in $AuxiliaryExtensions) {
        Get-ChildItem -Path . -Filter "$MainFile.$extension" -ErrorAction SilentlyContinue |
            Remove-Item -Force
    }
}

function Invoke-Tool {
    param(
        [string]$Name,
        [string]$Command,
        [string[]]$Arguments,
        [switch]$AllowNonZeroWithPdf
    )

    Write-Host "Running $Name..." -ForegroundColor Cyan
    & $Command @Arguments
    if (($LASTEXITCODE -ne 0) -and (-not ($AllowNonZeroWithPdf -and (Test-Path "$MainFile.pdf")))) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path $TexFile)) {
    throw "Run this script from fcs-template; $TexFile was not found."
}

if ($Clean -or $FullClean) {
    Remove-AuxiliaryFiles
}

if ($Clean) {
    exit 0
}

if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    throw "pdflatex was not found in PATH."
}
if (-not (Get-Command bibtex -ErrorAction SilentlyContinue)) {
    throw "bibtex was not found in PATH."
}

$passCount = 0
Invoke-Tool -Name "pdfLaTeX pass 1" -Command "pdflatex" -Arguments @(
    "-interaction=nonstopmode", "-file-line-error", $TexFile
) -AllowNonZeroWithPdf
$passCount++

Invoke-Tool -Name "BibTeX" -Command "bibtex" -Arguments @($MainFile)

for ($pass = 2; $pass -le 4; $pass++) {
    Invoke-Tool -Name "pdfLaTeX pass $pass" -Command "pdflatex" -Arguments @(
        "-interaction=nonstopmode", "-file-line-error", $TexFile
    ) -AllowNonZeroWithPdf
    $passCount++
}

if (-not (Test-Path "$MainFile.pdf")) {
    throw "$MainFile.pdf was not created."
}

$logText = Get-Content -LiteralPath "$MainFile.log" -Raw
$fatalPatterns = @(
    "! LaTeX Error:", "Emergency stop", "Fatal error", "Undefined control sequence",
    "Citation .* undefined", "Reference .* undefined", "There were undefined citations",
    "There were undefined references"
)
foreach ($pattern in $fatalPatterns) {
    if ($logText -match $pattern) {
        throw "Final LaTeX log contains: $pattern"
    }
}

Write-Host "FCS manuscript build passed after $passCount LaTeX pass(es)." -ForegroundColor Green
Write-Host "PDF output: $((Get-Item "$MainFile.pdf").FullName)" -ForegroundColor Green

if ($FullClean) {
    Remove-AuxiliaryFiles
}

exit 0
