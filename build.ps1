# Compile the journal-independent paper draft with pdfLaTeX + BibTeX.

param(
    [switch]$CleanOnly,
    [switch]$FullClean
)

$RepositoryRoot = Split-Path -Parent $PSCommandPath
$PaperDirectory = Join-Path $RepositoryRoot "paper_sparse_gradient"
$MainFile = "main"
$TexFile = "$MainFile.tex"
$GeneratedPdfFile = "$MainFile.pdf"
$ReportFile = Join-Path $RepositoryRoot "build-output.txt"
$MaxPdfLaTeXPasses = 3
$ReportWriter = $null

function Write-BuildLine {
    param(
        [string]$Message,
        [ConsoleColor]$Color = [ConsoleColor]::Gray
    )

    Write-Host $Message -ForegroundColor $Color
    $script:ReportWriter.WriteLine($Message)
}

function Remove-LaTeXAuxFiles {
    $extensions = @(
        "aux", "bbl", "blg", "log", "out", "toc", "lof", "lot",
        "synctex.gz", "synctex", "fls", "fdb_latexmk"
    )

    foreach ($extension in $extensions) {
        Get-ChildItem -Path . -Filter "$MainFile.$extension" -ErrorAction SilentlyContinue |
            Remove-Item -Force
    }
}

function Assert-RequiredCommands {
    param([string[]]$Commands)

    foreach ($command in $Commands) {
        if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
            throw "$command was not found in PATH."
        }
    }
}

function Invoke-NativeCommand {
    param(
        [string]$Name,
        [string]$Command,
        [string[]]$Arguments
    )

    Write-BuildLine ""
    Write-BuildLine "Running $Name..." Cyan
    $output = & $Command @Arguments 2>&1
    $exitCode = $LASTEXITCODE

    foreach ($line in $output) {
        $text = [string]$line
        Write-Host $text
        $script:ReportWriter.WriteLine($text)
    }

    Write-BuildLine "$Name exit code: $exitCode"
    return [int]$exitCode
}

function Test-NeedsPdfLaTeXRerun {
    param([string]$LogPath)

    if (-not (Test-Path $LogPath)) {
        return $false
    }

    return [bool](Select-String -LiteralPath $LogPath -Pattern @(
        'Rerun to get',
        'Label\(s\) may have changed',
        'There were undefined references'
    ) -Quiet)
}

function Test-FinalLaTeXLog {
    param([string]$LogPath)

    $fatalPatterns = @(
        '!\s+LaTeX Error:',
        'Emergency stop',
        'Fatal error',
        'Missing \$ inserted',
        'Extra \}, or forgotten \$',
        'Missing \} inserted',
        'Missing delimiter',
        'Undefined control sequence',
        'Citation .* undefined',
        'Reference .* undefined',
        'There were undefined references',
        'multiply defined',
        'multiply-defined labels'
    )

    foreach ($pattern in $fatalPatterns) {
        if (Select-String -LiteralPath $LogPath -Pattern $pattern -Quiet) {
            return $false
        }
    }

    return $true
}

function Show-LayoutWarnings {
    param([string]$LogPath)

    $warnings = Select-String -LiteralPath $LogPath -Pattern @('Overfull \\hbox', 'Overfull \\vbox')
    if ($null -eq $warnings) {
        Write-BuildLine "Layout warnings: none"
        return
    }

    Write-BuildLine "Layout warnings:" Yellow
    foreach ($warning in $warnings) {
        Write-BuildLine "$($warning.Path):$($warning.LineNumber): $($warning.Line)" Yellow
    }
}

$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$ReportWriter = [System.IO.StreamWriter]::new($ReportFile, $false, $utf8NoBom)
$scriptExitCode = 0

try {
    if (-not (Test-Path (Join-Path $PaperDirectory $TexFile))) {
        throw "Paper source was not found: $PaperDirectory\$TexFile"
    }

    Push-Location $PaperDirectory
    try {
        if ($CleanOnly) {
            Write-BuildLine "Cleaning old auxiliary files..." Cyan
            Remove-LaTeXAuxFiles
        }
        else {
            Assert-RequiredCommands -Commands @("pdflatex", "bibtex")

            if ($FullClean) {
                Write-BuildLine "Cleaning old auxiliary files..." Cyan
                Remove-LaTeXAuxFiles
            }
            else {
                Write-BuildLine "Incremental build: keeping existing auxiliary files." Cyan
            }

            $latexExitCodes = @(
                (Invoke-NativeCommand -Name "pdfLaTeX pass 1" -Command "pdflatex" -Arguments @(
                    "-interaction=nonstopmode", "-file-line-error", $TexFile
                ))
            )

            $bibtexExit = 0
            if (Select-String -LiteralPath "$MainFile.aux" -Pattern '\\citation' -Quiet) {
                $bibtexExit = Invoke-NativeCommand -Name "BibTeX" -Command "bibtex" -Arguments @($MainFile)
            }
            else {
                Write-BuildLine "Skipping BibTeX: no citations are present yet." Cyan
            }

            if ($bibtexExit -ne 0) {
                throw "BibTeX failed with exit code $bibtexExit."
            }

            $latexPass = 2
            while ((Test-NeedsPdfLaTeXRerun -LogPath "$MainFile.log") -and ($latexPass -le $MaxPdfLaTeXPasses)) {
                $latexExitCodes += Invoke-NativeCommand -Name "pdfLaTeX pass $latexPass" -Command "pdflatex" -Arguments @(
                    "-interaction=nonstopmode", "-file-line-error", $TexFile
                )
                $latexPass++
            }

            if (Test-NeedsPdfLaTeXRerun -LogPath "$MainFile.log") {
                throw "pdfLaTeX rerun limit reached."
            }

            $latexOk = Test-FinalLaTeXLog -LogPath "$MainFile.log"
            Show-LayoutWarnings -LogPath "$MainFile.log"

            if (-not (Test-Path $GeneratedPdfFile)) {
                throw "Build finished, but $GeneratedPdfFile was not created."
            }

            if (($latexExitCodes | Where-Object { $_ -ne 0 }) -or ($bibtexExit -ne 0) -or (-not $latexOk)) {
                throw "Final LaTeX validation failed."
            }

            Write-BuildLine "Final validation passed." Green
            Write-BuildLine "PDF output: $PaperDirectory\$GeneratedPdfFile" Green

            if ($FullClean) {
                Write-BuildLine "Cleaning auxiliary files after successful build..." Cyan
                Remove-LaTeXAuxFiles
            }
        }
    }
    finally {
        Pop-Location
    }
}
catch {
    $scriptExitCode = 1
    Write-BuildLine $_.Exception.Message Red
}
finally {
    if ($null -ne $ReportWriter) {
        $ReportWriter.Flush()
        $ReportWriter.Close()
        $ReportWriter.Dispose()
    }
}

exit $scriptExitCode
