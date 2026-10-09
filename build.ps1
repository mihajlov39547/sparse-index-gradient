# build.ps1
# Compile LaTeX thesis with XeLaTeX + Biber.

param(
    [switch]$CleanOnly,
    [switch]$FullClean
)

$MainFile = "main"
$TexFile = "$MainFile.tex"
$GeneratedPdfFile = "$MainFile.pdf"
$PdfFile = "precizna_segmentacija.pdf"
$ReportFile = "build-output.txt"
$MaxXeLaTeXPasses = 5
$ReportWriter = $null

function Write-BuildLine {
    param(
        [string]$Message,
        [ConsoleColor]$Color = [ConsoleColor]::Gray
    )

    Write-Host $Message -ForegroundColor $Color
    $script:ReportWriter.WriteLine($Message)
}

function Remove-LaTeXAuxFilesForBase {
    param([string]$BaseName)

    $extensions = @(
        "aux", "bbl", "bcf", "blg", "log", "out", "run.xml",
        "toc", "lof", "lot", "synctex.gz", "synctex", "fls", "fdb_latexmk"
    )

    foreach ($ext in $extensions) {
        $files = Get-ChildItem -Path . -Filter "$BaseName.$ext" -ErrorAction SilentlyContinue
        foreach ($file in $files) {
            for ($attempt = 1; $attempt -le 5; $attempt++) {
                try {
                    Remove-Item -LiteralPath $file.FullName -Force -ErrorAction Stop
                    break
                }
                catch {
                    if ($attempt -eq 5) {
                        throw
                    }
                    Start-Sleep -Milliseconds 250
                }
            }
        }
    }
}

function Remove-OldBuildArtifacts {
    Remove-LaTeXAuxFilesForBase $MainFile
}

function Assert-RequiredCommands {
    param([string[]]$Commands)

    foreach ($command in $Commands) {
        if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
            Write-BuildLine "$command was not found in PATH." Red
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

function Test-NeedsXeLaTeXRerun {
    param([string]$LogPath)

    if (-not (Test-Path $LogPath)) {
        return $false
    }

    $patterns = @(
        'Rerun to get',
        'Label\(s\) may have changed',
        'There were undefined references'
    )

    return [bool](Select-String -LiteralPath $LogPath -Pattern $patterns -Quiet)
}

function Test-NeedsBiber {
    param(
        [string]$BcfPath,
        [string]$BblPath,
        [string[]]$BibliographyPaths,
        [string]$PreviousBcfHash
    )

    if ((-not (Test-Path $BcfPath)) -or (-not (Test-Path $BblPath))) {
        return $true
    }

    $bblWriteTime = (Get-Item -LiteralPath $BblPath).LastWriteTimeUtc
    $currentBcfHash = (Get-FileHash -LiteralPath $BcfPath -Algorithm SHA256).Hash
    if (($PreviousBcfHash -ne "") -and ($currentBcfHash -ne $PreviousBcfHash)) {
        return $true
    }

    foreach ($bibliographyPath in $BibliographyPaths) {
        if ((Test-Path $bibliographyPath) -and ((Get-Item -LiteralPath $bibliographyPath).LastWriteTimeUtc -gt $bblWriteTime)) {
            return $true
        }
    }

    return $false
}

function Get-LogMatches {
    param(
        [string]$Path,
        [string[]]$Patterns
    )

    $results = @()
    if (-not (Test-Path $Path)) {
        return $results
    }

    foreach ($pattern in $Patterns) {
        $found = Select-String -LiteralPath $Path -Pattern $pattern
        foreach ($match in $found) {
            $results += "$($match.Path):$($match.LineNumber): $($match.Line)"
        }
    }
    return $results
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
        'multiply-defined labels',
        'Empty bibliography'
    )

    $problems = Get-LogMatches -Path $LogPath -Patterns $fatalPatterns
    if ($problems.Count -gt 0) {
        Write-BuildLine "Final LaTeX log contains blocking problems:" Red
        foreach ($problem in $problems) {
            Write-BuildLine $problem Red
        }
        return $false
    }

    return $true
}

function Test-FinalBiberLog {
    param([string]$LogPath)

    $fatalPatterns = @(
        "I didn't find a database entry",
        'could not be found in the database',
        'Empty bibliography'
    )

    $problems = Get-LogMatches -Path $LogPath -Patterns $fatalPatterns
    if ($problems.Count -gt 0) {
        Write-BuildLine "Biber log contains blocking problems:" Red
        foreach ($problem in $problems) {
            Write-BuildLine $problem Red
        }
        return $false
    }

    return $true
}

function Show-LayoutWarnings {
    param([string]$LogPath)

    $warnings = Get-LogMatches -Path $LogPath -Patterns @('Overfull \\hbox', 'Overfull \\vbox')
    if ($warnings.Count -eq 0) {
        Write-BuildLine "Layout warnings: none"
        return
    }

    Write-BuildLine "Layout warnings:" Yellow
    foreach ($warning in $warnings) {
        Write-BuildLine $warning Yellow
    }
}

$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$ReportWriter = [System.IO.StreamWriter]::new($ReportFile, $false, $utf8NoBom)

$scriptExitCode = 0

try {
    if ($CleanOnly) {
        Write-BuildLine "Cleaning old auxiliary files..." Cyan
        Remove-OldBuildArtifacts
    }
    else {
        Assert-RequiredCommands -Commands @("xelatex", "biber")

        Write-BuildLine "Closing common PDF viewers..." Yellow
        Get-Process AcroRd32, Acrobat, SumatraPDF -ErrorAction SilentlyContinue |
            Stop-Process -Force

        if ($FullClean) {
            Write-BuildLine "Cleaning old auxiliary files..." Cyan
            Remove-OldBuildArtifacts
        }
        else {
            Write-BuildLine "Incremental build: keeping existing auxiliary files." Cyan
        }

        $previousBcfHash = ""
        if (Test-Path "$MainFile.bcf") {
            $previousBcfHash = (Get-FileHash -LiteralPath "$MainFile.bcf" -Algorithm SHA256).Hash
        }

        $pass1 = Invoke-NativeCommand `
            -Name "XeLaTeX pass 1" `
            -Command "xelatex" `
            -Arguments @("-interaction=nonstopmode", "-file-line-error", $TexFile)

        if (-not (Test-Path "$MainFile.bcf")) {
            Write-BuildLine "Biber input file $MainFile.bcf was not created." Red
            $scriptExitCode = 1
            throw "Build failed before Biber."
        }

        $biberExit = 0
        $biberRan = $false
        if (Test-NeedsBiber -BcfPath "$MainFile.bcf" -BblPath "$MainFile.bbl" -BibliographyPaths @("literatura.bib") -PreviousBcfHash $previousBcfHash) {
            $biberExit = Invoke-NativeCommand `
                -Name "Biber" `
                -Command "biber" `
                -Arguments @($MainFile)
            $biberRan = $true

            if ($biberExit -ne 0) {
                Write-BuildLine "Biber failed." Red
                $scriptExitCode = $biberExit
                throw "Biber failed."
            }
        }
        else {
            Write-BuildLine ""
            Write-BuildLine "Skipping Biber: bibliography inputs are unchanged." Cyan
        }

        $latexExitCodes = @($pass1)
        $latexPass = 2
        while (($biberRan -or (Test-NeedsXeLaTeXRerun -LogPath "$MainFile.log")) -and ($latexPass -le $MaxXeLaTeXPasses)) {
            $biberRan = $false
            $latexExitCodes += Invoke-NativeCommand `
                -Name "XeLaTeX pass $latexPass" `
                -Command "xelatex" `
                -Arguments @("-interaction=nonstopmode", "-file-line-error", $TexFile)
            $latexPass++
        }

        if (Test-NeedsXeLaTeXRerun -LogPath "$MainFile.log") {
            Write-BuildLine "XeLaTeX still requests another pass after $MaxXeLaTeXPasses passes." Red
            $scriptExitCode = 1
            throw "XeLaTeX rerun limit reached."
        }

        $latexOk = Test-FinalLaTeXLog -LogPath "$MainFile.log"
        $biberOk = Test-FinalBiberLog -LogPath "$MainFile.blg"
        Show-LayoutWarnings -LogPath "$MainFile.log"

        if (-not (Test-Path $GeneratedPdfFile)) {
            Write-BuildLine "Build finished, but $GeneratedPdfFile was not created." Red
            $scriptExitCode = 1
            throw "PDF was not created."
        }

        if (($latexExitCodes | Where-Object { $_ -ne 0 }) -or ($biberExit -ne 0) -or (-not $latexOk) -or (-not $biberOk)) {
            Write-BuildLine "Build failed final validation." Red
            $scriptExitCode = 1
            throw "Final validation failed."
        }

        Write-BuildLine "Final validation passed." Green
        if (Test-Path $PdfFile) {
            Remove-Item -LiteralPath $PdfFile -Force
        }
        Move-Item -LiteralPath $GeneratedPdfFile -Destination $PdfFile -Force
        Write-BuildLine "PDF output: $PdfFile" Green

        if ($FullClean) {
            Write-BuildLine "Cleaning auxiliary and temporary files..." Cyan
            Remove-OldBuildArtifacts
        }
        else {
            Write-BuildLine "Keeping auxiliary files for the next incremental build." Cyan
        }
        Write-BuildLine "Build complete: $PdfFile" Green
        Write-BuildLine "Build report: $ReportFile" Green
    }
}
catch {
    if ($scriptExitCode -eq 0) {
        $scriptExitCode = 1
    }
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
