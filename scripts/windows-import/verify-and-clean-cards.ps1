# verify-and-clean-cards.ps1   (lives in the repo: scripts/windows-import/ — see README.md)
# For every file in the SD-card folders (card-sources.ps1: DCIM, PRIVATE\M4ROOT\CLIP,
# M4ROOT\CLIP on each of $CardDrives), confirm a same-size copy exists somewhere
# under D:\Photos\Archive (nested YYYY\YYYY-MM-DD\ first, legacy flat YYYY-MM-DD\ as fallback).
# Defaults to DRY RUN. Pass -Delete to actually remove verified files from the cards.
# Pass -Hash for SHA256 comparison instead of size-only.

Param(
    [switch]$Delete,
    [switch]$Hash
)

# Same config sources as import-photos-safe.ps1: env PHOTOIMPORT_LocalArchive,
# then import-config.local.ps1 ($LocalArchive, $CardDrives) wins.
$LocalArchive = "D:\Photos\Archive"
$envVal = [Environment]::GetEnvironmentVariable("PHOTOIMPORT_LocalArchive")
if (-not [string]::IsNullOrWhiteSpace($envVal)) { $LocalArchive = $envVal }
$localConfig = Join-Path $PSScriptRoot "import-config.local.ps1"
if (Test-Path -LiteralPath $localConfig) { . $localConfig }
. (Join-Path $PSScriptRoot "card-sources.ps1")
$localArchive = $LocalArchive

$shell = New-Object -ComObject Shell.Application

function Get-File-Date {
    Param ($object)
    $dir = $shell.NameSpace($object.Directory.FullName)
    $file = $dir.ParseName($object.Name)

    # Date Taken (index 12) preferred
    $date = Get-Date-Property-Value $dir $file 12

    if ($null -eq $date) {
        0..287 | ForEach-Object {
            $name = $dir.GetDetailsof($dir.items, $_)
            if ($name -match '(date)|(created)') {
                $tmp = Get-Date-Property-Value $dir $file $_
                if (($null -ne $tmp) -and (($null -eq $date) -or ($tmp -lt $date))) {
                    $date = $tmp
                }
            }
        }
    }
    return $date
}

function Get-Date-Property-Value {
    Param ($dir, $file, $index)
    # -creplace + explicit range, NOT -replace '\P{IsBasicLatin}': -replace is
    # case-insensitive, and under IgnoreCase .NET treats I/i as equivalent to
    # the Turkish dotted/dotless I (U+0130/U+0131) which are outside Basic
    # Latin, so that pattern strips every I/i from the value.
    $value = $dir.GetDetailsof($file, $index) -creplace '[^\x20-\x7E]', ''
    if ($value -and $value -ne '') {
        return [DateTime]::ParseExact($value, "g", $null)
    }
    return $null
}

function Find-ArchiveMatch {
    Param ($file)

    $date = Get-File-Date $file
    if (-not $date) { return $null }

    # The derived date first, then the day either side: a video's earliest
    # date property can be UTC "Media created", which lands on the neighbouring
    # day for clips shot near midnight.
    foreach ($offset in 0, -1, 1) {
        $d          = $date.AddDays($offset)
        $year       = Get-Date -Date $d -Format "yyyy"
        $dateFolder = Get-Date -Date $d -Format "yyyy-MM-dd"

        $nested = Join-Path -Path $localArchive -ChildPath "$year\$dateFolder\$($file.Name)"
        if (Test-Path -LiteralPath $nested) { return $nested }

        $flat = Join-Path -Path $localArchive -ChildPath "$dateFolder\$($file.Name)"
        if (Test-Path -LiteralPath $flat) { return $flat }
    }
    return $null
}

function Files-Match {
    Param ($cardFile, [string]$archivePath, [bool]$useHash)

    $archInfo = Get-Item -LiteralPath $archivePath
    if ($cardFile.Length -ne $archInfo.Length) { return $false }
    if (-not $useHash) { return $true }

    $cardHash = (Get-FileHash -LiteralPath $cardFile.FullName -Algorithm SHA256).Hash
    $archHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash
    return $cardHash -eq $archHash
}

$mode = if ($Delete) { "DELETE" } else { "DRY RUN" }
Write-Host "=== Verify and clean SD cards ($mode) ===" -ForegroundColor Cyan
Write-Host "Archive: $localArchive"
Write-Host "Compare: $(if ($Hash) { 'size + SHA256' } else { 'size only' })"
Write-Host "Cards:   $($CardDrives -join ', ')  (DCIM, PRIVATE\M4ROOT\CLIP, M4ROOT\CLIP)"
if (-not $Delete) { Write-Host "(Pass -Delete to actually remove verified files from the cards)" -ForegroundColor DarkGray }
Write-Host ""

$totalChecked    = 0
$totalDeletable  = 0
$totalDeleted    = 0
$totalUnmatched  = 0
$totalErrors     = 0
$unmatchedFiles  = New-Object System.Collections.Generic.List[string]
$errorFiles      = New-Object System.Collections.Generic.List[string]

$cardSources = @(Get-CardSources)
if ($cardSources.Count -eq 0) {
    Write-Host "No card folders found on $($CardDrives -join ', ')." -ForegroundColor Yellow
}
foreach ($src in $cardSources) {
    $cardPath = $src.Path
    Write-Host "Scanning $cardPath..." -ForegroundColor Cyan
    $files = Get-ChildItem -LiteralPath $cardPath -Recurse:$src.Recurse -File
    foreach ($f in $files) {
        $totalChecked++
        try {
            $match = Find-ArchiveMatch $f
            if ($match -and (Files-Match $f $match $Hash.IsPresent)) {
                $totalDeletable++
                if ($Delete) {
                    Remove-Item -LiteralPath $f.FullName -Force -ErrorAction Stop
                    $totalDeleted++
                }
            } else {
                $totalUnmatched++
                $reason = if (-not $match) { "no archive copy" } else { "size/hash differs from $match" }
                $unmatchedFiles.Add("$($f.FullName)  ($reason)")
                Write-Host "  UNMATCHED: $($f.FullName)  [$reason]" -ForegroundColor Yellow
            }
        } catch {
            $totalErrors++
            $errorFiles.Add("$($f.FullName)  ($_)")
            Write-Host "  ERROR on $($f.FullName): $_" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host "Files checked:   $totalChecked"
Write-Host "Verified copies: $totalDeletable"
if ($Delete) {
    Write-Host "Deleted:         $totalDeleted" -ForegroundColor Green
} else {
    Write-Host "Would delete:    $totalDeletable  (DRY RUN -- pass -Delete to actually remove)" -ForegroundColor Yellow
}
Write-Host "Unmatched:       $totalUnmatched" -ForegroundColor $(if ($totalUnmatched -gt 0) { 'Yellow' } else { 'Green' })
if ($totalErrors -gt 0) {
    Write-Host "Errors:          $totalErrors" -ForegroundColor Red
}

if ($unmatchedFiles.Count -gt 0) {
    Write-Host "`nUnmatched files (not deleted):" -ForegroundColor Yellow
    $unmatchedFiles | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
}
