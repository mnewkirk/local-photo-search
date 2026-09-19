# import-photos-safe.ps1   (lives in the repo: scripts/windows-import/ — see README.md)
# Imports SD-card media to D:\Photos\Archive\YYYY\YYYY-MM-DD\ (local backup),
# then pushes everything to the NAS _incoming\<CameraModel>\ staging area so the
# NAS `ingest-incoming` job dates, dedups, and (for photos) indexes them —
# replacing the old per-year robocopy + ssh-index dance.
#
# Flow:
#   1. Copy new files from SD cards into the staging folder.
#   2. Push staging media (JPEG/HEIC + RAW + video) ->
#      \\nas\Photos\_incoming\<CameraModel>\  (grouped by EXIF camera model,
#      e.g. ILCE-7RM6 / ILCE-7M4; videos with no model use the session model
#      or 'unknown-camera').
#   3. organize.ps1 moves staging -> D:\Photos\Archive\YYYY\YYYY-MM-DD\ (local backup kept).
#   4. (Drift) push any local Archive date that has no folder on the NAS yet.
#   5. ssh-trigger `ingest-incoming` on the NAS (unless -NoIngest).
#
# NOTE: ingest-incoming moves RAW + video into the dated library folders too
# (so every file reaches the NAS) but does NOT index them — only JPEG/HEIC get
# CLIP/colors. Sony db sidecars (.XML/.BIN/etc.) are not media and stay local.

param(
    [switch]$NoIngest,       # skip the remote ingest-incoming trigger
    [switch]$DryRunIngest    # run ingest-incoming with --dry-run (scan, no moves)
)

# --- Configuration -----------------------------------------------------------
# Machine-specific values (the NAS share + SSH target) are NOT in this file: it
# is tracked in git, and hostnames/usernames stay out of the repo. They come
# from import-config.local.ps1 beside this script (git-ignored; copy
# import-config.example.ps1), or from environment variables of the same name
# prefixed PHOTOIMPORT_ (e.g. $env:PHOTOIMPORT_NasPhotosShare).
$StagingPath    = "D:\Photos\00 - Latest from the SD Cards"
$LocalArchive   = "D:\Photos\Archive"
$OrganizeScript = "D:\Photos\organize.ps1"
$NasComposeDir  = "/volume1/docker/photosearch"
$NasPhotosShare = $null      # e.g. \\<nas-host>\<share>\Photos   (REQUIRED)
$NasSshTarget   = $null      # e.g. <user>@<nas-host>              (needed unless -NoIngest)

foreach ($name in 'StagingPath','LocalArchive','OrganizeScript','NasComposeDir','NasPhotosShare','NasSshTarget') {
    $envVal = [Environment]::GetEnvironmentVariable("PHOTOIMPORT_$name")
    if (-not [string]::IsNullOrWhiteSpace($envVal)) { Set-Variable -Name $name -Value $envVal }
}
$localConfig = Join-Path $PSScriptRoot "import-config.local.ps1"
if (Test-Path -LiteralPath $localConfig) { . $localConfig }   # local file wins over env

if ([string]::IsNullOrWhiteSpace($NasPhotosShare)) {
    Write-Host "ERROR: NasPhotosShare is not set." -ForegroundColor Red
    Write-Host "  Copy import-config.example.ps1 to import-config.local.ps1 (same folder) and fill it in." -ForegroundColor Red
    exit 2
}
if (-not $NoIngest -and [string]::IsNullOrWhiteSpace($NasSshTarget)) {
    Write-Host "ERROR: NasSshTarget is not set (needed to trigger ingest). Set it, or pass -NoIngest." -ForegroundColor Red
    exit 2
}

# (PowerShell variable names are case-insensitive: $stagingPath / $localArchive /
# $nasComposeDir below ARE the settings above.)
$nasPath         = $NasPhotosShare
$nasIncoming     = Join-Path $nasPath "_incoming"
$nasUser         = $NasSshTarget
$touchedYearsFile = Join-Path $env:TEMP "photo-touched-years-$PID.txt"

# Media types pushed to the NAS. Photos (JPEG/HEIC) get CLIP-indexed by
# ingest-incoming; RAW + video are moved into the library but not indexed.
# Photos + RAW expose an EXIF camera model; video usually doesn't.
$photoExts     = @('.jpg', '.jpeg', '.heic', '.heif')
$rawExts       = @('.arw', '.cr2', '.cr3', '.nef', '.nrw', '.dng', '.raf', '.rw2', '.orf', '.pef', '.srw', '.raw', '.rwl', '.sr2')
$videoExts     = @('.mp4', '.mov', '.m4v', '.avi', '.mts', '.m2ts', '.3gp')
$modelExts     = $photoExts + $rawExts            # types that expose a camera model
$allMediaExts  = $photoExts + $rawExts + $videoExts

Write-Host "=== Photo Import Starting ===" -ForegroundColor Cyan
Write-Host "Staging:  $stagingPath"
Write-Host "Archive:  $localArchive  (YYYY\YYYY-MM-DD\)"
Write-Host "NAS push: $nasIncoming\<CameraModel>\  (ingest-incoming dates + indexes)"
Write-Host ""

# COM shell used to read EXIF 'Camera model' from each photo.
$shell = New-Object -ComObject Shell.Application
$script:cameraModelIndex = $null   # discovered once, reused (system-wide constant)

# Helper function to copy only new files (skip existing)
function Copy-NewFiles {
    param(
        [string]$Source,
        [string]$Destination,
        [switch]$Recurse
    )

    $copied = 0
    $skipped = 0

    $files = if ($Recurse) {
        Get-ChildItem $Source -Recurse -File
    } else {
        Get-ChildItem $Source -File
    }

    foreach ($file in $files) {
        $relativePath = $file.FullName.Substring($Source.TrimEnd('\').Length + 1)
        $destFile = Join-Path $Destination $relativePath
        $destDir = Split-Path $destFile -Parent

        if (Test-Path $destFile) {
            $skipped++
        } else {
            if (-not (Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir -Force | Out-Null
            }
            Copy-Item $file.FullName $destFile
            $copied++
        }
    }

    return @{ Copied = $copied; Skipped = $skipped }
}

# Read the EXIF 'Camera model' for a file via the Windows shell property store.
# Falls back to System.Drawing EXIF tag 0x0110 for JPEGs, then 'unknown-camera'.
function Get-CameraModel {
    param([System.IO.FileInfo]$File)

    $dir = $shell.NameSpace($File.DirectoryName)
    if ($null -eq $dir) { return $null }
    $item = $dir.ParseName($File.Name)

    # Discover the 'Camera model' column index once (constant per machine).
    if ($null -eq $script:cameraModelIndex) {
        foreach ($i in 0..330) {
            if ($dir.GetDetailsOf($null, $i) -eq 'Camera model') {
                $script:cameraModelIndex = $i
                break
            }
        }
        if ($null -eq $script:cameraModelIndex) { $script:cameraModelIndex = -1 }
    }

    $model = $null
    if ($script:cameraModelIndex -ge 0) {
        $model = $dir.GetDetailsOf($item, $script:cameraModelIndex)
    }

    # Fallback: read EXIF Model (0x0110) directly for JPEGs.
    if ([string]::IsNullOrWhiteSpace($model) -and
        ($File.Extension.ToLower() -in @('.jpg', '.jpeg'))) {
        try {
            Add-Type -AssemblyName System.Drawing -ErrorAction SilentlyContinue
            $img = [System.Drawing.Image]::FromFile($File.FullName)
            try {
                $prop = $img.GetPropertyItem(0x0110)
                $model = [System.Text.Encoding]::ASCII.GetString($prop.Value)
            } finally { $img.Dispose() }
        } catch { }
    }

    if ([string]::IsNullOrWhiteSpace($model)) { return $null }

    # Sanitize for use as a folder name: keep printable ASCII only.
    # -creplace (case-SENSITIVE) with an explicit range, NOT -replace '\P{IsBasicLatin}':
    # -replace is case-insensitive, and under PowerShell 7 / newer .NET a
    # case-insensitive match treats 'I' as equivalent to U+0130 (dotted capital I),
    # which is not Basic Latin - so the "strip non-ASCII" regex could eat a plain
    # ASCII 'I'. A one-off run on 2026-08-08 filed 5,119 photos under
    # _incoming\LCE-7RM6\ instead of ILCE-7RM6.
    $model = ($model -creplace '[^\x20-\x7E]', '').Trim()
    $model = $model -replace '[\\/:*?"<>|]', '_'
    $model = ($model -replace '\s+', ' ').Trim()
    if ([string]::IsNullOrWhiteSpace($model)) { return $null }
    return $model
}

# Determine the single camera model for a batch (from the files that expose
# one: photos + RAW). Returns the model if exactly one is present, else $null.
# Used to give model-less files (video) the same folder as the shoot's photos
# when the batch is from a single body.
function Get-SessionModel {
    param([System.IO.FileInfo[]]$Files)
    $models = @{}
    foreach ($file in $Files) {
        if ($modelExts -contains $file.Extension.ToLower()) {
            $m = Get-CameraModel $file
            if (-not [string]::IsNullOrWhiteSpace($m)) { $models[$m] = $true }
        }
    }
    if ($models.Count -eq 1) { return @($models.Keys)[0] }
    return $null
}

# Push a set of media files to \\nas\...\_incoming\<CameraModel>\<relpath>.
# Copies (not moves) so the local archive copy is preserved. Pushes photos,
# RAW, and video; non-media is skipped. Files with a readable EXIF model
# (photos + RAW) go to that model's dir; model-less files (video) use the
# session model, falling back to 'unknown-camera'. $BaseForRel anchors the
# relative subpath under each model dir so same-name files from different
# folders don't collide before ingest-incoming runs.
function Push-ToIncoming {
    param(
        [System.IO.FileInfo[]]$Files,
        [string]$BaseForRel
    )

    $sessionModel = Get-SessionModel -Files $Files

    $pushed = 0
    $skippedExisting = 0
    $skippedNonMedia = 0
    $models = @{}

    foreach ($file in $Files) {
        $ext = $file.Extension.ToLower()
        if ($allMediaExts -notcontains $ext) { $skippedNonMedia++; continue }

        if ($modelExts -contains $ext) {
            $model = Get-CameraModel $file
            # The shell property store has no codec for a new body's RAWs, so a
            # RAW often reads blank (2026-09-19: every ILCE-7RM6 .ARW). Before
            # giving up, borrow the model from the same-stem photo beside it
            # (DSC00078.JPG for DSC00078.ARW), then from the shoot's single body.
            if ([string]::IsNullOrWhiteSpace($model)) {
                foreach ($pe in $photoExts) {
                    $sib = [System.IO.Path]::ChangeExtension($file.FullName, $pe)
                    if (Test-Path -LiteralPath $sib) {
                        $model = Get-CameraModel (Get-Item -LiteralPath $sib)
                        if (-not [string]::IsNullOrWhiteSpace($model)) { break }
                    }
                }
            }
            if ([string]::IsNullOrWhiteSpace($model) -and $sessionModel) { $model = $sessionModel }
            if ([string]::IsNullOrWhiteSpace($model)) { $model = 'unknown-camera' }
        } else {
            # Video / model-less: ride along with the shoot's body when known.
            $model = if ($sessionModel) { $sessionModel } else { 'unknown-camera' }
        }

        $rel = $file.FullName.Substring($BaseForRel.TrimEnd('\').Length + 1)
        $dest = Join-Path (Join-Path $nasIncoming $model) $rel
        $destDir = Split-Path $dest -Parent

        if (Test-Path -LiteralPath $dest) { $skippedExisting++; continue }
        if (-not (Test-Path -LiteralPath $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        Copy-Item -LiteralPath $file.FullName -Destination $dest
        $pushed++
        $models[$model] = ([int]$models[$model]) + 1
    }

    return @{
        Pushed = $pushed; SkippedExisting = $skippedExisting;
        SkippedNonMedia = $skippedNonMedia; Models = $models
    }
}

# Step 1: Import from SD cards (COPY not MOVE, skip existing)
Write-Host "=== Importing from SD cards ===" -ForegroundColor Cyan

if (Test-Path "F:\DCIM") {
    Write-Host "Copying from F:\DCIM..."
    $result = Copy-NewFiles -Source "F:\DCIM" -Destination $stagingPath -Recurse
    Write-Host "  Copied: $($result.Copied), Skipped: $($result.Skipped)" -ForegroundColor Gray
}
if (Test-Path "F:\private\m4root\clip") {
    Write-Host "Copying videos from F:\private\m4root\clip..."
    $result = Copy-NewFiles -Source "F:\private\m4root\clip" -Destination $stagingPath
    Write-Host "  Copied: $($result.Copied), Skipped: $($result.Skipped)" -ForegroundColor Gray
}
if (Test-Path "H:\dcim") {
    Write-Host "Copying from H:\dcim..."
    $result = Copy-NewFiles -Source "H:\dcim" -Destination $stagingPath -Recurse
    Write-Host "  Copied: $($result.Copied), Skipped: $($result.Skipped)" -ForegroundColor Gray
}
if (Test-Path "H:\private\m4root\clip") {
    Write-Host "Copying videos from H:\private\m4root\clip..."
    $result = Copy-NewFiles -Source "H:\private\m4root\clip" -Destination $stagingPath
    Write-Host "  Copied: $($result.Copied), Skipped: $($result.Skipped)" -ForegroundColor Gray
}

# Count files
$stagingCount = (Get-ChildItem $stagingPath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "Files imported: $stagingCount" -ForegroundColor Cyan

# Step 2: Push staging photos to the NAS _incoming\<CameraModel>\ staging area.
# Done BEFORE organize.ps1 (which moves files out of staging into the archive),
# while everything this run brought in is still in one place.
if ($stagingCount -gt 0) {
    Write-Host "`n=== Pushing photos to NAS _incoming (by camera model) ===" -ForegroundColor Cyan
    if (-not (Test-Path -LiteralPath $nasIncoming)) {
        New-Item -ItemType Directory -Path $nasIncoming -Force | Out-Null
    }
    $stagingFiles = Get-ChildItem $stagingPath -Recurse -File
    $push = Push-ToIncoming -Files $stagingFiles -BaseForRel $stagingPath
    Write-Host "  Pushed: $($push.Pushed)  AlreadyThere: $($push.SkippedExisting)  Non-media(skipped): $($push.SkippedNonMedia)" -ForegroundColor Gray
    foreach ($m in ($push.Models.Keys | Sort-Object)) {
        Write-Host "     _incoming\$m\  ($($push.Models[$m]) file(s))" -ForegroundColor Gray
    }
    if ($push.SkippedNonMedia -gt 0) {
        Write-Host "  Note: $($push.SkippedNonMedia) non-media file(s) skipped (e.g. Sony .XML/.BIN sidecars); kept in local archive." -ForegroundColor DarkYellow
    }
} else {
    Write-Host "(No new files in staging -- skipping NAS push.)" -ForegroundColor DarkGray
}

# Step 3: Organize staging into the local archive (D:\Photos\Archive) — keeps a
# local, date-organized backup independent of the NAS library.
Remove-Item $touchedYearsFile -Force -ErrorAction SilentlyContinue
$env:PHOTO_TOUCHED_YEARS_FILE = $touchedYearsFile

if ($stagingCount -gt 0) {
    Write-Host "`n=== Organizing photos into local archive ===" -ForegroundColor Cyan
    if (Test-Path -LiteralPath $OrganizeScript) {
        & $OrganizeScript $stagingPath $localArchive
    } else {
        Write-Host "ERROR: organize script not found: $OrganizeScript" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "(No new files in staging -- skipping organize.)" -ForegroundColor DarkGray
}

# Read touched years emitted by organize.ps1 (the years for which files landed).
$organizedYears = @()
if (Test-Path $touchedYearsFile) {
    $organizedYears = @(Get-Content $touchedYearsFile | Where-Object { $_ -match '^\d{4}$' })
}

# Step 4: Drift detection -- catch local Archive dates that never made it onto
# the NAS (manual moves, prior crashed runs, etc). Under the _incoming flow the
# NAS library folders are YYYY\YYYY-MM-DD_<model>\, so we compare on the date
# prefix only and re-push any local date with no NAS folder at all.
Write-Host "`n=== Checking for local/NAS drift ===" -ForegroundColor Cyan
$organizedSet = @{}
foreach ($y in $organizedYears) { $organizedSet[$y] = $true }
$driftDates = New-Object System.Collections.Generic.List[string]   # local date-folder full paths
$driftDetail = New-Object System.Collections.Generic.List[string]

Get-ChildItem -LiteralPath $localArchive -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^\d{4}$' } |
    ForEach-Object {
        $year = $_.Name
        # This run's years were just pushed via staging -- skip them.
        if ($organizedSet.ContainsKey($year)) { return }

        $localYearPath = $_.FullName
        $nasYearPath   = Join-Path $nasPath $year

        # Date prefixes already present on the NAS for this year (strip _<model> suffix).
        $nasDatePrefixes = @{}
        if (Test-Path -LiteralPath $nasYearPath) {
            Get-ChildItem -LiteralPath $nasYearPath -Directory -ErrorAction SilentlyContinue |
                ForEach-Object { $nasDatePrefixes[(($_.Name -split '_')[0])] = $true }
        }

        Get-ChildItem -LiteralPath $localYearPath -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^\d{4}-\d{2}-\d{2}' } |
            ForEach-Object {
                $datePrefix = ($_.Name -split '_')[0]
                if (-not $nasDatePrefixes.ContainsKey($datePrefix)) {
                    $driftDates.Add($_.FullName)
                    $driftDetail.Add("  $year\$($_.Name) has no folder on NAS")
                }
            }
    }

if ($driftDates.Count -gt 0) {
    Write-Host "Drift detected ($($driftDates.Count) local date-folder(s) missing on NAS):" -ForegroundColor Yellow
    $driftDetail | ForEach-Object { Write-Host $_ -ForegroundColor Yellow }
    Write-Host ""
    $resp = Read-Host "Push drift folders to _incoming now? [Y/n]"
    if ($resp -eq '' -or $resp -match '^[Yy]') {
        $driftFiles = @()
        foreach ($d in $driftDates) {
            $driftFiles += Get-ChildItem -LiteralPath $d -File -ErrorAction SilentlyContinue
        }
        if ($driftFiles.Count -gt 0) {
            $dpush = Push-ToIncoming -Files $driftFiles -BaseForRel $localArchive
            Write-Host "  Drift pushed: $($dpush.Pushed)  AlreadyThere: $($dpush.SkippedExisting)  Non-media: $($dpush.SkippedNonMedia)" -ForegroundColor Cyan
        }
    } else {
        Write-Host "  Skipping drift push." -ForegroundColor DarkGray
    }
} else {
    Write-Host "No drift detected." -ForegroundColor Green
}

# Step 5: Trigger the NAS ingest-incoming sweep (dates, dedups, CLIP-indexes
# everything we just pushed). Skippable -- the daily cron and the /status
# "Ingest incoming" button run the same job.
if ($NoIngest) {
    Write-Host "`n-NoIngest set -- not triggering the NAS sweep." -ForegroundColor DarkGray
    Write-Host "Run it later from /status (Ingest incoming) or wait for the daily cron." -ForegroundColor DarkGray
} else {
    $dryFlag = if ($DryRunIngest) { " --dry-run" } else { "" }
    Write-Host "`n=== Triggering NAS ingest-incoming$dryFlag ===" -ForegroundColor Cyan
    $remoteCmd = "cd $nasComposeDir && nohup docker compose -f docker-compose.nas.yml run --rm -e PYTHONUNBUFFERED=1 photosearch ingest-incoming --no-colors$dryFlag > /tmp/ingest-incoming.log 2>&1 < /dev/null &"
    ssh $nasUser $remoteCmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  WARN: ssh exited $LASTEXITCODE -- trigger may not have started." -ForegroundColor Yellow
    } else {
        Write-Host "  ingest-incoming started on NAS (log: /tmp/ingest-incoming.log)" -ForegroundColor Green
        Write-Host "  Watch progress on the /status page." -ForegroundColor Green
    }
}

Write-Host "`n=== DONE ===" -ForegroundColor Green
Write-Host "Local archive updated: $localArchive" -ForegroundColor Green
Write-Host "Photos pushed to NAS _incoming and queued for ingest." -ForegroundColor Green
Write-Host "SAFE to delete from SD cards." -ForegroundColor Yellow
Write-Host "Staging preserved at: $stagingPath" -ForegroundColor Yellow

Remove-Item $touchedYearsFile -Force -ErrorAction SilentlyContinue
$env:PHOTO_TOUCHED_YEARS_FILE = $null

