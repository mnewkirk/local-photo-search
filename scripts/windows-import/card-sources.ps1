# card-sources.ps1 -- the SD-card folders to import from / verify against.
# Dot-sourced by import-photos-safe.ps1 and verify-and-clean-cards.ps1 so the
# two can't drift apart again: 2026-10-10 a card with its clips at
# H:\M4ROOT\CLIP (not H:\PRIVATE\M4ROOT\CLIP) was skipped by BOTH scripts --
# the clips were never imported, and the cleaner never saw them.
#
# Sony bodies differ: newer ones write video to <card>\PRIVATE\M4ROOT\CLIP,
# some write <card>\M4ROOT\CLIP. Windows paths are case-insensitive.
#
# Override the drive letters in import-config.local.ps1:  $CardDrives = @('F:', 'H:', 'I:')

if (-not $CardDrives) { $CardDrives = @('F:', 'H:') }

# Sub-folder, and whether to recurse into it (DCIM has 100MSDCF\ etc.; CLIP is flat).
$CardSubPaths = @(
    @{ Path = 'DCIM';                 Recurse = $true  },
    @{ Path = 'PRIVATE\M4ROOT\CLIP';  Recurse = $false },
    @{ Path = 'M4ROOT\CLIP';          Recurse = $false }
)

# Every card folder that exists right now, as @{ Path; Recurse }.
function Get-CardSources {
    foreach ($drive in $CardDrives) {
        $root = $drive.TrimEnd('\') + '\'
        foreach ($sub in $CardSubPaths) {
            $p = Join-Path $root $sub.Path
            if (Test-Path -LiteralPath $p -PathType Container) {
                @{ Path = $p; Recurse = $sub.Recurse }
            }
        }
    }
}
