# Windows SD-card importer

`import-photos-safe.ps1` copies new media off the SD cards into a local staging
folder, pushes it to the NAS `_incoming\<CameraModel>\` staging area, files the
local copy into a dated archive (via your own `organize.ps1`), and ssh-triggers
`ingest-incoming --no-clip` on the NAS — photos are dated, deduped and get DB
rows there, but CLIP is left to the worker fleet's `clip` pass rather than the
N100 (**the NAS must be running code with `--no-clip`**, or the trigger fails
on an unknown option). See the header of the script for the full flow,
and "Phone-photo daily ingest" in the repo's `CLAUDE.md` for the NAS side.

## Setup (once per machine)

```powershell
cd <repo>\scripts\windows-import
Copy-Item import-config.example.ps1 import-config.local.ps1
notepad import-config.local.ps1      # set $NasPhotosShare and $NasSshTarget
```

`import-config.local.ps1` is git-ignored — the NAS hostname and username never
go in the repository. Environment variables work too
(`PHOTOIMPORT_NasPhotosShare`, `PHOTOIMPORT_NasSshTarget`, …); the local file
wins if both are set. The script refuses to run without `NasPhotosShare`.

## Run

```powershell
.\import-photos-safe.ps1                 # import, push, archive, trigger ingest
.\import-photos-safe.ps1 -NoIngest       # everything except the NAS trigger
.\import-photos-safe.ps1 -DryRunIngest   # trigger ingest-incoming --dry-run
```

## Two things this script gets wrong less than it used to

- **RAWs with no readable model.** The camera model is read through the Windows
  shell property store, which has no codec for a new body's RAWs — every
  ILCE-7RM6 `.ARW` read blank and went to `unknown-camera`, away from its JPEG.
  A blank RAW now borrows the model of the same-stem photo beside it, then the
  shoot's single body. The NAS also self-corrects (`ingest._file_suffix` trusts
  the file's own EXIF over the `unknown-camera` fallback), so an old copy of
  this script can no longer misfile anything.
- **The folder-name sanitizer** uses case-sensitive `-creplace '[^\x20-\x7E]'`.
  It was `-replace '\P{IsBasicLatin}'`; `-replace` is case-insensitive, and the
  working theory for a one-off `_incoming\LCE-7RM6\` (2026-08-08, 5,119 files)
  is that under PowerShell 7 a case-insensitive match treats `I` as U+0130,
  which is not Basic Latin. Unconfirmed — but the explicit range is right
  either way.

Avoid running a large backup/sync that reads the NAS share while an import is
ingesting: on 2026-09-19 a ~50 MB/s SMB read slowed ingest to one file per
~3 minutes.
