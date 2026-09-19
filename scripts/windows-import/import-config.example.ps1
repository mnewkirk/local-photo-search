# Copy to import-config.local.ps1 (same folder) and fill in. That file is
# git-ignored: hostnames and usernames stay out of the repository.
# It is dot-sourced by import-photos-safe.ps1, so plain assignments are all it needs.

$NasPhotosShare = '\\<nas-host>\<share>\Photos'   # UNC path to the NAS photo library root
$NasSshTarget   = '<user>@<nas-host>'             # ssh target used to trigger ingest-incoming

# Optional overrides (defaults shown):
# $StagingPath    = 'D:\Photos\00 - Latest from the SD Cards'
# $LocalArchive   = 'D:\Photos\Archive'
# $OrganizeScript = 'D:\Photos\organize.ps1'
# $NasComposeDir  = '/volume1/docker/photosearch'
