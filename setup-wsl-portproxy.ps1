# setup-wsl-portproxy.ps1 — expose the WSL2 photosearch client to LAN / Tailscale.
#
# The local-replica web app (run-local-replica.sh) runs INSIDE WSL2, on the
# WSL2 VM's private IP (172.x). Tailscale runs on the Windows host, so
# Tailscale/LAN devices can't reach a WSL2 port directly — Windows must forward
# it. (This is why LM Studio's :1234, which is Windows-native, is reachable but
# the photosearch port isn't.)
#
# Run in an **elevated PowerShell (Run as Administrator)** on Windows. The WSL2
# IP changes on every WSL restart, so re-run this after a reboot (or schedule it
# at logon via Task Scheduler).
#
#   powershell -ExecutionPolicy Bypass -File setup-wsl-portproxy.ps1 -Port 8011
#
# Permanent alternative (Windows 11): set `[wsl2] networkingMode=mirrored` in
# %UserProfile%\.wslconfig and `wsl --shutdown` — then WSL ports appear on the
# host directly and no portproxy is needed.

param([int]$Port = 8011)

$wslIp = (wsl hostname -I).Trim().Split(" ")[0]
if (-not $wslIp) { Write-Error "Could not read the WSL2 IP (is WSL running?)."; exit 1 }
Write-Host "WSL2 IP = $wslIp  ->  forwarding 0.0.0.0:$Port to ${wslIp}:$Port"

# Reset any stale mapping for this port, then add the fresh one.
netsh interface portproxy delete v4tov4 listenport=$Port listenaddress=0.0.0.0 2>$null | Out-Null
netsh interface portproxy add    v4tov4 listenport=$Port listenaddress=0.0.0.0 connectport=$Port connectaddress=$wslIp

# Allow inbound on the port through the Windows firewall (idempotent).
netsh advfirewall firewall delete rule name="photosearch $Port" 2>$null | Out-Null
netsh advfirewall firewall add rule name="photosearch $Port" dir=in action=allow protocol=TCP localport=$Port | Out-Null

Write-Host "`nActive port proxies:"
netsh interface portproxy show v4tov4

$ts = (Get-NetIPAddress -AddressFamily IPv4 |
       Where-Object { $_.IPAddress -like "100.*" } |
       Select-Object -First 1 -ExpandProperty IPAddress)
if ($ts) { Write-Host "`nReach it from any Tailscale device at:  http://${ts}:$Port" }
else     { Write-Host "`nReach it at  http://<this-machine-tailscale-ip>:$Port" }
