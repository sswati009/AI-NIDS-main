# Isolated Docker lab

The reproducible lab definition is in `docker-compose.lab.yml`. Both services
are attached only to `ai-nids-lab-isolated`, an internal bridge on
`172.30.50.0/24`. Juice Shop has no published host port. Docker's internal
network mode blocks traffic between the lab and other networks; the Docker host
can still communicate directly with container IPs, so the host remains inside
the trust boundary.

Both services target `linux/amd64`. The client has fixed address
`172.30.50.10`; Juice Shop has fixed address `172.30.50.20`. The client image
pins Alpine 3.24.1 by digest and pins the
`tcpdump` and `busybox-extras` package versions. Juice Shop is pinned to version
20.2.0 and an image digest. Capture PCAPs are written to the named
`ai-nids-lab-captures` volume, which is retained when containers are recreated.
Do not pass `--volumes` or `-v` to `docker compose down` if captures must be
kept.

The first setup builds/pulls images before starting containers. Traffic
scenarios, when deliberately launched from the dashboard, run only from the
client to `http://juice-shop:3000` inside this isolated network.

```powershell
docker compose -f docker-compose.lab.yml config -q
docker compose -f docker-compose.lab.yml pull juice-shop
docker compose -f docker-compose.lab.yml build --pull nids-client
docker compose -f docker-compose.lab.yml up -d --no-build --pull never
```

## Migrating the existing lab

The current `nids-client` writes rotating PCAPs into its container's `/tmp` and
has no mounted volume. Preserve and close those files before stopping the
current containers. These commands are provided for the reviewed migration;
they have not been run as part of the code/configuration change.

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$archive = Join-Path (Get-Location) "data\live\pre-isolation-$stamp"
New-Item -ItemType Directory -Path $archive | Out-Null

docker exec nids-client pkill -INT -x tcpdump
if ($LASTEXITCODE -notin @(0, 1)) { throw 'Could not signal tcpdump in nids-client.' }

$remaining = $null
for ($i = 0; $i -lt 10; $i++) {
    $remaining = docker exec nids-client pidof tcpdump 2>$null
    if ($LASTEXITCODE -eq 1) { break }
    Start-Sleep -Seconds 1
}
if ($LASTEXITCODE -ne 1) { throw 'tcpdump is still running; do not stop the client yet.' }

$pcaps = @(docker exec nids-client sh -c "find /tmp -maxdepth 1 -type f -name 'ai-nids-lab-*.pcap' -print")
if ($LASTEXITCODE -ne 0) { throw 'Could not list existing PCAPs in nids-client.' }
foreach ($pcap in $pcaps) {
    $pcap = $pcap.Trim()
    if (-not $pcap) { continue }
    $destination = Join-Path $archive (Split-Path -Leaf $pcap)
    docker cp ("nids-client:$pcap") $destination
    if ($LASTEXITCODE -ne 0) { throw "Could not preserve $pcap at $destination." }
}
Get-ChildItem -File $archive | Get-FileHash -Algorithm SHA256

docker compose -f docker-compose.lab.yml config -q
if ($LASTEXITCODE -ne 0) { throw 'Compose configuration validation failed.' }
docker compose -f docker-compose.lab.yml pull juice-shop
if ($LASTEXITCODE -ne 0) { throw 'Could not pull the pinned Juice Shop image.' }
docker compose -f docker-compose.lab.yml build --pull nids-client
if ($LASTEXITCODE -ne 0) { throw 'Could not build the pinned NIDS client image.' }

docker stop nids-client juice-shop
if ($LASTEXITCODE -ne 0) { throw 'Could not stop the existing lab containers cleanly.' }
docker compose -f docker-compose.lab.yml up -d --no-build --pull never
if ($LASTEXITCODE -ne 0) { throw 'Could not start the isolated lab stack.' }
```

This migration leaves the original containers and `ai-nids-lab` network in
place, stopped, so their writable layers remain available for rollback. The
new stack uses distinct names and network resources. The saved PCAP copies are
additional local backups; existing project PCAPs, flow CSVs and prediction
CSVs are not part of this migration sequence.
