# Segment slides on a Windows machine: download -> CellViT -> .npz -> keep.
#
# A port of workstation_segment.sh for boxes where WSL will not start. Same
# pipeline, same skip logic, same .npz layout, so its output is interchangeable
# with the bash version's and with Myriad's.
#
#     .\workstation_segment.ps1 -Batch $HOME\batch_pc.tsv
#
# batch.tsv is "uuid filename size" per line. The size column is not optional:
# a download can return success having written a TRUNCATED file, and CellViT
# then dies with "Unsupported or missing image file" twenty minutes later.
#
# Parameters mirror the shell script's environment variables:
#   -Work        scratch workspace       default: the batch file's directory
#   -Keep        where .npz files go     default: $HOME\cellvit_out
#   -N           stop after N slides     default: 0 (all)
#   -RayWorker   Ray pool size; needed on machines with few cores
#   -BatchSize   patch batch; lower on cards with <16GB
#
# Safe to re-run: slides whose .npz is already in -Keep are skipped.

param(
    [Parameter(Mandatory = $true)][string]$Batch,
    [string]$Work,
    [string]$Keep = "$HOME\cellvit_out",
    [int]$N = 0,
    [int]$RayWorker = 0,
    [int]$BatchSize = 8
)

$ErrorActionPreference = "Continue"   # one bad slide must not kill the run

if (-not (Test-Path $Batch)) { throw "no such batch file: $Batch" }
$Batch = (Resolve-Path $Batch).Path
if (-not $Work) { $Work = Split-Path $Batch -Parent }

$here = Split-Path $MyInvocation.MyCommand.Path -Parent
$cachePy = Join-Path $here "cache_cells.py"
$mppPy = Join-Path $here "check_mpp.py"
$launchPy = Join-Path $here "cellvit_launch.py"
foreach ($f in @($cachePy, $mppPy, $launchPy)) {
    if (-not (Test-Path $f)) { throw "missing $f" }
}
if (-not (Get-Command cellvit-inference -ErrorAction SilentlyContinue)) {
    throw "cellvit-inference not on PATH -- activate the conda env first"
}

New-Item -ItemType Directory -Force -Path "$Work\slides", "$Work\out", $Keep | Out-Null

function Get-Size($path) {
    if (Test-Path $path) { (Get-Item $path).Length } else { 0 }
}

# resume-and-verify, 3 attempts. Invoke-WebRequest rather than curl so this
# works on a machine with nothing installed.
function Get-Slide($uuid, $dest, $want) {
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Invoke-WebRequest -Uri "https://api.gdc.cancer.gov/data/$uuid" `
                -OutFile $dest -UseBasicParsing -TimeoutSec 3600
        } catch {
            Write-Host "  attempt $attempt failed: $($_.Exception.Message)"
            Start-Sleep -Seconds 5; continue
        }
        $got = Get-Size $dest
        if ($want -le 0 -or $got -eq $want) { return $true }
        Write-Host "  attempt ${attempt}: got $got of $want bytes, retrying"
        Start-Sleep -Seconds 5
    }
    return $false
}

$rows = Get-Content $Batch | Where-Object { $_.Trim() -ne "" }
Write-Host "batch $Batch ($($rows.Count) slides)"
Write-Host "work  $Work"
$already = @(Get-ChildItem $Keep -Recurse -Filter cells_cache.npz -ErrorAction SilentlyContinue).Count
Write-Host "keep  $Keep  (already have $already)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
Write-Host ""

$done = 0; $skip = 0; $fail = 0
$noMpp = Join-Path $Keep "no_mpp.txt"

foreach ($row in $rows) {
    if ($N -gt 0 -and $done -ge $N) { Write-Host "reached N=$N"; break }
    $f = $row -split '\s+'
    if ($f.Count -lt 2) { continue }
    $uuid = $f[0]; $fname = $f[1]
    $want = if ($f.Count -ge 3) { [int64]$f[2] } else { 0 }

    # TCGA-A2-A0CQ-01Z-...svs -> TCGA-A2-A0CQ-01Z, matching what Myriad stores
    $id = $fname.Split('.')[0]
    if (Test-Path (Join-Path $Keep "$id\cells_cache.npz")) { $skip++; continue }
    if ((Test-Path $noMpp) -and (Select-String -Path $noMpp -Pattern "^$id$" -Quiet)) {
        $skip++; continue
    }

    $svs = Join-Path $Work "slides\$fname"
    $outdir = Join-Path $Work "out\$id"
    Write-Host "=============================================================="
    Write-Host "[$($done + $fail + 1)] $id  $(Get-Date -Format HH:mm:ss)"

    if (-not (Get-Slide $uuid $svs $want)) {
        Write-Host "  download failed after 3 attempts -- skipping"
        $fail++; Remove-Item $svs -Force -ErrorAction SilentlyContinue; continue
    }

    # Reject slides with no usable MPP BEFORE spending GPU time: a truncated
    # Aperio header carries neither MPP nor magnification, and guessing 40x
    # would silently rescale every micron-denominated parameter.
    $meta = (& python $mppPy $svs 2>$null | Out-String).Trim()
    if (-not $meta) {
        Write-Host "  SKIPPED: no usable MPP in slide metadata"
        Add-Content -Path $noMpp -Value $id
        $skip++; Remove-Item $svs -Force -ErrorAction SilentlyContinue; continue
    }
    $mpp, $mp = $meta -split '\s+'
    $mb = [math]::Round((Get-Size $svs) / 1MB)
    Write-Host "  ${mb}MB downloaded (size verified), mpp $mpp, ${mp}MP, segmenting ..."

    New-Item -ItemType Directory -Force -Path $outdir | Out-Null
    $log = Join-Path $outdir "cellvit.log"
    $cvArgs = @()
    if ($RayWorker -gt 0) { $cvArgs += @("--ray_worker", $RayWorker) }
    $cvArgs += @("--model", "SAM", "--nuclei_taxonomy", "pannuke", "--enforce_amp",
               "--batch_size", $BatchSize, "--geojson", "--outdir", $outdir,
               "process_wsi", "--wsi_path", $svs)

    # Via cellvit_launch.py, not the cellvit-inference shim: on Windows
    # torchvision's bundled DLLs shadow GDAL's, and the launcher imports
    # rasterio first so it binds them before torchvision loads.
    # $LASTEXITCODE, not $?: for a native executable $? reports whether the
    # command was launched, not what it returned.
    & python $launchPy @cvArgs *> $log
    $ok = ($LASTEXITCODE -eq 0)
    if ($ok) { & python $cachePy $outdir *>> $log; $ok = ($LASTEXITCODE -eq 0) }

    if ($ok -and (Test-Path (Join-Path $outdir "cells_cache.npz"))) {
        New-Item -ItemType Directory -Force -Path (Join-Path $Keep $id) | Out-Null
        Copy-Item (Join-Path $outdir "cells_cache.npz") (Join-Path $Keep $id)
        $done++
        Write-Host "  OK -> $Keep\$id\cells_cache.npz"
    } else {
        $fail++
        Write-Host "  FAILED -- last lines of ${log}:"
        Get-Content $log -Tail 5 -ErrorAction SilentlyContinue | ForEach-Object { "    $_" }
        Copy-Item $log (Join-Path $Keep "$id.failed.log") -ErrorAction SilentlyContinue
    }

    Remove-Item $outdir -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item $svs -Force -ErrorAction SilentlyContinue
    try {
        $free = [math]::Round((Get-PSDrive (Split-Path $Work -Qualifier).TrimEnd(':')).Free / 1GB)
        Write-Host "  done $done | failed $fail | free ${free}GB"
    } catch { Write-Host "  done $done | failed $fail" }
}

Write-Host ""
Write-Host "=== segmented $done | skipped $skip | failed $fail ==="
$total = @(Get-ChildItem $Keep -Recurse -Filter cells_cache.npz -ErrorAction SilentlyContinue).Count
Write-Host "$total caches in $Keep"
if (Test-Path $noMpp) {
    Write-Host "$((Get-Content $noMpp).Count) slide(s) skipped for missing MPP -- listed in $noMpp"
}
Write-Host "COPY $Keep TO MYRIAD"
