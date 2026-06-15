#Requires -Version 5.1
<#
.SYNOPSIS
    Compiles the Rust safety-envelope payload for all three guardrail packages
    and stages each guardrail.wasm next to its main.py.

.DESCRIPTION
    Iterates over every *_wasm_guardrail package, runs a release build against
    the wasm32-unknown-unknown target, and copies the resulting module to the
    package root so `python main.py` can load it directly.

.EXAMPLE
    .\build.ps1
    .\build.ps1 -Clean
#>

[CmdletBinding()]
param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$Packages = @(
    "cartpole_wasm_guardrail",
    "lander_wasm_guardrail",
    "bipedal_wasm_guardrail"
)

$Target  = "wasm32-unknown-unknown"
$Artifact = "guardrail.wasm"
$RepoRoot = $PSScriptRoot

function Assert-Toolchain {
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        throw "cargo not found. Install the Rust toolchain from https://rustup.rs and re-run."
    }
    $installed = (& rustup target list --installed)
    if ($installed -notcontains $Target) {
        Write-Host "==> Installing missing Rust target: $Target" -ForegroundColor Yellow
        & rustup target add $Target
    }
}

function Build-Package {
    param([string]$Name)

    $pkgPath = Join-Path $RepoRoot $Name
    if (-not (Test-Path $pkgPath)) {
        Write-Warning "Skipping '$Name' (directory not found)."
        return
    }

    Write-Host ""
    Write-Host "==> Building $Name" -ForegroundColor Cyan

    Push-Location $pkgPath
    try {
        if ($Clean) {
            & cargo clean
            if ($LASTEXITCODE -ne 0) { throw "cargo clean failed for $Name" }
        }

        & cargo build --target $Target --release
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed for $Name" }

        $built = Join-Path $pkgPath "target\$Target\release\$Artifact"
        if (-not (Test-Path $built)) {
            throw "Expected artifact not found: $built"
        }

        $dest = Join-Path $pkgPath $Artifact
        Copy-Item -Path $built -Destination $dest -Force

        $sizeKb = [math]::Round((Get-Item $dest).Length / 1KB, 1)
        Write-Host "    staged $Artifact ($sizeKb KB) -> $Name\" -ForegroundColor Green
    }
    finally {
        Pop-Location
    }
}

Write-Host "Deterministic Safety Envelope :: WASM build pipeline" -ForegroundColor White
Assert-Toolchain

foreach ($pkg in $Packages) {
    Build-Package -Name $pkg
}

Write-Host ""
Write-Host "Build complete. Run a benchmark with:" -ForegroundColor White
Write-Host "    cd <package>; pip install -r requirements.txt; python main.py" -ForegroundColor Gray