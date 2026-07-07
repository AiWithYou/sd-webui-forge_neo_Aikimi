$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$Utf8NoBom = New-Object System.Text.UTF8Encoding $false
[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom
chcp 65001 | Out-Null

Set-Location -LiteralPath $PSScriptRoot

$Downloads = @(
    @{
        Label = "Krea2 NF4 diffusion model"
        Url = "https://huggingface.co/EllaPriest45/Krea2_checkpoints/resolve/main/Krea2_Center%20NF4%20-%20Krea2.safetensors"
        RelativePath = "models\Stable-diffusion\Krea2_Center NF4 - Krea2.safetensors"
        Bytes = [Int64]7673678728
        Markers = @("txtfusion.projector.weight", "_quantization_metadata")
    },
    @{
        Label = "Krea2 Qwen3-VL text encoder"
        Url = "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors"
        RelativePath = "models\text_encoder\qwen3vl_4b_fp8_scaled.safetensors"
        Bytes = [Int64]5242467968
        Markers = @("model.embed_tokens.weight", "model.visual.blocks.0.attn.qkv.weight")
    },
    @{
        Label = "Krea2 Qwen Image VAE"
        Url = "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors"
        RelativePath = "models\VAE\qwen_image_vae.safetensors"
        Bytes = [Int64]253806246
        Markers = @("conv1.weight", "decoder.conv1.weight")
    }
)

function Format-Bytes {
    param([Int64]$Bytes)

    if ($Bytes -ge 1GB) {
        return ("{0:N2} GiB" -f ($Bytes / 1GB))
    }
    if ($Bytes -ge 1MB) {
        return ("{0:N2} MiB" -f ($Bytes / 1MB))
    }
    return ("{0:N0} bytes" -f $Bytes)
}

function New-RequiredDirectory {
    param([string]$Path)

    if (Test-Path -LiteralPath $Path) {
        $item = Get-Item -LiteralPath $Path
        if (-not $item.PSIsContainer) {
            throw "A file exists where a directory is required: $Path"
        }
        return
    }

    New-Item -ItemType Directory -Path $Path | Out-Null
}

function Assert-FileSize {
    param(
        [string]$Path,
        [Int64]$ExpectedBytes
    )

    $actualBytes = (Get-Item -LiteralPath $Path).Length
    if ($actualBytes -ne $ExpectedBytes) {
        $expected = Format-Bytes $ExpectedBytes
        $actual = Format-Bytes $actualBytes
        throw "File size mismatch: $Path`nExpected: $expected ($ExpectedBytes bytes)`nActual: $actual ($actualBytes bytes)"
    }
}

function Test-SafetensorsHeader {
    param(
        [string]$Path,
        [string[]]$Markers
    )

    $stream = [System.IO.File]::OpenRead($Path)
    try {
        if ($stream.Length -lt 16) {
            throw "File is too small to be safetensors: $Path"
        }

        $lengthBytes = New-Object byte[] 8
        $read = $stream.Read($lengthBytes, 0, 8)
        if ($read -ne 8) {
            throw "Cannot read safetensors header length: $Path"
        }

        $headerLength = [BitConverter]::ToUInt64($lengthBytes, 0)
        if ($headerLength -le 2 -or $headerLength -gt 104857600) {
            throw "Invalid safetensors header length: $Path ($headerLength bytes)"
        }

        $headerBytes = New-Object byte[] ([Int32]$headerLength)
        $offset = 0
        while ($offset -lt $headerBytes.Length) {
            $read = $stream.Read($headerBytes, $offset, $headerBytes.Length - $offset)
            if ($read -le 0) {
                throw "Cannot read the full safetensors header: $Path"
            }
            $offset += $read
        }
    }
    finally {
        $stream.Dispose()
    }

    $headerJson = [System.Text.Encoding]::UTF8.GetString($headerBytes)
    $null = $headerJson | ConvertFrom-Json

    foreach ($marker in $Markers) {
        if (-not $headerJson.Contains($marker)) {
            throw "Required safetensors marker was not found: $marker`nTarget: $Path"
        }
    }
}

function Download-ModelFile {
    param(
        [string]$CurlPath,
        [string]$Label,
        [string]$Url,
        [string]$TargetPath,
        [Int64]$ExpectedBytes,
        [string[]]$Markers
    )

    $targetDirectory = Split-Path -Parent $TargetPath
    New-RequiredDirectory $targetDirectory

    $expectedText = Format-Bytes $ExpectedBytes
    Write-Host ""
    Write-Host "== $Label =="
    Write-Host "Target: $TargetPath"
    Write-Host "Size: $expectedText"

    if (Test-Path -LiteralPath $TargetPath) {
        Assert-FileSize -Path $TargetPath -ExpectedBytes $ExpectedBytes
        Test-SafetensorsHeader -Path $TargetPath -Markers $Markers
        Write-Host "Existing file is valid. Skipping download."
        return
    }

    $partialPath = "$TargetPath.part"
    if (Test-Path -LiteralPath $partialPath) {
        $partialBytes = (Get-Item -LiteralPath $partialPath).Length
        if ($partialBytes -eq $ExpectedBytes) {
            Assert-FileSize -Path $partialPath -ExpectedBytes $ExpectedBytes
            Test-SafetensorsHeader -Path $partialPath -Markers $Markers
            Move-Item -LiteralPath $partialPath -Destination $TargetPath
            Assert-FileSize -Path $TargetPath -ExpectedBytes $ExpectedBytes
            Test-SafetensorsHeader -Path $TargetPath -Markers $Markers
            Write-Host "Completed partial file was finalized."
            return
        }
        if ($partialBytes -gt $ExpectedBytes) {
            throw "Partial file is larger than expected. Stop: $partialPath"
        }
        Write-Host ("Resuming from partial file: {0}" -f (Format-Bytes $partialBytes))
    }

    & $CurlPath --location --fail --continue-at - --retry 5 --retry-delay 5 --retry-all-errors --output $partialPath $Url
    if ($LASTEXITCODE -ne 0) {
        throw "curl download failed: $Label"
    }

    Assert-FileSize -Path $partialPath -ExpectedBytes $ExpectedBytes
    Test-SafetensorsHeader -Path $partialPath -Markers $Markers
    Move-Item -LiteralPath $partialPath -Destination $TargetPath
    Assert-FileSize -Path $TargetPath -ExpectedBytes $ExpectedBytes
    Test-SafetensorsHeader -Path $TargetPath -Markers $Markers
    Write-Host "Done."
}

$curlCommand = Get-Command curl.exe -ErrorAction Stop
$curlPath = $curlCommand.Source

Write-Host "This script downloads Krea2 files into Forge model folders."
$totalBytes = [Int64]0
foreach ($download in $Downloads) {
    $totalBytes += [Int64]$download["Bytes"]
}
Write-Host ("Total download size is about {0} from Hugging Face." -f (Format-Bytes $totalBytes))
Write-Host "If interrupted, run this file again to resume from the .part file."

foreach ($download in $Downloads) {
    $targetPath = Join-Path $PSScriptRoot $download["RelativePath"]
    Download-ModelFile `
        -CurlPath $curlPath `
        -Label $download["Label"] `
        -Url $download["Url"] `
        -TargetPath $targetPath `
        -ExpectedBytes $download["Bytes"] `
        -Markers $download["Markers"]
}

Write-Host ""
Write-Host "Krea2 files are ready."
Write-Host "Start webui-user.bat next, then select these values in Forge UI."
Write-Host "Preset: krea"
Write-Host "Checkpoint: Krea2_Center NF4 - Krea2.safetensors"
Write-Host "VAE / Text Encoder: qwen_image_vae.safetensors, qwen3vl_4b_fp8_scaled.safetensors"
Write-Host "Diffusion in Low Bits: bnb-nf4"
