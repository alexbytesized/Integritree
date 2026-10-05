# Extract an authenticated LibreOffice MSI into the project runtime.
# This is an administrative image (/a), not a system-wide installation.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$backendDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runtimeDirectory = Join-Path $backendDirectory 'runtime/document_tools'
$downloadDirectory = Join-Path $runtimeDirectory 'downloads'
$converterDirectory = Join-Path $runtimeDirectory 'libreoffice'
$converterExecutable = Join-Path $converterDirectory 'program/soffice.com'
$version = '26.8.1'
$installerName = "LibreOffice_${version}_Win_x86-64.msi"
$installerPath = Join-Path $downloadDirectory $installerName
$expectedSha256 = '4AA6C6E1895F4055104EFFCB556BD3362D20C6AD707C149543304F395EF9DB95'

New-Item -ItemType Directory -Force -Path $downloadDirectory | Out-Null
if (-not (Test-Path -LiteralPath $installerPath)) {
    Invoke-WebRequest -UseBasicParsing -Uri "https://download.documentfoundation.org/libreoffice/stable/$version/win/x86_64/$installerName" -OutFile $installerPath -TimeoutSec 900
}
if ((Get-FileHash -LiteralPath $installerPath -Algorithm SHA256).Hash -ne $expectedSha256) {
    throw "Installer checksum mismatch. Remove only $installerPath and retry."
}
$signature = Get-AuthenticodeSignature -LiteralPath $installerPath
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=The Document Foundation') {
    throw 'The LibreOffice installer does not have a valid Document Foundation signature.'
}
if (-not (Test-Path -LiteralPath $converterExecutable)) {
    New-Item -ItemType Directory -Force -Path $converterDirectory | Out-Null
    $logPath = Join-Path $runtimeDirectory 'libreoffice-extraction.log'
    $arguments = @('/a', ('"' + $installerPath + '"'), '/qn', ('TARGETDIR="' + $converterDirectory + '"'), '/L*v', ('"' + $logPath + '"'))
    $process = Start-Process -FilePath 'msiexec.exe' -ArgumentList $arguments -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $converterExecutable)) {
        throw "LibreOffice extraction failed (exit $($process.ExitCode)); inspect $logPath."
    }
}
& $converterExecutable --headless --version
if ($LASTEXITCODE -ne 0) {
    throw 'The extracted LibreOffice executable could not start. Check the extraction log and runtime files.'
}
Write-Output "PDF converter ready: $converterExecutable"
