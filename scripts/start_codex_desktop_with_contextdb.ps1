<#
Launch Codex Desktop with the ContextDB MCP configuration directory.

Close every existing Codex Desktop window before running this script. A new
Desktop process is required because MCP servers are discovered at process and
task startup.
#>

$ErrorActionPreference = 'Stop'

$codexHome = Join-Path $env:USERPROFILE '.codex'
$package = Get-AppxPackage -Name 'OpenAI.Codex' | Sort-Object Version -Descending | Select-Object -First 1

if (-not $package) {
    throw 'Codex Desktop is not installed for the current Windows user.'
}

$desktopExecutable = Join-Path $package.InstallLocation 'app\ChatGPT.exe'
if (-not (Test-Path -LiteralPath $desktopExecutable)) {
    throw "Codex Desktop executable was not found: $desktopExecutable"
}

$env:CODEX_HOME = $codexHome
Start-Process -FilePath $desktopExecutable -WorkingDirectory $env:USERPROFILE

Write-Output "Started Codex Desktop with CODEX_HOME=$codexHome"
