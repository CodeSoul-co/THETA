$ErrorActionPreference = 'Stop'
$desktopRoot = Split-Path $PSScriptRoot -Parent
$oldFolder = Join-Path $env:RUNNER_TEMP 'THETA-previous'
$newFolder = Join-Path $env:RUNNER_TEMP 'THETA-current'
$baselineFolder = Join-Path $env:RUNNER_TEMP 'theta-upgrade-baseline'
New-Item -ItemType Directory -Force $baselineFolder | Out-Null
& gh release download desktop-v0.3.2 --repo CodeSoul-co/THETA --pattern 'THETA-0.3.2-win-x64.exe' --dir $baselineFolder
if ($LASTEXITCODE -ne 0) { throw 'Could not download upgrade baseline' }
function Install-Theta($installer, $folder) {
  $process = Start-Process -FilePath $installer -ArgumentList @('/S', '/currentuser', "/D=$folder") -Wait -PassThru
  if ($process.ExitCode -ne 0) { throw "Installer failed: $($process.ExitCode)" }
  if (!(Test-Path (Join-Path $folder 'THETA.exe'))) { throw 'Installed executable missing' }
}
Install-Theta (Join-Path $baselineFolder 'THETA-0.3.2-win-x64.exe') $oldFolder
$obsolete = Join-Path $oldFolder 'obsolete-version-test.txt'
Set-Content -Path $obsolete -Value 'old application payload'
$userData = Join-Path ([Environment]::GetFolderPath('ApplicationData')) 'THETA'
New-Item -ItemType Directory -Force $userData | Out-Null
$keep = Join-Path $userData 'upgrade-data-retention-test.txt'
Set-Content -Path $keep -Value 'preserve user data'
$version = (Get-Content (Join-Path $desktopRoot 'package.json') | ConvertFrom-Json).version
Install-Theta (Join-Path $desktopRoot "release/THETA-$version-win-x64.exe") $newFolder
if ((Test-Path (Join-Path $oldFolder 'THETA.exe')) -or (Test-Path $obsolete) -or (Test-Path (Join-Path $oldFolder 'resources'))) { throw 'Previous application files remain after upgrade' }
if ((Get-Content $keep) -ne 'preserve user data') { throw 'Upgrade removed user data' }
$entries = @(Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall' | Get-ItemProperty | Where-Object { $_.DisplayName -eq 'THETA' })
if ($entries.Count -ne 1 -or $entries[0].DisplayVersion -ne $version) { throw 'Upgrade must leave one current THETA uninstall entry' }
$env:THETA_DESKTOP_TEST_HOME = Join-Path $env:RUNNER_TEMP 'theta-installed-smoke'
$process = Start-Process -FilePath (Join-Path $newFolder 'THETA.exe') -ArgumentList '--smoke-test' -Wait -PassThru -RedirectStandardOutput (Join-Path $env:RUNNER_TEMP 'theta-installed-smoke.log') -RedirectStandardError (Join-Path $env:RUNNER_TEMP 'theta-installed-smoke-error.log')
if ($process.ExitCode -ne 0) { Get-Content (Join-Path $env:RUNNER_TEMP 'theta-installed-smoke-error.log'); throw 'Installed application smoke test failed' }
Write-Output 'Upgrade verified: previous program removed, one current install entry, user data retained, installed app starts.'
