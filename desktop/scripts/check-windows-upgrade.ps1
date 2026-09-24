param([ValidateSet('download', 'baseline', 'upgrade', 'smoke')][string]$Phase)
$ErrorActionPreference = 'Stop'
$desktopRoot = Split-Path $PSScriptRoot -Parent
$oldFolder = Join-Path $env:RUNNER_TEMP 'THETA-previous'
$newFolder = Join-Path $env:RUNNER_TEMP 'THETA-current'
$baselineFolder = Join-Path $env:RUNNER_TEMP 'theta-upgrade-baseline'
if ($Phase -eq 'download') {
  New-Item -ItemType Directory -Force $baselineFolder | Out-Null
  & gh release download desktop-v0.3.3 --repo CodeSoul-co/THETA --pattern 'THETA-0.3.3-win-x64.exe' --dir $baselineFolder
  if ($LASTEXITCODE -ne 0) { throw 'Could not download upgrade baseline' }
  return
}
function Wait-ThetaProcess($process, $milliseconds, $label) {
  if (!$process.WaitForExit($milliseconds)) {
    Get-PSDrive -PSProvider FileSystem | Select-Object Name, Used, Free | Format-Table
    Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'THETA|uninstall' } | Select-Object ProcessId, ParentProcessId, Name, CommandLine | Format-List
    foreach ($folder in @($oldFolder, $newFolder)) { Get-ChildItem $folder -ErrorAction SilentlyContinue | Select-Object Name, Length | Format-Table }
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    throw "$label timed out after $($milliseconds / 1000) seconds"
  }
  $process.Refresh()
  if ($process.ExitCode -ne 0) { throw "$label failed: $($process.ExitCode)" }
}
function Install-Theta($installer, $folder) {
  $process = Start-Process -FilePath $installer -ArgumentList @('/S', '/currentuser', "/D=$folder") -PassThru
  Wait-ThetaProcess $process 600000 "Installer $installer"
  if (!(Test-Path (Join-Path $folder 'THETA.exe'))) { throw 'Installed executable missing' }
}
$obsolete = Join-Path $oldFolder 'obsolete-version-test.txt'
$userData = Join-Path ([Environment]::GetFolderPath('ApplicationData')) 'THETA'
$keep = Join-Path $userData 'upgrade-data-retention-test.txt'
if ($Phase -eq 'baseline') {
  Install-Theta (Join-Path $baselineFolder 'THETA-0.3.3-win-x64.exe') $oldFolder
  Set-Content -Path $obsolete -Value 'old application payload'
  New-Item -ItemType Directory -Force $userData | Out-Null
  Set-Content -Path $keep -Value 'preserve user data'
  Write-Output 'Baseline installed and data-retention fixture created.'
  return
}
$version = (Get-Content (Join-Path $desktopRoot 'package.json') | ConvertFrom-Json).version
if ($Phase -eq 'upgrade') {
Install-Theta (Join-Path $desktopRoot "release/THETA-$version-win-x64.exe") $newFolder
if ((Test-Path (Join-Path $oldFolder 'THETA.exe')) -or (Test-Path $obsolete) -or (Test-Path (Join-Path $oldFolder 'resources'))) { throw 'Previous application files remain after upgrade' }
if ((Get-Content $keep) -ne 'preserve user data') { throw 'Upgrade removed user data' }
$entries = @(Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall' | Get-ItemProperty | Where-Object { $_.DisplayName -eq 'THETA' })
if ($entries.Count -ne 1 -or $entries[0].DisplayVersion -ne $version) { throw 'Upgrade must leave one current THETA uninstall entry' }
Write-Output 'Upgrade verified: previous program removed, one current install entry, user data retained.'
return
}
$env:THETA_DESKTOP_TEST_HOME = Join-Path $env:RUNNER_TEMP 'theta-installed-smoke'
$process = Start-Process -FilePath (Join-Path $newFolder 'THETA.exe') -ArgumentList '--smoke-test' -PassThru -RedirectStandardOutput (Join-Path $env:RUNNER_TEMP 'theta-installed-smoke.log') -RedirectStandardError (Join-Path $env:RUNNER_TEMP 'theta-installed-smoke-error.log')
try { Wait-ThetaProcess $process 240000 'Installed application smoke test' }
finally {
  Get-Content (Join-Path $env:RUNNER_TEMP 'theta-installed-smoke.log') -ErrorAction SilentlyContinue
  Get-Content (Join-Path $env:RUNNER_TEMP 'theta-installed-smoke-error.log') -ErrorAction SilentlyContinue
}
Write-Output 'Upgrade verified: previous program removed, one current install entry, user data retained, installed app starts.'
