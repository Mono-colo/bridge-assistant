param(
    [string]$PythonExecutable
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ($PythonExecutable) {
    $python = (Resolve-Path $PythonExecutable).Path
}
else {
    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        $python = $venvPython
    }
    else {
        $python = (Get-Command python -ErrorAction Stop).Source
    }
}

Push-Location $projectRoot
try {
    & $python -c "import build, PyInstaller"
    if ($LASTEXITCODE -ne 0) {
        throw "Build tools are missing. Run: python -m pip install build pyinstaller"
    }

    & $python -m build --wheel
    if ($LASTEXITCODE -ne 0) {
        throw "Wheel build failed with exit code $LASTEXITCODE"
    }

    $pyinstallerArgs = @(
        "--noconfirm",
        "--clean",
        "--onefile",
        "--noupx",
        "--name", "bridge_assistant",
        "--distpath", "dist",
        "--workpath", "build\pyinstaller-current",
        "--specpath", "build",
        "--paths", $projectRoot,
        "--collect-submodules", "bridge_assistant",
        "--collect-all", "UnityPy",
        "--exclude-module", "PyQt5",
        "--exclude-module", "PySide6",
        "--exclude-module", "PySide2",
        "--exclude-module", "matplotlib",
        "--exclude-module", "pygame",
        "--exclude-module", "pytest",
        "--exclude-module", "_pytest",
        "--exclude-module", "pygments",
        "--exclude-module", "IPython",
        "--exclude-module", "jupyter",
        "scripts\bridge_assistant_launcher.py"
    )
    & $python -m PyInstaller @pyinstallerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "EXE build failed with exit code $LASTEXITCODE"
    }

    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = (Resolve-Path "dist\bridge_assistant.exe").Path
    $processInfo.Arguments = "--self-test"
    $processInfo.UseShellExecute = $false
    $processInfo.CreateNoWindow = $true
    $selfTest = [System.Diagnostics.Process]::Start($processInfo)
    $selfTest.WaitForExit()
    if ($selfTest.ExitCode -ne 0) {
        throw "Packaged self-test failed with exit code $($selfTest.ExitCode)"
    }

    Write-Host "Built and verified dist\bridge_assistant.exe and wheel."
}
finally {
    Pop-Location
}
