# YOLOv5 Validation Wrapper Script
# Run from project root: .\scripts\val.ps1 -Weights "path/to/best.pt"
#
# This script validates a trained YOLOv5 model on the validation set.

param(
    [Parameter(Mandatory=$false)]
    [string]$Weights = "runs/train/exp/weights/best.pt",
    [string]$Data = "../data/dataset.yaml",
    [int]$BatchSize = 32,
    [int]$ImgSize = 640,
    [string]$Task = "val",
    [switch]$Verbose
)

# Navigate to yolov5 directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$YoloDir = Join-Path $ProjectRoot "yolov5"

if (-not (Test-Path $YoloDir)) {
    Write-Error "YOLOv5 directory not found at $YoloDir"
    Write-Host "Please clone YOLOv5 first: git clone https://github.com/ultralytics/yolov5.git"
    exit 1
}

Push-Location $YoloDir

try {
    # Check if weights file exists
    if (-not (Test-Path $Weights)) {
        Write-Error "Weights file not found at $Weights"
        Write-Host "Please provide a valid path to model weights."
        Write-Host "Example: .\scripts\val.ps1 -Weights 'runs/train/exp/weights/best.pt'"
        exit 1
    }
    
    # Build the validation command
    $cmd = @("python", "val.py")
    $cmd += "--weights", $Weights
    $cmd += "--data", $Data
    $cmd += "--batch-size", $BatchSize
    $cmd += "--img", $ImgSize
    $cmd += "--task", $Task
    
    if ($Verbose) {
        $cmd += "--verbose"
    }
    
    Write-Host "Running YOLOv5 validation with command:"
    Write-Host ($cmd -join " ")
    Write-Host ""
    
    # Execute validation
    & $cmd[0] $cmd[1..($cmd.Length-1)]
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Validation complete! Check yolov5/runs/val/ for results."
    }
}
finally {
    Pop-Location
}
