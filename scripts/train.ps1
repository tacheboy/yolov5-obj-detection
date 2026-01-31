# YOLOv5 Training Wrapper Script
# Run from project root: .\scripts\train.ps1
#
# This script trains a YOLOv5 model on the dataset defined in data/dummy/dataset.yaml
# Adjust hyperparameters below as needed.

param(
    [int]$Epochs = 50,
    [int]$BatchSize = 16,
    [int]$ImgSize = 640,
    [string]$Weights = "yolov5s.pt",
    [string]$Data = "../data/dummy/dataset.yaml",
    [string]$Name = "",
    [switch]$Resume,
    [switch]$NoAugment,
    [string]$Device = "auto"
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
    # Resolve device
    $ResolvedDevice = $Device
    if ($Device -eq "auto") {
        try {
            $CudaAvailable = & python -c "import torch; print(torch.cuda.is_available())" 2>$null
        } catch {
            $CudaAvailable = ""
        }
        if ($CudaAvailable -match "True") {
            $ResolvedDevice = "0"
        } else {
            $ResolvedDevice = "cpu"
        }
    }

    # Build the training command
    $cmd = @("python", "train.py")
    $cmd += "--img", $ImgSize
    $cmd += "--batch", $BatchSize
    $cmd += "--epochs", $Epochs
    $cmd += "--data", $Data
    $cmd += "--weights", $Weights
    $cmd += "--device", $ResolvedDevice
    
    if ($Name) {
        $cmd += "--name", $Name
    }
    
    if ($Resume) {
        $cmd += "--resume"
    }
    
    if ($NoAugment) {
        # Use the no-augmentation hyperparameters
        $cmd += "--hyp", "data/hyps/hyp.no-augmentation.yaml"
    }
    
    Write-Host "Running YOLOv5 training with command:"
    Write-Host ($cmd -join " ")
    Write-Host ""
    
    # Execute training
    & $cmd[0] $cmd[1..($cmd.Length-1)]
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Training complete! Check yolov5/runs/train/ for results."
    }
}
finally {
    Pop-Location
}
