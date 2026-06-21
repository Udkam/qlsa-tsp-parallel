param(
    [Parameter(Mandatory = $true)]
    [string]$KeyPath,
    [Parameter(Mandatory = $true)]
    [string]$Vm1,
    [Parameter(Mandatory = $true)]
    [string]$Vm2,
    [string]$RemoteProject = "~/parallel-algorithm"
)

$ErrorActionPreference = "Stop"

$files = @(
    "data/ch130.tsp",
    "data/a280.tsp",
    "data/lin318.tsp",
    "data/rat575.tsp",
    "data/dsj1000.tsp",
    "scripts/run_large_mpi_vm.py",
    "scripts/analyze_large_mpi_vm.py",
    "configs/large_tsplib_instances.json"
)

foreach ($vm in @($Vm1, $Vm2)) {
    ssh -i $KeyPath $vm "mkdir -p $RemoteProject/data $RemoteProject/scripts $RemoteProject/configs"
    foreach ($file in $files) {
        if (-not (Test-Path -LiteralPath $file)) {
            Write-Warning "Missing local file: $file"
            continue
        }
        $remoteDir = Split-Path $file -Parent
        scp -i $KeyPath $file "${vm}:$RemoteProject/$remoteDir/"
    }
}
