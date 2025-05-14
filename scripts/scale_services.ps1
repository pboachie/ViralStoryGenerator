# PowerShell script to scale the Docker services for ViralStoryGenerator

# Default values
$backendReplicas = 2
$scraperReplicas = 3
$environment = "prod"

# Parse arguments
param(
    [Alias("b")]
    [int]$Backend = 2,

    [Alias("s")]
    [int]$Scraper = 3,

    [Alias("e")]
    [ValidateSet("dev", "prod")]
    [string]$Env = "prod",

    [Alias("h")]
    [switch]$Help = $false
)

# Help function
function Show-Help {
    Write-Host "Usage: .\scale_services.ps1 [options]"
    Write-Host "Options:"
    Write-Host "  -Backend, -b NUMBER   Number of backend replicas (default: 2)"
    Write-Host "  -Scraper, -s NUMBER   Number of scraper replicas (default: 3)"
    Write-Host "  -Env, -e ENV          Environment: dev or prod (default: prod)"
    Write-Host "  -Help, -h             Show this help message"
    exit 0
}

# If help flag is present, show help
if ($Help) {
    Show-Help
}

# Use the parameters
$backendReplicas = $Backend
$scraperReplicas = $Scraper
$environment = $Env

# Determine which docker-compose file to use
$composeFile = if ($environment -eq "dev") { "docker-compose.yml" } else { "docker-compose.prod.yml" }

Write-Host "Scaling services in $environment environment:"
Write-Host "- Backend: $backendReplicas replicas"
Write-Host "- Scraper: $scraperReplicas replicas"
Write-Host "Using compose file: $composeFile"

# Export variables for docker-compose
$env:BACKEND_REPLICAS = $backendReplicas
$env:SCRAPER_REPLICAS = $scraperReplicas

# Scale services
try {
    docker-compose -f $composeFile up -d --scale backend=$backendReplicas --scale scraper=$scraperReplicas

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Services scaled successfully!"
        Write-Host "Run 'docker-compose -f $composeFile ps' to see the current status."
    } else {
        Write-Host "Error scaling services. Exit code: $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "An error occurred: $_" -ForegroundColor Red
}