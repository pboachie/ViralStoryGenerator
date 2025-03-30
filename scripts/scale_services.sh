#!/bin/bash
# Script to scale the Docker services for ViralStoryGenerator

set -e

# Default values
BACKEND_REPLICAS=2
SCRAPER_REPLICAS=3
ENVIRONMENT="prod"

# Help function
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -b, --backend NUMBER   Number of backend replicas (default: 2)"
    echo "  -s, --scraper NUMBER   Number of scraper replicas (default: 3)"
    echo "  -e, --env ENV          Environment: dev or prod (default: prod)"
    echo "  -h, --help             Show this help message"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -b|--backend)
            BACKEND_REPLICAS="$2"
            shift
            shift
            ;;
        -s|--scraper)
            SCRAPER_REPLICAS="$2"
            shift
            shift
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Validate environment
if [ "$ENVIRONMENT" != "dev" ] && [ "$ENVIRONMENT" != "prod" ]; then
    echo "Error: Environment must be 'dev' or 'prod'"
    exit 1
fi

# Determine which docker-compose file to use
if [ "$ENVIRONMENT" == "dev" ]; then
    COMPOSE_FILE="docker-compose.yml"
else
    COMPOSE_FILE="docker-compose.prod.yml"
fi

echo "Scaling services in $ENVIRONMENT environment:"
echo "- Backend: $BACKEND_REPLICAS replicas"
echo "- Scraper: $SCRAPER_REPLICAS replicas"
echo "Using compose file: $COMPOSE_FILE"

# Export variables for docker-compose
export BACKEND_REPLICAS=$BACKEND_REPLICAS
export SCRAPER_REPLICAS=$SCRAPER_REPLICAS

# Scale services
docker-compose -f $COMPOSE_FILE up -d --scale backend=$BACKEND_REPLICAS --scale scraper=$SCRAPER_REPLICAS

echo "Services scaled successfully!"
echo "Run 'docker-compose -f $COMPOSE_FILE ps' to see the current status."