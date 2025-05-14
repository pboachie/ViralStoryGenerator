#!/bin/bash
# Script to check the health of the containerized services

set -e

# Default values
ENVIRONMENT="prod"

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -e|--env)
            ENVIRONMENT="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-e|--env dev|prod]"
            exit 1
            ;;
    esac
done

# Determine which docker-compose file to use
if [ "$ENVIRONMENT" == "dev" ]; then
    COMPOSE_FILE="docker-compose.yml"
else
    COMPOSE_FILE="docker-compose.prod.yml"
fi

echo "==============================================="
echo "Checking service health for $ENVIRONMENT environment"
echo "==============================================="

# Check container status
echo "Container Status:"
docker-compose -f $COMPOSE_FILE ps

# Check Redis
echo -e "\n\nRedis Health Check:"
if docker-compose -f $COMPOSE_FILE exec redis redis-cli ping | grep -q "PONG"; then
    echo "✅ Redis is healthy"
else
    echo "❌ Redis is not responding"
fi

# Check queue size
echo -e "\n\nCrawl4AI Queue Status:"
docker-compose -f $COMPOSE_FILE exec redis redis-cli llen crawl4ai_queue

# Check system resources
echo -e "\n\nSystem Resources:"
echo "CPU Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Print logs summary for troubleshooting
echo -e "\n\nRecent Errors (last 10 from each service):"
echo "Backend errors:"
docker-compose -f $COMPOSE_FILE logs --tail=10 backend | grep -i "error\|exception"
echo "Scraper errors:"
docker-compose -f $COMPOSE_FILE logs --tail=10 scraper | grep -i "error\|exception"

echo -e "\n\nMonitoring Links:"
echo "- Grafana Dashboard: http://localhost:3000"
echo "- Prometheus: http://localhost:9090"

echo -e "\n\nHealth check complete!"