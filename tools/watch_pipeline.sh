#!/bin/bash

echo "Real-Time Pipeline Log Viewer"
echo "============================="
echo "This will show live Docker logs from the running pipeline"
echo "Press Ctrl+C to exit"
echo ""

while true; do
    # Get current running container
    CONTAINER=$(docker ps --format "{{.ID}}" --filter ancestor=pipelines-detection | head -n 1)
    
    if [ -n "$CONTAINER" ]; then
        echo " Watching pipeline container: $CONTAINER"
        echo " Real-time logs:"
        echo "----------------------------------------"
        docker logs -f "$CONTAINER" 2>&1
        echo ""
        echo "Container finished. Waiting for next container..."
    else
        echo " No pipeline container running. Waiting..."
        sleep 5
    fi
done