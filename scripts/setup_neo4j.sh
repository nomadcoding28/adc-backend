#!/usr/bin/env bash
# scripts/setup_neo4j.sh — Set up Neo4j with Docker for the ACD Framework
# Usage: bash scripts/setup_neo4j.sh [--reset]
set -euo pipefail

NEO4J_VERSION="${NEO4J_VERSION:-5.15.0}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-acd_framework_2026}"
CONTAINER_NAME="acd-neo4j"
HTTP_PORT=7474
BOLT_PORT=7687

echo "=== ACD Framework — Neo4j Setup ==="
echo "Neo4j version: ${NEO4J_VERSION}"
echo "Container:     ${CONTAINER_NAME}"
echo "Bolt port:     ${BOLT_PORT}"

# Reset if requested
if [[ "${1:-}" == "--reset" ]]; then
    echo "Resetting: removing existing container..."
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    docker volume rm acd_neo4j_data 2>/dev/null || true
fi

# Check if already running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Neo4j is already running."
    echo "  Browser: http://localhost:${HTTP_PORT}"
    echo "  Bolt:    bolt://localhost:${BOLT_PORT}"
    exit 0
fi

# Create volume
docker volume create acd_neo4j_data 2>/dev/null || true

# Run Neo4j with APOC
echo "Starting Neo4j container..."
docker run -d \
    --name "${CONTAINER_NAME}" \
    -p "${HTTP_PORT}:7474" \
    -p "${BOLT_PORT}:7687" \
    -v acd_neo4j_data:/data \
    -e "NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}" \
    -e "NEO4J_PLUGINS=[\"apoc\"]" \
    -e "NEO4J_dbms_security_procedures_unrestricted=apoc.*" \
    -e "NEO4J_dbms_security_procedures_allowlist=apoc.*" \
    "neo4j:${NEO4J_VERSION}"

# Wait for startup
echo "Waiting for Neo4j to start..."
for i in $(seq 1 30); do
    if curl -s "http://localhost:${HTTP_PORT}" > /dev/null 2>&1; then
        echo "Neo4j is ready!"
        break
    fi
    sleep 2
    echo "  Waiting... (${i}/30)"
done

echo ""
echo "=== Neo4j Setup Complete ==="
echo "  Browser: http://localhost:${HTTP_PORT}"
echo "  Bolt:    bolt://localhost:${BOLT_PORT}"
echo "  User:    neo4j"
echo "  Pass:    ${NEO4J_PASSWORD}"
echo ""
echo "To apply the ACD schema:"
echo "  python -c \"from knowledge.neo4j_schema import apply_schema; from neo4j import GraphDatabase; d=GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j','${NEO4J_PASSWORD}')); apply_schema(d); d.close()\""
