#!/bin/bash

# 'Oumuamua Simulator Startup Script

echo "🚀 Starting 'Oumuamua Simulator..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

echo "📦 Building and starting all services..."
docker-compose up --build -d

echo "⏳ Waiting for services to start..."
sleep 30

echo "🔍 Checking service health..."
curl -s http://localhost:8000/health | python3 -m json.tool

echo ""
echo "✅ 'Oumuamua Simulator is ready!"
echo ""
echo "🌐 API Documentation: http://localhost:8000/docs"
echo "🔗 Health Check: http://localhost:8000/health"
echo "📊 Available Materials: http://localhost:8000/materials"
echo ""
echo "📝 Example usage:"
echo "curl -X POST http://localhost:8000/simulate \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"material_type\": \"solid_nitrogen\","
echo "    \"object_geometry\": {"
echo "      \"shape\": \"cigar\","
echo "      \"dimensions\": {\"x\": 100.0, \"y\": 10.0, \"z\": 10.0},"
echo "      \"aspect_ratio\": 10.0,"
echo "      \"surface_area\": 6283.0,"
echo "      \"volume\": 7854.0"
echo "    },"
echo "    \"start_date\": \"2017-09-01T00:00:00\","
echo "    \"end_date\": \"2018-02-01T00:00:00\","
echo "    \"time_step_days\": 1.0,"
echo "    \"include_tidal_heating\": true"
echo "  }'"
echo ""
echo "🛑 To stop the simulator: docker-compose down"
