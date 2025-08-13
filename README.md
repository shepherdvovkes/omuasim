# 'Oumuamua Simulator

Architecture for simulating and analyzing the anomalous acceleration of the 'Oumuamua object through physical modeling.

## Overview

The 'Oumuamua Simulator is a comprehensive system that models the physical processes that could explain the anomalous acceleration of the 'Oumuamua object. The system implements a 6-step pipeline that simulates:

1. **Orbital Dynamics** - Base gravitational trajectory
2. **Tidal Forces** - Stresses on the object surface
3. **Thermal Modeling** - Temperature distribution from solar and tidal heating
4. **Sublimation** - Gas emission from heated surfaces
5. **Thrust Generation** - Force from outgassing
6. **Trajectory Correction** - Final trajectory with thrust effects

## Architecture

The system is built on a microservices architecture with a data processing pipeline:

### Core Components:

1. **Ingestor & Orbital Modeler** - Collection of observational data and construction of base orbit
2. **Material Properties DB** - Database of physical properties of materials
3. **Tidal Force Simulator** - Calculation of tidal forces and stresses
4. **Thermodynamic Modeler** - Object heating modeling
5. **Sublimation & Outgassing Engine** - Sublimation and outgassing calculations
6. **Thrust & Trajectory Analyzer** - Thrust analysis and trajectory correction
7. **Orchestrator & API Gateway** - Process management

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.8+ (for testing)
- curl (for API testing)

### Installation and Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd omuasim
   ```

2. **Start the system**
   ```bash
   ./start.sh
   ```
   
   Or manually:
   ```bash
   docker-compose up -d
   ```

3. **Wait for services to start** (about 30 seconds)

4. **Test the system**
   ```bash
   python3 test_system.py
   ```

### Access Points

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Available Materials**: http://localhost:8000/materials

## Usage Examples

### 1. Check System Health

```bash
curl http://localhost:8000/health
```

### 2. View Available Materials

```bash
curl http://localhost:8000/materials
```

### 3. Run a Nitrogen Iceberg Simulation

```bash
curl -X POST "http://localhost:8000/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "material_type": "solid_nitrogen",
    "object_geometry": {
      "shape": "cigar",
      "dimensions": {"x": 100.0, "y": 10.0, "z": 10.0},
      "aspect_ratio": 10.0,
      "surface_area": 6283.0,
      "volume": 7854.0
    },
    "start_date": "2017-09-01T00:00:00",
    "end_date": "2018-02-01T00:00:00",
    "time_step_days": 1.0,
    "include_tidal_heating": true
  }'
```

### 4. Check Simulation Status

```bash
curl "http://localhost:8000/simulation/sim_20240115_103000_a1b2c3d4"
```

### 5. List All Simulations

```bash
curl "http://localhost:8000/simulations"
```

## Project Structure

```
omuasim/
├── docker-compose.yml          # Container orchestration
├── start.sh                    # Startup script
├── test_system.py              # System test script
├── orchestrator/               # Central orchestrator
├── ingestor/                   # Data collector and orbital modeler
├── material-db/                # Materials database
├── tidal-simulator/            # Tidal forces simulator
├── thermodynamic-modeler/      # Thermodynamic model
├── outgassing-engine/          # Sublimation engine
├── trajectory-analyzer/        # Trajectory analyzer
├── shared/                     # Shared libraries and utilities
└── docs/                       # Documentation
    ├── API.md                  # API documentation
    └── ARCHITECTURE.md         # Architecture details
```

## Supported Materials

The simulator includes several material types with realistic physical properties:

- **Solid Nitrogen (N2)** - Proposed by Jackson & Desch (2021)
- **Solid Hydrogen (H2)** - Alternative hypothesis
- **Rock** - Typical asteroid material
- **Ice** - Water ice composition
- **Carbonaceous** - Dark, organic-rich material

## Simulation Pipeline

Each simulation follows this 6-step process:

1. **Orbital Calculation**: Calculate base gravitational trajectory using real observational data
2. **Tidal Forces**: Model tidal stresses on the object surface
3. **Thermal Modeling**: Calculate temperature distribution from solar and tidal heating
4. **Sublimation**: Model gas emission using Clausius-Clapeyron equation
5. **Thrust Analysis**: Convert outgassing to thrust vectors
6. **Trajectory Correction**: Apply thrust corrections and compare to observed trajectory

## API Reference

### Main Endpoints

- `GET /health` - System health check
- `GET /materials` - List available materials
- `POST /simulate` - Start new simulation
- `GET /simulation/{id}` - Get simulation status
- `GET /simulations` - List all simulations

### Detailed API Documentation

See [docs/API.md](docs/API.md) for complete API documentation.

## Technologies

- **Backend**: Python, FastAPI, NumPy, SciPy, Astropy
- **Database**: PostgreSQL
- **Computations**: C++/Fortran for critical computations
- **Orchestration**: Docker Compose, Kubernetes
- **API**: REST API with OpenAPI documentation

## Development

### Running in Development Mode

```bash
# Start services in development mode
docker-compose -f docker-compose.yml up --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Adding New Materials

1. Add material properties to `material-db/init.sql`
2. Update material descriptions in `orchestrator/main.py`
3. Restart the material-db service

### Adding New Physics Models

1. Implement new physics in the appropriate service
2. Update the shared models in `shared/models.py`
3. Add new endpoints to the service
4. Update the orchestrator to use the new functionality

## Troubleshooting

### Common Issues

1. **Services not starting**: Check Docker and Docker Compose installation
2. **Database connection errors**: Ensure PostgreSQL container is running
3. **Simulation failures**: Check service logs with `docker-compose logs`
4. **API timeouts**: Increase timeout values for long-running simulations

### Debugging

```bash
# View all service logs
docker-compose logs

# View specific service logs
docker-compose logs orchestrator

# Check service health
curl http://localhost:8000/health

# Test individual services
curl http://localhost:8001/health  # Ingestor
curl http://localhost:8002/health  # Tidal Simulator
curl http://localhost:8003/health  # Thermodynamic Modeler
curl http://localhost:8004/health  # Outgassing Engine
curl http://localhost:8005/health  # Trajectory Analyzer
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License

## Acknowledgments

This simulator is based on scientific research into the 'Oumuamua object, particularly the work of Jackson & Desch (2021) on the nitrogen iceberg hypothesis.
