# 'Oumuamua Simulator API Documentation

## Overview

The 'Oumuamua Simulator provides a comprehensive API for analyzing the anomalous acceleration of the 'Oumuamua object through physical modeling. The system uses a microservices architecture with a central orchestrator managing the simulation pipeline.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, this should be implemented.

## Endpoints

### Health Check

#### GET /health

Check the health status of the orchestrator and all microservices.

**Response:**
```json
{
  "status": "healthy",
  "service": "orchestrator",
  "timestamp": "2024-01-15T10:30:00",
  "services": {
    "ingestor": "healthy",
    "tidal_simulator": "healthy",
    "thermodynamic_modeler": "healthy",
    "outgassing_engine": "healthy",
    "trajectory_analyzer": "healthy"
  }
}
```

### Materials

#### GET /materials

Get list of available materials for simulation.

**Response:**
```json
[
  {
    "material_type": "solid_nitrogen",
    "density": 1026.0,
    "heat_capacity": 1040.0,
    "sublimation_temperature": 63.15,
    "description": "Solid nitrogen (N2) - proposed by Jackson & Desch (2021) as explanation for 'Oumuamua's anomalous acceleration"
  },
  {
    "material_type": "solid_hydrogen",
    "density": 86.0,
    "heat_capacity": 14300.0,
    "sublimation_temperature": 13.8,
    "description": "Solid hydrogen (H2) - alternative hypothesis for the object's composition"
  }
]
```

### Simulation

#### POST /simulate

Start a new simulation with the complete pipeline.

**Request Body:**
```json
{
  "material_type": "solid_nitrogen",
  "object_geometry": {
    "shape": "cigar",
    "dimensions": {
      "x": 100.0,
      "y": 10.0,
      "z": 10.0
    },
    "aspect_ratio": 10.0,
    "surface_area": 6283.0,
    "volume": 7854.0
  },
  "start_date": "2017-09-01T00:00:00",
  "end_date": "2018-02-01T00:00:00",
  "time_step_days": 1.0,
  "include_tidal_heating": true
}
```

**Response:**
```json
{
  "simulation_id": "sim_20240115_103000_a1b2c3d4",
  "status": "started",
  "message": "Simulation pipeline started in background"
}
```

#### GET /simulation/{simulation_id}

Get the status and results of a specific simulation.

**Response:**
```json
{
  "simulation_id": "sim_20240115_103000_a1b2c3d4",
  "status": "completed",
  "message": "Simulation completed successfully",
  "results": {
    "deviation_from_observed": 0.000123,
    "confidence_score": 0.987,
    "parameters": {
      "material_type": "solid_nitrogen",
      "object_geometry": {...},
      "start_date": "2017-09-01T00:00:00",
      "end_date": "2018-02-01T00:00:00"
    },
    "results_summary": {
      "total_thrust": 1.23e-6,
      "max_temperature": 245.6,
      "total_mass_loss": 0.00123
    }
  }
}
```

#### GET /simulations

List all simulations with their basic information.

**Response:**
```json
[
  {
    "simulation_id": "sim_20240115_103000_a1b2c3d4",
    "material_type": "solid_nitrogen",
    "object_shape": "cigar",
    "start_time": "2017-09-01T00:00:00",
    "end_time": "2018-02-01T00:00:00",
    "deviation_from_observed": 0.000123,
    "confidence_score": 0.987,
    "created_at": "2024-01-15T10:30:00"
  }
]
```

## Simulation Pipeline

The simulation follows this 6-step pipeline:

1. **Orbital Calculation** - Calculate base gravitational trajectory
2. **Tidal Forces** - Calculate tidal stresses on the object
3. **Thermal Modeling** - Calculate temperature distribution
4. **Outgassing** - Calculate sublimation and gas emission
5. **Thrust Analysis** - Calculate thrust from outgassing
6. **Trajectory Correction** - Apply thrust corrections to trajectory

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200` - Success
- `400` - Bad Request
- `404` - Not Found
- `500` - Internal Server Error

Error responses include a detail message:

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Rate Limiting

Currently, there are no rate limits implemented. In production, this should be added to prevent abuse.

## Examples

### Running a Nitrogen Iceberg Simulation

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

### Checking Simulation Status

```bash
curl "http://localhost:8000/simulation/sim_20240115_103000_a1b2c3d4"
```

### Getting Available Materials

```bash
curl "http://localhost:8000/materials"
```
