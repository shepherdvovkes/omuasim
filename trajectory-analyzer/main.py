"""
Thrust & Trajectory Analyzer Service
Calculates thrust from outgassing and applies it to correct orbital trajectory
"""
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np
from scipy.integrate import solve_ivp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import requests

# Add shared module to path
sys.path.append('/app/shared')

from shared.models import ThrustVector, Vector3D, OrbitalState, OutgassingData, SimulationResult
from shared.database import get_db_manager, init_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Thrust & Trajectory Analyzer", version="1.0.0")

# Initialize database
init_database()


class TrajectoryAnalysisRequest(BaseModel):
    """Request model for trajectory analysis"""
    simulation_id: str
    outgassing_data: Dict[str, Any]
    base_trajectory: Dict[str, Any]


class TrajectoryAnalysisResponse(BaseModel):
    """Response model for trajectory analysis"""
    simulation_id: str
    corrected_trajectory: List[Dict[str, Any]]
    thrust_vectors: List[Dict[str, Any]]
    deviation_analysis: Dict[str, Any]
    message: str


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "trajectory-analyzer", "timestamp": datetime.now()}


@app.post("/analyze-trajectory", response_model=TrajectoryAnalysisResponse)
async def analyze_trajectory(request: TrajectoryAnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze trajectory with thrust corrections"""
    try:
        # Start background task for trajectory analysis
        background_tasks.add_task(
            calculate_corrected_trajectory,
            request.simulation_id,
            request.outgassing_data,
            request.base_trajectory
        )
        
        return TrajectoryAnalysisResponse(
            simulation_id=request.simulation_id,
            corrected_trajectory=[],
            thrust_vectors=[],
            deviation_analysis={},
            message="Trajectory analysis started in background"
        )
        
    except Exception as e:
        logger.error(f"Error in trajectory analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trajectory-data/{simulation_id}")
async def get_trajectory_data(simulation_id: str):
    """Get trajectory analysis data for a specific simulation"""
    try:
        # This would retrieve stored trajectory data
        # For now, return a placeholder
        return {
            "simulation_id": simulation_id,
            "corrected_trajectory": [],
            "thrust_vectors": [],
            "deviation_analysis": {},
            "message": "Trajectory data retrieval not yet implemented"
        }
        
    except Exception as e:
        logger.error(f"Error retrieving trajectory data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def calculate_corrected_trajectory(simulation_id: str, outgassing_data: Dict[str, Any], 
                                base_trajectory: Dict[str, Any]):
    """Calculate corrected trajectory with thrust effects"""
    try:
        logger.info(f"Starting trajectory analysis for {simulation_id}")
        
        # Parse base trajectory
        base_states = parse_orbital_states(base_trajectory)
        
        # Parse outgassing data
        outgassing_states = parse_outgassing_data(outgassing_data)
        
        # Calculate thrust vectors from outgassing
        thrust_vectors = calculate_thrust_vectors(outgassing_states)
        
        # Apply thrust corrections to trajectory
        corrected_states = apply_thrust_corrections(base_states, thrust_vectors)
        
        # Calculate deviation from observed trajectory
        deviation = calculate_deviation_from_observed(corrected_states)
        
        # Store results (would implement database storage)
        logger.info(f"Completed trajectory analysis for {simulation_id}")
        logger.info(f"Deviation from observed: {deviation:.6f} AU")
        
    except Exception as e:
        logger.error(f"Error in trajectory calculation: {e}")


def parse_orbital_states(trajectory_data: Dict[str, Any]) -> List[OrbitalState]:
    """Parse orbital states from trajectory data"""
    states = []
    
    for state_data in trajectory_data["orbital_states"]:
        state = OrbitalState(
            timestamp=datetime.fromisoformat(state_data["timestamp"]),
            position=Vector3D(**state_data["position"]),
            velocity=Vector3D(**state_data["velocity"]),
            acceleration=Vector3D(**state_data["acceleration"])
        )
        states.append(state)
    
    return states


def parse_outgassing_data(outgassing_data: Dict[str, Any]) -> List[OutgassingData]:
    """Parse outgassing data"""
    outgassing_states = []
    
    for data in outgassing_data["outgassing_states"]:
        outgassing = OutgassingData(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            gas_flux=np.array(data["gas_flux"]),
            velocity_vectors=np.array(data["velocity_vectors"]),
            total_mass_loss=data["total_mass_loss"],
            surface_points=np.array(data["surface_points"]),
            material_type=data["material_type"]
        )
        outgassing_states.append(outgassing)
    
    return outgassing_states


def calculate_thrust_vectors(outgassing_states: List[OutgassingData]) -> List[ThrustVector]:
    """Calculate thrust vectors from outgassing data"""
    thrust_vectors = []
    
    for outgassing in outgassing_states:
        # Calculate total thrust magnitude
        total_thrust = calculate_total_thrust(outgassing)
        
        # Calculate thrust direction (simplified - would be more complex in reality)
        thrust_direction = calculate_thrust_direction(outgassing)
        
        # Calculate origin point (center of mass)
        origin_point = Vector3D(0.0, 0.0, 0.0)  # Simplified
        
        thrust_vector = ThrustVector(
            timestamp=outgassing.timestamp,
            force=Vector3D(
                total_thrust * thrust_direction[0],
                total_thrust * thrust_direction[1],
                total_thrust * thrust_direction[2]
            ),
            magnitude=total_thrust,
            direction=Vector3D(*thrust_direction),
            origin_point=origin_point
        )
        
        thrust_vectors.append(thrust_vector)
    
    return thrust_vectors


def calculate_total_thrust(outgassing: OutgassingData) -> float:
    """Calculate total thrust magnitude from outgassing"""
    
    # Thrust = mass flow rate * exhaust velocity
    # F = ṁ * v
    
    # Average exhaust velocity
    velocities = [np.linalg.norm(v) for v in outgassing.velocity_vectors]
    avg_velocity = np.mean(velocities) if velocities else 0.0
    
    # Mass flow rate (kg/s)
    mass_flow_rate = outgassing.total_mass_loss
    
    # Thrust (N)
    thrust = mass_flow_rate * avg_velocity
    
    return thrust


def calculate_thrust_direction(outgassing: OutgassingData) -> np.ndarray:
    """Calculate thrust direction vector"""
    
    # Simplified: assume thrust is in the direction of the Sun
    # In reality, this would depend on the surface normal and gas flow patterns
    
    # For now, use a simplified model where thrust opposes the Sun direction
    # This is a rough approximation
    
    # Calculate center of mass of the object
    com = np.mean(outgassing.surface_points, axis=0)
    
    # Assume thrust direction is radially outward from COM
    # This is a very simplified model
    thrust_direction = com / (np.linalg.norm(com) + 1e-10)
    
    return thrust_direction


def apply_thrust_corrections(base_states: List[OrbitalState], 
                          thrust_vectors: List[ThrustVector]) -> List[OrbitalState]:
    """Apply thrust corrections to base trajectory"""
    
    corrected_states = []
    
    # Match thrust vectors to orbital states by timestamp
    thrust_dict = {tv.timestamp: tv for tv in thrust_vectors}
    
    for base_state in base_states:
        # Get thrust vector for this time (if any)
        thrust_vector = thrust_dict.get(base_state.timestamp)
        
        if thrust_vector:
            # Apply thrust correction
            corrected_state = apply_thrust_at_time(base_state, thrust_vector)
        else:
            # No thrust at this time
            corrected_state = base_state
        
        corrected_states.append(corrected_state)
    
    return corrected_states


def apply_thrust_at_time(base_state: OrbitalState, thrust_vector: ThrustVector) -> OrbitalState:
    """Apply thrust correction at a specific time"""
    
    # Object mass (simplified - would vary with mass loss)
    object_mass = 1e6  # kg (1 million kg - typical for 'Oumuamua)
    
    # Thrust acceleration (F = ma, so a = F/m)
    thrust_acceleration = Vector3D(
        thrust_vector.force.x / object_mass,
        thrust_vector.force.y / object_mass,
        thrust_vector.force.z / object_mass
    )
    
    # Add thrust acceleration to gravitational acceleration
    total_acceleration = Vector3D(
        base_state.acceleration.x + thrust_acceleration.x,
        base_state.acceleration.y + thrust_acceleration.y,
        base_state.acceleration.z + thrust_acceleration.z
    )
    
    # Create corrected orbital state
    corrected_state = OrbitalState(
        timestamp=base_state.timestamp,
        position=base_state.position,
        velocity=base_state.velocity,
        acceleration=total_acceleration
    )
    
    return corrected_state


def calculate_deviation_from_observed(corrected_states: List[OrbitalState]) -> float:
    """Calculate deviation from observed trajectory"""
    
    # Get observed trajectory data from database
    db_manager = get_db_manager()
    query = """
    SELECT * FROM observation_data 
    ORDER BY timestamp
    """
    observed_data = db_manager.execute_query(query)
    
    if not observed_data:
        logger.warning("No observed data found for deviation calculation")
        return 0.0
    
    # Calculate deviation at each observation point
    deviations = []
    
    for obs in observed_data:
        obs_time = obs['timestamp']
        obs_position = Vector3D(
            obs['position_x'],
            obs['position_y'],
            obs['position_z']
        )
        
        # Find closest simulated state
        closest_state = None
        min_time_diff = float('inf')
        
        for state in corrected_states:
            time_diff = abs((state.timestamp - obs_time).total_seconds())
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_state = state
        
        if closest_state:
            # Calculate position deviation
            position_diff = np.sqrt(
                (closest_state.position.x - obs_position.x)**2 +
                (closest_state.position.y - obs_position.y)**2 +
                (closest_state.position.z - obs_position.z)**2
            )
            deviations.append(position_diff)
    
    # Return average deviation
    return np.mean(deviations) if deviations else 0.0


def integrate_trajectory_with_thrust(initial_state: OrbitalState, 
                                  thrust_vectors: List[ThrustVector],
                                  time_span: timedelta) -> List[OrbitalState]:
    """Integrate trajectory with thrust using numerical methods"""
    
    # This would implement proper orbital integration
    # For now, return simplified result
    
    # Convert to AU/day² for acceleration
    au_per_day_squared = 1.496e11 / (86400**2)  # AU/day²
    
    # Simplified integration (Euler method)
    states = [initial_state]
    dt = 1.0  # 1 day time step
    
    current_state = initial_state
    current_time = initial_state.timestamp
    
    while current_time < initial_state.timestamp + time_span:
        # Find thrust at current time
        thrust = None
        for tv in thrust_vectors:
            if abs((tv.timestamp - current_time).total_seconds()) < 86400:  # Within 1 day
                thrust = tv
                break
        
        # Update velocity (v = v₀ + a*dt)
        new_velocity = Vector3D(
            current_state.velocity.x + current_state.acceleration.x * dt,
            current_state.velocity.y + current_state.acceleration.y * dt,
            current_state.velocity.z + current_state.acceleration.z * dt
        )
        
        # Update position (r = r₀ + v*dt)
        new_position = Vector3D(
            current_state.position.x + new_velocity.x * dt,
            current_state.position.y + new_velocity.y * dt,
            current_state.position.z + new_velocity.z * dt
        )
        
        # Update time
        current_time += timedelta(days=dt)
        
        # Create new state
        new_state = OrbitalState(
            timestamp=current_time,
            position=new_position,
            velocity=new_velocity,
            acceleration=current_state.acceleration  # Simplified
        )
        
        states.append(new_state)
        current_state = new_state
    
    return states


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
