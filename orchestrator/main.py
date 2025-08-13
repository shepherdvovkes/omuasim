"""
Orchestrator & API Gateway Service
Manages the complete simulation pipeline for 'Oumuamua analysis
"""
import os
import sys
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import aiohttp
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

# Add shared module to path
sys.path.append('/app/shared')

from shared.models import SimulationParameters, SimulationResult, MaterialType, ObjectGeometry, Vector3D
from shared.database import get_db_manager, init_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="'Oumuamua Simulator Orchestrator",
    description="Central orchestrator for 'Oumuamua simulation pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_database()

# Service URLs
SERVICE_URLS = {
    'ingestor': os.getenv('INGESTOR_URL', 'http://ingestor:8001'),
    'tidal_simulator': os.getenv('TIDAL_SIM_URL', 'http://tidal-simulator:8002'),
    'thermodynamic_modeler': os.getenv('THERMO_URL', 'http://thermodynamic-modeler:8003'),
    'outgassing_engine': os.getenv('OUTGASSING_URL', 'http://outgassing-engine:8004'),
    'trajectory_analyzer': os.getenv('TRAJECTORY_URL', 'http://trajectory-analyzer:8005')
}


class SimulationRequest(BaseModel):
    """Request model for complete simulation"""
    material_type: str
    object_geometry: Dict[str, Any]
    start_date: str
    end_date: str
    time_step_days: float = 1.0
    include_tidal_heating: bool = True


class SimulationResponse(BaseModel):
    """Response model for simulation results"""
    simulation_id: str
    status: str
    message: str
    results: Optional[Dict[str, Any]] = None


class MaterialInfo(BaseModel):
    """Material information model"""
    material_type: str
    density: float
    heat_capacity: float
    sublimation_temperature: float
    description: str


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "orchestrator", 
        "timestamp": datetime.now(),
        "services": await check_service_health()
    }


@app.get("/materials")
async def get_available_materials() -> List[MaterialInfo]:
    """Get list of available materials for simulation"""
    try:
        db_manager = get_db_manager()
        query = "SELECT * FROM materials ORDER BY material_type"
        materials = db_manager.execute_query(query)
        
        material_info = []
        for mat in materials:
            info = MaterialInfo(
                material_type=mat['material_type'],
                density=mat['density'],
                heat_capacity=mat['heat_capacity'],
                sublimation_temperature=mat['sublimation_temperature'],
                description=get_material_description(mat['material_type'])
            )
            material_info.append(info)
        
        return material_info
        
    except Exception as e:
        logger.error(f"Error retrieving materials: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest, background_tasks: BackgroundTasks):
    """Run complete simulation pipeline"""
    try:
        # Generate unique simulation ID
        simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Start background task for complete simulation
        background_tasks.add_task(
            run_complete_simulation_pipeline,
            simulation_id,
            request
        )
        
        return SimulationResponse(
            simulation_id=simulation_id,
            status="started",
            message="Simulation pipeline started in background"
        )
        
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/simulation/{simulation_id}")
async def get_simulation_status(simulation_id: str) -> SimulationResponse:
    """Get simulation status and results"""
    try:
        db_manager = get_db_manager()
        query = """
        SELECT * FROM simulation_results 
        WHERE simulation_id = %s
        """
        result = db_manager.execute_query(query, (simulation_id,))
        
        if not result:
            return SimulationResponse(
                simulation_id=simulation_id,
                status="running",
                message="Simulation in progress"
            )
        
        sim_data = result[0]
        return SimulationResponse(
            simulation_id=simulation_id,
            status="completed",
            message="Simulation completed successfully",
            results={
                "deviation_from_observed": sim_data['deviation_from_observed'],
                "confidence_score": sim_data['confidence_score'],
                "parameters": sim_data['parameters'],
                "results_summary": sim_data['results_summary']
            }
        )
        
    except Exception as e:
        logger.error(f"Error retrieving simulation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/simulations")
async def list_simulations() -> List[Dict[str, Any]]:
    """List all simulations"""
    try:
        db_manager = get_db_manager()
        query = """
        SELECT simulation_id, material_type, object_shape, start_time, end_time,
               deviation_from_observed, confidence_score, created_at
        FROM simulation_results 
        ORDER BY created_at DESC
        LIMIT 50
        """
        results = db_manager.execute_query(query)
        
        return [
            {
                "simulation_id": r['simulation_id'],
                "material_type": r['material_type'],
                "object_shape": r['object_shape'],
                "start_time": r['start_time'].isoformat(),
                "end_time": r['end_time'].isoformat(),
                "deviation_from_observed": r['deviation_from_observed'],
                "confidence_score": r['confidence_score'],
                "created_at": r['created_at'].isoformat()
            }
            for r in results
        ]
        
    except Exception as e:
        logger.error(f"Error listing simulations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_complete_simulation_pipeline(simulation_id: str, request: SimulationRequest):
    """Run the complete simulation pipeline"""
    try:
        logger.info(f"Starting complete simulation pipeline for {simulation_id}")
        
        # Step 1: Fetch observational data and calculate base orbit
        logger.info("Step 1: Calculating base orbital trajectory")
        orbital_data = await calculate_base_orbit(simulation_id, request)
        
        # Step 2: Calculate tidal forces
        logger.info("Step 2: Calculating tidal forces")
        tidal_data = await calculate_tidal_forces(simulation_id, request)
        
        # Step 3: Calculate heating and temperature distribution
        logger.info("Step 3: Calculating temperature distribution")
        temperature_data = await calculate_temperature_distribution(simulation_id, request)
        
        # Step 4: Calculate outgassing
        logger.info("Step 4: Calculating outgassing")
        outgassing_data = await calculate_outgassing(simulation_id, request, temperature_data)
        
        # Step 5: Analyze trajectory with thrust corrections
        logger.info("Step 5: Analyzing trajectory with thrust corrections")
        trajectory_data = await analyze_trajectory(simulation_id, orbital_data, outgassing_data)
        
        # Step 6: Calculate final results and store
        logger.info("Step 6: Calculating final results")
        await calculate_final_results(simulation_id, request, trajectory_data)
        
        logger.info(f"Completed simulation pipeline for {simulation_id}")
        
    except Exception as e:
        logger.error(f"Error in simulation pipeline: {e}")
        # Store error in database
        await store_simulation_error(simulation_id, str(e))


async def calculate_base_orbit(simulation_id: str, request: SimulationRequest) -> Dict[str, Any]:
    """Calculate base orbital trajectory"""
    async with aiohttp.ClientSession() as session:
        url = f"{SERVICE_URLS['ingestor']}/simulate-orbit"
        data = {
            "start_date": request.start_date,
            "end_date": request.end_date,
            "time_step_days": request.time_step_days
        }
        
        async with session.post(url, json=data) as response:
            if response.status != 200:
                raise Exception(f"Orbital calculation failed: {response.status}")
            
            result = await response.json()
            return result


async def calculate_tidal_forces(simulation_id: str, request: SimulationRequest) -> Dict[str, Any]:
    """Calculate tidal forces"""
    async with aiohttp.ClientSession() as session:
        url = f"{SERVICE_URLS['tidal_simulator']}/calculate-tidal-forces"
        data = {
            "simulation_id": simulation_id,
            "object_geometry": request.object_geometry,
            "start_time": request.start_date,
            "end_time": request.end_date
        }
        
        async with session.post(url, json=data) as response:
            if response.status != 200:
                raise Exception(f"Tidal force calculation failed: {response.status}")
            
            result = await response.json()
            return result


async def calculate_temperature_distribution(simulation_id: str, request: SimulationRequest) -> Dict[str, Any]:
    """Calculate temperature distribution"""
    async with aiohttp.ClientSession() as session:
        url = f"{SERVICE_URLS['thermodynamic_modeler']}/calculate-heating"
        data = {
            "simulation_id": simulation_id,
            "material_type": request.material_type,
            "object_geometry": request.object_geometry,
            "include_tidal_heating": request.include_tidal_heating
        }
        
        async with session.post(url, json=data) as response:
            if response.status != 200:
                raise Exception(f"Temperature calculation failed: {response.status}")
            
            result = await response.json()
            return result


async def calculate_outgassing(simulation_id: str, request: SimulationRequest, 
                            temperature_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate outgassing"""
    async with aiohttp.ClientSession() as session:
        url = f"{SERVICE_URLS['outgassing_engine']}/calculate-outgassing"
        data = {
            "simulation_id": simulation_id,
            "material_type": request.material_type,
            "temperature_data": temperature_data
        }
        
        async with session.post(url, json=data) as response:
            if response.status != 200:
                raise Exception(f"Outgassing calculation failed: {response.status}")
            
            result = await response.json()
            return result


async def analyze_trajectory(simulation_id: str, orbital_data: Dict[str, Any], 
                          outgassing_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze trajectory with thrust corrections"""
    async with aiohttp.ClientSession() as session:
        url = f"{SERVICE_URLS['trajectory_analyzer']}/analyze-trajectory"
        data = {
            "simulation_id": simulation_id,
            "outgassing_data": outgassing_data,
            "base_trajectory": orbital_data
        }
        
        async with session.post(url, json=data) as response:
            if response.status != 200:
                raise Exception(f"Trajectory analysis failed: {response.status}")
            
            result = await response.json()
            return result


async def calculate_final_results(simulation_id: str, request: SimulationRequest, 
                               trajectory_data: Dict[str, Any]):
    """Calculate final results and store in database"""
    try:
        # Extract deviation from trajectory data
        deviation = trajectory_data.get('deviation_analysis', {}).get('average_deviation', 0.0)
        
        # Calculate confidence score (simplified)
        confidence_score = max(0.0, 1.0 - deviation * 10.0)  # Simplified confidence calculation
        
        # Store results in database
        db_manager = get_db_manager()
        query = """
        INSERT INTO simulation_results 
        (simulation_id, material_type, object_shape, start_time, end_time,
         deviation_from_observed, confidence_score, parameters, results_summary)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        import json
        params = (
            simulation_id,
            request.material_type,
            request.object_geometry.get('shape', 'unknown'),
            request.start_date,
            request.end_date,
            deviation,
            confidence_score,
            json.dumps(request.dict()),
            json.dumps(trajectory_data)
        )
        
        db_manager.execute_command(query, params)
        
        logger.info(f"Stored final results for simulation {simulation_id}")
        
    except Exception as e:
        logger.error(f"Error storing final results: {e}")
        raise


async def store_simulation_error(simulation_id: str, error_message: str):
    """Store simulation error in database"""
    try:
        db_manager = get_db_manager()
        query = """
        INSERT INTO simulation_results 
        (simulation_id, material_type, object_shape, start_time, end_time,
         deviation_from_observed, confidence_score, parameters, results_summary)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        import json
        params = (
            simulation_id,
            "error",
            "error",
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            0.0,
            0.0,
            json.dumps({"error": error_message}),
            json.dumps({"status": "error", "message": error_message})
        )
        
        db_manager.execute_command(query, params)
        
    except Exception as e:
        logger.error(f"Error storing simulation error: {e}")


async def check_service_health() -> Dict[str, str]:
    """Check health of all services"""
    health_status = {}
    
    for service_name, url in SERVICE_URLS.items():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/health", timeout=5) as response:
                    if response.status == 200:
                        health_status[service_name] = "healthy"
                    else:
                        health_status[service_name] = "unhealthy"
        except Exception as e:
            health_status[service_name] = f"error: {str(e)}"
    
    return health_status


def get_material_description(material_type: str) -> str:
    """Get description for material type"""
    descriptions = {
        "solid_nitrogen": "Solid nitrogen (N2) - proposed by Jackson & Desch (2021) as explanation for 'Oumuamua's anomalous acceleration",
        "solid_hydrogen": "Solid hydrogen (H2) - alternative hypothesis for the object's composition",
        "rock": "Typical asteroid material - silicate rock composition",
        "ice": "Water ice - cometary composition",
        "carbonaceous": "Carbonaceous material - dark, organic-rich composition typical of some asteroids"
    }
    
    return descriptions.get(material_type, "Unknown material type")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
