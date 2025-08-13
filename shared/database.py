"""
Shared database utilities for the 'Oumuamua simulator
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Generator, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection and management utilities"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("Database URL not provided")
    
    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Get a database connection with automatic cleanup"""
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    @contextmanager
    def get_cursor(self) -> Generator[psycopg2.extensions.cursor, None, None]:
        """Get a database cursor with automatic cleanup"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database operation error: {e}")
                raise
            finally:
                cursor.close()
    
    def execute_query(self, query: str, params: tuple = None) -> list:
        """Execute a query and return results"""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute_command(self, command: str, params: tuple = None) -> int:
        """Execute a command and return number of affected rows"""
        with self.get_cursor() as cursor:
            cursor.execute(command, params)
            return cursor.rowcount
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists"""
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = %s
        );
        """
        result = self.execute_query(query, (table_name,))
        return result[0]['exists']
    
    def create_tables_if_not_exist(self):
        """Create all necessary tables if they don't exist"""
        tables = [
            self._create_materials_table,
            self._create_orbital_states_table,
            self._create_simulation_results_table,
            self._create_observation_data_table
        ]
        
        for create_func in tables:
            try:
                create_func()
                logger.info(f"Table created successfully: {create_func.__name__}")
            except Exception as e:
                logger.error(f"Error creating table {create_func.__name__}: {e}")
    
    def _create_materials_table(self):
        """Create materials table"""
        query = """
        CREATE TABLE IF NOT EXISTS materials (
            id SERIAL PRIMARY KEY,
            material_type VARCHAR(50) UNIQUE NOT NULL,
            density REAL NOT NULL,
            heat_capacity REAL NOT NULL,
            sublimation_temperature REAL NOT NULL,
            thermal_conductivity REAL NOT NULL,
            tensile_strength REAL NOT NULL,
            albedo REAL NOT NULL,
            emissivity REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.execute_command(query)
    
    def _create_orbital_states_table(self):
        """Create orbital states table"""
        query = """
        CREATE TABLE IF NOT EXISTS orbital_states (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            position_x REAL NOT NULL,
            position_y REAL NOT NULL,
            position_z REAL NOT NULL,
            velocity_x REAL NOT NULL,
            velocity_y REAL NOT NULL,
            velocity_z REAL NOT NULL,
            acceleration_x REAL NOT NULL,
            acceleration_y REAL NOT NULL,
            acceleration_z REAL NOT NULL,
            simulation_id VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.execute_command(query)
    
    def _create_simulation_results_table(self):
        """Create simulation results table"""
        query = """
        CREATE TABLE IF NOT EXISTS simulation_results (
            id SERIAL PRIMARY KEY,
            simulation_id VARCHAR(100) UNIQUE NOT NULL,
            material_type VARCHAR(50) NOT NULL,
            object_shape VARCHAR(50) NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP NOT NULL,
            deviation_from_observed REAL,
            confidence_score REAL,
            parameters JSONB,
            results_summary JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.execute_command(query)
    
    def _create_observation_data_table(self):
        """Create observation data table"""
        query = """
        CREATE TABLE IF NOT EXISTS observation_data (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            position_x REAL NOT NULL,
            position_y REAL NOT NULL,
            position_z REAL NOT NULL,
            velocity_x REAL NOT NULL,
            velocity_y REAL NOT NULL,
            velocity_z REAL NOT NULL,
            uncertainty_position_x REAL,
            uncertainty_position_y REAL,
            uncertainty_position_z REAL,
            uncertainty_velocity_x REAL,
            uncertainty_velocity_y REAL,
            uncertainty_velocity_z REAL,
            observatory VARCHAR(100),
            instrument VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.execute_command(query)


# Global database manager instance
db_manager = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager


def init_database():
    """Initialize the database with all tables"""
    manager = get_db_manager()
    manager.create_tables_if_not_exist()
    logger.info("Database initialized successfully")
