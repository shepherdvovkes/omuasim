#!/usr/bin/env python3
"""
Test script for 'Oumuamua Simulator
Tests basic functionality of the system
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_materials():
    """Test materials endpoint"""
    print("📊 Testing materials endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/materials", timeout=10)
        if response.status_code == 200:
            materials = response.json()
            print(f"✅ Found {len(materials)} materials:")
            for mat in materials:
                print(f"   - {mat['material_type']}: {mat['description'][:50]}...")
            return True
        else:
            print(f"❌ Materials request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Materials request error: {e}")
        return False

def test_simulation():
    """Test simulation endpoint"""
    print("🚀 Testing simulation endpoint...")
    
    # Test data for solid nitrogen simulation
    simulation_data = {
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
        "end_date": "2017-10-01T00:00:00",  # Shorter period for testing
        "time_step_days": 1.0,
        "include_tidal_heating": True
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/simulate",
            json=simulation_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            simulation_id = result['simulation_id']
            print(f"✅ Simulation started: {simulation_id}")
            
            # Wait a bit and check status
            print("⏳ Waiting for simulation to progress...")
            time.sleep(10)
            
            status_response = requests.get(f"{BASE_URL}/simulation/{simulation_id}", timeout=10)
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"📊 Simulation status: {status_data['status']}")
                print(f"📝 Message: {status_data['message']}")
                return True
            else:
                print(f"❌ Status check failed: {status_response.status_code}")
                return False
        else:
            print(f"❌ Simulation request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Simulation request error: {e}")
        return False

def test_simulations_list():
    """Test simulations list endpoint"""
    print("📋 Testing simulations list endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/simulations", timeout=10)
        if response.status_code == 200:
            simulations = response.json()
            print(f"✅ Found {len(simulations)} simulations in database")
            return True
        else:
            print(f"❌ Simulations list failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Simulations list error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing 'Oumuamua Simulator System")
    print("=" * 50)
    
    tests = [
        test_health,
        test_materials,
        test_simulation,
        test_simulations_list
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the system configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
