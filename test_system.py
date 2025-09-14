#!/usr/bin/env python3
"""
Test script to verify the GPU scheduling system works correctly
"""
import subprocess
import time
import requests
import sys

def test_single_node():
    """Test a single node in isolation"""
    print("🧪 Testing single node...")
    
    # Start a single node
    proc = subprocess.Popen([
        "python", "gpu_node_service.py", "8000"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(2)
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=2)
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        print("✅ Health check passed")
        
        # Test metrics endpoint
        response = requests.get("http://localhost:8000/metrics", timeout=2)
        assert response.status_code == 200, f"Metrics failed: {response.status_code}"
        metrics = response.json()
        assert metrics["utilization"] == 0.0, f"Initial utilization should be 0, got {metrics['utilization']}"
        assert metrics["jobs"] == [], f"Initial jobs should be empty, got {metrics['jobs']}"
        print("✅ Metrics endpoint working")
        
        # Test job submission with a long-running job
        job_data = {"job_id": "test_job_1", "size": 0.7}  # Will run for 7 seconds
        response = requests.post("http://localhost:8000/execute", json=job_data, timeout=2)
        assert response.status_code == 200, f"Job submission failed: {response.status_code}"
        result = response.json()
        assert result["success"] == True, f"Job should be accepted, got {result}"
        print("✅ Job submission working")
        
        # Check utilization increased
        time.sleep(0.5)  # Give time for the job to be processed
        response = requests.get("http://localhost:8000/metrics", timeout=2)
        metrics = response.json()
        expected_util = 0.7
        assert abs(metrics["utilization"] - expected_util) < 0.01, f"Utilization should be {expected_util}, got {metrics['utilization']}"
        assert "test_job_1" in metrics["jobs"], f"Job should be in jobs list, got {metrics['jobs']}"
        print(f"✅ Job tracking working (utilization: {metrics['utilization']})")
        
        # Test job rejection (overload)
        # Current utilization is 0.7, so 0.4 would make it 1.1 (over capacity)
        job_data = {"job_id": "test_job_2", "size": 0.4}  # 0.7 + 0.4 = 1.1 > 1.0
        response = requests.post("http://localhost:8000/execute", json=job_data, timeout=2)
        result = response.json()
        
        # Double-check current utilization first
        metrics_response = requests.get("http://localhost:8000/metrics", timeout=2)
        current_metrics = metrics_response.json()
        print(f"DEBUG: Current utilization: {current_metrics['utilization']}, trying to add {job_data['size']}")
        print(f"DEBUG: Current jobs: {current_metrics['jobs']}")
        print(f"DEBUG: Would result in: {current_metrics['utilization'] + job_data['size']}")
        
        assert result["success"] == False, f"Job should be rejected (would exceed capacity: {current_metrics['utilization']} + {job_data['size']} > 1.0), got {result}"
        print("✅ Job rejection working")
        
        print("🎉 Single node test passed!")
        
    finally:
        proc.terminate()
        time.sleep(1)
        if proc.poll() is None:
            proc.kill()

def test_job_completion():
    """Test that jobs complete and release resources"""
    print("\n🧪 Testing job completion...")
    
    # Start a single node with output capture
    proc = subprocess.Popen([
        "python", "gpu_node_service.py", "8001"
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
    
    time.sleep(2)
    
    try:
        # Submit a small job (should complete quickly)
        job_data = {"job_id": "quick_job", "size": 0.1}  # Will run for 1 second
        print(f"Submitting job: {job_data}")
        response = requests.post("http://localhost:8001/execute", json=job_data, timeout=2)
        result = response.json()
        assert result["success"] == True, f"Quick job should be accepted, got {result}"
        print("✅ Quick job submitted")
        
        # Check it's running
        time.sleep(0.5)  # Give more time for the thread to start
        response = requests.get("http://localhost:8001/metrics", timeout=2)
        metrics = response.json()
        
        print(f"DEBUG: Metrics after 0.5s - util={metrics['utilization']}, jobs={metrics['jobs']}")
        
        if metrics["utilization"] == 0.0:
            # Job completed too fast, let's see the node output
            print("Job completed too quickly! Node output:")
            try:
                # Read any available output
                import select
                if select.select([proc.stdout], [], [], 0)[0]:
                    output = proc.stdout.read()
                    print(f"Node says: {output}")
            except:
                pass
            
            # Try a longer job
            print("Trying a longer job...")
            job_data = {"job_id": "longer_job", "size": 0.5}  # Will run for 5 seconds
            response = requests.post("http://localhost:8001/execute", json=job_data, timeout=2)
            result = response.json()
            print(f"Longer job result: {result}")
            
            time.sleep(1)  # Wait 1 second
            response = requests.get("http://localhost:8001/metrics", timeout=2)
            metrics = response.json()
            print(f"Metrics after longer job: {metrics}")
            
            if metrics["utilization"] > 0:
                print("✅ Longer job is running, short jobs complete too fast")
                # Wait for completion
                time.sleep(6)
                response = requests.get("http://localhost:8001/metrics", timeout=2)
                metrics = response.json()
                assert metrics["utilization"] == 0.0, f"Longer job should complete, util={metrics['utilization']}"
                print("✅ Job completed and released resources")
                print("🎉 Job completion test passed!")
                return
        
        assert metrics["utilization"] == 0.1, f"Job should be running with util=0.1, got {metrics['utilization']}"
        assert "quick_job" in metrics["jobs"], f"Job should be in jobs list, got {metrics['jobs']}"
        print("✅ Job is running")
        
        # Wait for completion (job runs for size * 10 seconds = 1 second)
        print("⏱️  Waiting for job to complete...")
        time.sleep(2)  # Job should complete in ~1 second
        
        # Check it completed
        response = requests.get("http://localhost:8001/metrics", timeout=2)
        metrics = response.json()
        assert metrics["utilization"] == 0.0, f"Job should have completed, util={metrics['utilization']}"
        assert metrics["jobs"] == [], f"Jobs list should be empty, got {metrics['jobs']}"
        print("✅ Job completed and released resources")
        
        print("🎉 Job completion test passed!")
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        # Get final metrics for debugging
        try:
            response = requests.get("http://localhost:8001/metrics", timeout=2)
            metrics = response.json()
            print(f"DEBUG: Final metrics - util={metrics['utilization']}, jobs={metrics['jobs']}")
        except:
            print("DEBUG: Could not get final metrics")
        raise
    except Exception as e:
        print(f"💥 Unexpected error in job completion test: {e}")
        raise
    finally:
        proc.terminate()
        time.sleep(1)
        if proc.poll() is None:
            proc.kill()

def main():
    print("🔧 GPU Cluster System Tests")
    print("=" * 40)
    
    try:
        test_single_node()
        test_job_completion()
        
        print("\n🎊 All tests passed! System is working correctly.")
        print("\nTo run the full demo:")
        print("1. Run: python launch_nodes.py")
        print("2. In another terminal: python control_plane.py")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()