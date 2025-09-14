cd C:\Users\mriga\Nebula\sage-project
sage-env\Scripts\activate

python launch_nodes.py

Wait for the health checks (“✓ Node {port} is healthy”) before proceeding.

Leave this terminal open! The nodes are now running and listening for jobs.

Step 2: Open a New Terminal for the Job Scheduler/Control Plane
Open a second terminal, activate your environment again, and cd to your project folder:

Step 3: Run the Control Plane Demo (Scheduler/Batch Submission)
Launch your job scheduling/testing script (often called control_plane.py or similar):

python control_plane.py
This will:
Discover all healthy nodes

Submit a batch of jobs according to your scenarios

Print job assignment, job success/failure, and live cluster metrics (utilization/job counts)

Run the cluster monitoring visualizer (bar charts in terminal)

Step 4: Observe Functionality
In the second terminal, watch console outputs as jobs are scheduled and the cluster status updates.

In the first terminal, watch the launcher and node logs—see job executions and utilization changes.


Step 5: (Optional) Interact with Node APIs Directly
Open a browser and visit FastAPI docs of any running node (for manual testing):
http://localhost:8000/docs
http://localhost:8001/docs
http://localhost:8002/docs
http://localhost:8003/docs


This lets you submit jobs, check metrics, and health interactively.

Step 6: Stopping the Cluster
When you’re done, return to the first (launcher) terminal and press Enter to cleanly shut down all running nodes.



To kill processes at a port

Even Quicker - Windows Command Line:
You can also just run these commands directly in your terminal:
cmdnetstat -ano | findstr :8000
taskkill /PID [PID_NUMBER] /F

netstat -ano | findstr :8001  
taskkill /PID [PID_NUMBER] /F

netstat -ano | findstr :8002
taskkill /PID [PID_NUMBER] /F

netstat -ano | findstr :8003
taskkill /PID [PID_NUMBER] /F