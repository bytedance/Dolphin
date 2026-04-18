import sys
import os
import time

print("=" * 50)
print("Starting Dolphin Demo...")
print("=" * 50, flush=True)

# Change to project directory
project_dir = r"E:\_MyPKM\10_Projects\11_Python\bytedance"
os.chdir(project_dir)
sys.path.insert(0, project_dir)

# Check if model files exist
model_dir = os.path.join(project_dir, "hf_model")
print(f"Model directory: {model_dir}", flush=True)
print(f"Files in model dir:", flush=True)
for f in os.listdir(model_dir):
    print(f"  - {f}", flush=True)

# Run the demo
print("\nRunning demo_page.py...", flush=True)
start_time = time.time()

# Import and run
from demo_page import main as demo_main
demo_main()

elapsed = time.time() - start_time
print(f"\nDone! Elapsed: {elapsed:.1f}s", flush=True)
