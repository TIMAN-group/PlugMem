import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ["PYTHONPATH"] = str(project_root)

def load_env():
    env_file = project_root / ".env"
    if not env_file.exists():
        print("No .env file found at project root.", flush=True)
        return

    print("Loading environment variables from .env...", flush=True)
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Handle "export KEY=VAL"
            if line.startswith("export "):
                line = line[7:]
            
            if "=" not in line:
                continue
            
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            
            # Strip quotes
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            elif val.startswith("'") and val.endswith("'"):
                val = val[1:-1]
                
            os.environ[key] = val
            print(f"  {key}={val}", flush=True)

if __name__ == "__main__":
    load_env()
    
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    elif "PORT" in os.environ:
        try:
            port = int(os.environ["PORT"])
        except ValueError:
            pass

    # Import uvicorn inside so env variables are already set
    import uvicorn
    uvicorn.run("plugmem.api.app:app", host="0.0.0.0", port=port, reload=True)
