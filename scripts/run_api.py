from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

if __name__ == "__main__":
    # Local dev (start_demo.bat) always uses 127.0.0.1:8000. On a hosting
    # platform (Render, Railway, Fly, ...) the process is only reachable if
    # it binds 0.0.0.0 and listens on whatever port the platform assigns via
    # $PORT — hardcoding 127.0.0.1:8000 would make the deployed instance
    # unreachable from outside its own container even though it "starts
    # fine" in the logs. HOST/PORT env vars let one script cover both.
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port)
