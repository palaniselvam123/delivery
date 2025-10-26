#!/bin/bash
# Optional: ensure all dependencies are installed (uncomment the next line if needed)
# pip install -r requirements.txt

# Start your FastAPI app using gunicorn + uvicorn worker
exec gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 main:app
# force redeploy
