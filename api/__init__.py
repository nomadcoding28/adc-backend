"""
api/
====
FastAPI application for the ACD Framework.

Exposes the full ACD system over HTTP (REST) and WebSocket.

Sub-packages
------------
    routers/      Route handlers for each system component
    schemas/      Pydantic v2 request / response models
    websocket/    WebSocket connection manager and event handlers
    middleware/   CORS, auth, rate limiting, error handling, logging

Quick-start
-----------
    uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
"""