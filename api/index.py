"""Vercel entrypoint for the public ScamIntelli analysis API.

This intentionally does not start the long-lived worker/cleanup/Neo4j lifespan
used by the full Render service. The website only needs stateless message
analysis at POST /api/v1/analyze.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api_gateway.routes import router

app = FastAPI(
    title="ScamIntelli Analysis API",
    description="Public suspicious-message analysis API",
    version="2.0.0-vercel",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://scamintelligence.online",
        "https://www.scamintelligence.online",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {"service": "ScamIntelli Analysis API", "status": "ready"}


@app.get("/health")
async def health():
    return {"status": "ok"}
