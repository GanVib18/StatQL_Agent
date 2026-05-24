from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cache.semantic_cache import CachedAgent


class QueryRequest(BaseModel):
    question: str


def build_app(cached_agent: CachedAgent) -> FastAPI:
    app = FastAPI(title="Analytical AI Agent", version="1.0.0")

    @app.post("/query")
    async def query(req: QueryRequest):
        if not req.question.strip():
            raise HTTPException(400, "Question cannot be empty.")
        try:
            return JSONResponse(cached_agent.ask(req.question))
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.get("/cache/stats")
    async def cache_stats():
        return JSONResponse(cached_agent.cache.get_analytics())

    @app.get("/health")
    async def health():
        return {"status": "ok", "timestamp": datetime.now().isoformat()}

    return app
