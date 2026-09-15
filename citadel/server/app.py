"""FastAPI lifecycle and error handling; GUI assets are independent files."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from citadel.configuration import PROJECT_ROOT
from citadel.domain.errors import BusyError, GateError
from .routes import create_router


def create_app(runtime):
    @asynccontextmanager
    async def lifespan(app):
        runtime.jobs.start()
        try:
            yield
        finally:
            runtime.jobs.close()

    app = FastAPI(title="Data Citadel", version="2.0.0", lifespan=lifespan)
    app.state.runtime = runtime

    @app.exception_handler(KeyError)
    async def missing(_, exc):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def invalid(_, exc):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(BusyError)
    @app.exception_handler(GateError)
    async def conflict(_, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    app.include_router(create_router(runtime))
    gui = PROJECT_ROOT / "gui"
    if gui.is_dir():
        app.mount("/", StaticFiles(directory=gui, html=True), name="gui")
    return app
