from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .indexer import load_index_json
from .routes import router


STATIC_DIR = Path(__file__).resolve().parent / "static"


def load_persisted_index():
    load_index_json()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_persisted_index()
    yield


app = FastAPI(title="Markdown Knowledge Base Q&A Bot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def browser_ui():
    return FileResponse(STATIC_DIR / "index.html")


app.include_router(router)
