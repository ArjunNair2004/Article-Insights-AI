from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.website.routes import router  

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory="src/website/static"),
    name="static"
)

app.include_router(router)
