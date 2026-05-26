import logging

from fastapi import FastAPI

from .routes import router


logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s"
)


app = FastAPI()
app.include_router(router)
