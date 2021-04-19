from fastapi import FastAPI

from model import generate_mouse_movements

app = FastAPI()


@app.get("/")
async def root():
    return generate_mouse_movements()
