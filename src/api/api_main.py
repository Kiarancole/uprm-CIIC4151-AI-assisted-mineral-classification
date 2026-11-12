from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from handler import MineralImage, model_obj

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_obj.load_model()
    yield

app = FastAPI()

origins = [ "http://localhost",
            "http://localhost:8080",
        ] # TODO change to ["*"] wildcard if having issues

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.post("/predict")
def predict(mineral: MineralImage):
    return model_obj.predict_mineral(mineral)