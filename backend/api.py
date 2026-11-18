"""PathSnap evaluation API (runs on laptop 1).

    python api.py                      # listens on PATHSNAP_API_HOST:PATHSNAP_API_PORT (default 0.0.0.0:8000)

GET  /health          liveness check
GET  /trips           the evaluation trip IDs (nothing else)
POST /predict         {"trip_id": "00000007"} -> predicted route + accuracy and route continuity

The true route is read and used only inside scoring.py on this machine; it is never part of any response.
Every request is printed to the terminal (method, path, status, time) and every prediction with its scores.
"""
import json
import logging
import re
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import predict
import scoring

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("pathsnap")

@asynccontextmanager
async def lifespan(_):
    predict.load_models()                       # load the classifier once, not on the first request
    log.info("model loaded; %d evaluation trips in %s", len(trip_ids()), config.EVAL_DIR)
    yield


app = FastAPI(title="PathSnap evaluation API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def trip_ids():
    if not config.EVAL_DIR.is_dir():
        raise HTTPException(500, f"evaluation folder not found: {config.EVAL_DIR}")
    return sorted(p.name for p in config.EVAL_DIR.iterdir() if p.is_dir())


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    client = request.client.host if request.client else "?"
    try:
        response = await call_next(request)
    except Exception:
        log.exception("%s %s %s -> 500 (unhandled)", client, request.method, request.url.path)
        raise
    log.info("%s %s %s -> %d (%.2fs)", client, request.method, request.url.path, response.status_code,
             time.time() - t0)
    return response


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trips")
def trips():
    return {"trips": trip_ids()}


@app.post("/predict")
def predict_route(req: PredictRequest):
    # the ID must be one of the listed folders (also blocks path tricks like "../x")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", req.trip_id) or req.trip_id not in trip_ids():
        raise HTTPException(404, f"unknown trip_id '{req.trip_id}'")
    trip_dir = config.EVAL_DIR / req.trip_id
    t0 = time.time()
    res = predict.match_trip(trip_dir)
    t_pred = time.time() - t0
    metrics = scoring.score_trip(trip_dir, res["fix_arc"], res["route"])
    tm = res["timing"]
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trip_id": req.trip_id,
        "n_fixes": res["n_fixes"],
        "n_candidates": res["n_candidates"],
        "time_s": {"candidate_generation": round(tm["candidates_s"], 3), "classifier": round(tm["classifier_s"], 3),
                   "hmm": round(tm["hmm_s"], 3), "cleaning": round(tm["cleaning_s"], 3),
                   "features": round(tm["features_s"], 3), "total": round(t_pred, 3)},
        "accuracy": round(metrics["accuracy"], 4),
        "accuracy_all_fixes": round(metrics["accuracy_all_fixes"], 4),
        "route_continuity": round(metrics["route_continuity"], 4),
    }
    log.info("PREDICT\n%s", json.dumps(entry, indent=2))
    return {
        "trip_id": req.trip_id,
        "metrics": metrics,
        "route": res["polyline"],               # [[lat, lon], ...] of the predicted route
        "track": res["track"],                  # the GPS fixes the car recorded (model input, not the truth)
        "route_arcs": res["route"],
        "seconds": round(t_pred, 2),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_level="warning", access_log=False)  # our own request log is above
