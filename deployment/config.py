"""Central settings for the PathSnap backend / UI. Every value can be overridden with an environment variable.

    PATHSNAP_API_HOST   address the API listens on (0.0.0.0 = reachable from other laptops)   default 0.0.0.0
    PATHSNAP_API_PORT   API port                                                              default 8000
    PATHSNAP_API_URL    (UI side) where the UI finds the API, e.g. http://192.168.1.20:8000   default localhost
    PATHSNAP_EVAL_DIR   folder with the evaluation trips (one sub-folder per trip: .track .nodes .arcs .route)
    PATHSNAP_MODEL_DIR  folder with classifier.joblib, classifier_config.json, hmm_config.json
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # repository root

API_HOST = os.environ.get("PATHSNAP_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("PATHSNAP_API_PORT", "8000"))
API_URL = os.environ.get("PATHSNAP_API_URL", f"http://localhost:{API_PORT}")


def _first_existing(*paths):
    for p in paths:
        if p and Path(p).is_dir():
            return Path(p)
    return Path(paths[0])  # fall back to the first guess so the error message names a sensible path


EVAL_DIR = _first_existing(
    os.environ.get("PATHSNAP_EVAL_DIR"),
    ROOT / "Student folder" / "student_test",
    ROOT.parent / "Student folder" / "student_test",
)
MODEL_DIR = Path(os.environ.get("PATHSNAP_MODEL_DIR", ROOT / "results" / "final_model"))
# code folders the API imports from
DATA_PROCESSING_DIR = ROOT / "data_processing"
HMM_DIR = ROOT / "hmm_viterbi"
TRAINING_DIR = ROOT / "model_training"
