# PathSnap backend: what was built

The backend takes one raw GPS trip, matches it to the road network with the final model (tuned HistGradientBoosting
classifier + HMM/Viterbi) and returns the predicted route with accuracy and route continuity. It runs on laptop 1.

## Files

| File | Role |
|---|---|
| `config.py` | Host, port, API address, eval-trip folder and model folder. Each can be overridden by an environment variable. |
| `predict.py` | `match_trip(trip_dir)`: raw trip in, matched route out. Task 1 of the brief. |
| `scoring.py` | Accuracy and route continuity against the true route. Server side only. |
| `api.py` | FastAPI app that lists the trips and runs a chosen trip through `predict.py` and `scoring.py`. |
| `pipeline/pathsnap_hmm.py` | Two small backward-compatible changes (see below). |

## `predict.py`: the pipeline for one unseen trip

Reads only `<id>.track`, `<id>.nodes` and `<id>.arcs`. The `.route` file is never opened here.

1. **Cleaning that needs only GPS and map:** local projection (C1), segments split at pauses over 60 s (C2),
   speeds (C3), heading, dwell detection, position jumps (C8) and eligible arcs (C9). The label-dependent rules
   (C4, C6, C7) are training-only and are not used.
2. **Candidates:** every eligible arc within 50 m of each fix, topped up to at least 5 (KD-tree over points sampled
   every 10 m along arcs).
3. **Features:** the 13 selected features, copied from `feature_engineering.ipynb` with the label columns removed.
4. **Classifier:** `models_out/final_model/classifier.joblib` gives a probability per candidate.
5. **HMM:** `pathsnap_hmm.py` with the settings from `hmm_config.json` (beta 3, vmax 50 m/s, ...) gives one arc per
   fix, then a connected arc sequence is built with shortest-path fill.

The model and its two JSON configs are loaded once per process. Output: route arcs, a `[lat, lon]` polyline of the
route, the GPS track, the arc chosen for each fix, and counts of HMM breaks and holes.

## `scoring.py`: what the evaluator sees

The true route and the labels derived from it are read on the server only. Labels and the "reliable fix" flag come from
the same `clean.py` code that produced the training labels, so the numbers are comparable to the notebooks.

| Metric | Meaning |
|---|---|
| `accuracy` | Top-1: share of reliable fixes matched to the correct directed arc (the notebook headline number) |
| `accuracy_all_fixes` | Same, over every fix |
| `route_continuity` | Share of consecutive fixes whose matched arcs are equal or connected within 3 arcs |
| `route_precision`, `route_recall`, `route_f1` | Length-weighted overlap of predicted and true arc sets |

## `api.py`: endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Returns `{"status": "ok"}` |
| `GET /trips` | The 10 evaluation trip IDs and nothing else |
| `POST /predict` | Body `{"trip_id": "00000007"}`. Returns `metrics`, `route` (polyline), `track`, `route_arcs`, `seconds` |

- **The true route never leaves the server.** No response contains it.
- **The trip ID is checked** against the evaluation folder listing, so path tricks such as `../x` get a 404.
- **Network:** the API binds to `0.0.0.0:8000` so another laptop can reach it. CORS is open.
- **Logging (terminal):** one line per request with client IP, method, path, status and time. Each prediction
  adds a line with trip, fixes, route arcs, accuracy, continuity and predict time. Errors are logged with a stack trace.

```
15:46:01 | INFO | trip 00000007: 1354 fixes, 122 route arcs | accuracy 99.74% (all fixes 97.42%) | continuity 100.00% | predict 0.17s
15:46:01 | INFO | 127.0.0.1 POST /predict -> 200 (0.26s)
```

## Changes to existing code

`pipeline/pathsnap_hmm.py` normally reads `data_clean/`. Two backward-compatible additions let the API run without it:
`TripNet(arrays=(from, to, length, eligible))` and `prepare_trip(..., fx=<fixes table>)`. The notebooks are unaffected.

## Verification

All 10 `student_test` trips were run through the live API.

- Route F1 and continuity match `models_out/hmm/per_trip_student_test.csv` (checked on trips `00000007` and
  `00000009`).
- Reliable-fix accuracy is within one fix of the notebook (trip `00000009`: 92.09% against 92.18%). Training rounds
  coordinates to 3 decimals, which probably explains the difference.
- Accuracy ranges from 84.3% (`00000077`) to 100% (`00000052`). Continuity is 99.5% or higher on every trip.
- Prediction takes 0.06 s to 1.2 s per trip.

## Running it

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt     # requirements.txt is still to be written
.venv/bin/python api.py                                                  # laptop 1
```

Settings by environment variable: `PATHSNAP_API_HOST`, `PATHSNAP_API_PORT`, `PATHSNAP_EVAL_DIR`, `PATHSNAP_MODEL_DIR`.
The default evaluation folder is `../Student folder/student_test`.

## Open points

- `requirements.txt` and the README are not written yet.
- Laptop 1's firewall may need to allow the API port.
