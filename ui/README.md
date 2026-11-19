# PathSnap frontend: what was built

The evaluator UI is a plain HTML/CSS/JavaScript page (no framework, no build step) that runs in a browser on
laptop 2. It lists the evaluation trips, sends the chosen trip ID to the API on laptop 1 and shows the returned
route on a map with the scores. It holds no data or model, so the true route never reaches it.

## Files (`ui/`)

| File | Role |
|---|---|
| `index.html` | Page structure: connection and trip controls, results, map |
| `style.css` | Light styling: grey page, white panels, system font, blue Run button |
| `app.js` | Calls the API, fills in the scores, draws the route and GPS fixes on the map |
| `config.js` | The API address (`apiUrl`) and the request timeout |
| `vendor/` | Leaflet 1.9.4 (JS, CSS, images), stored locally so the page works without a CDN |

## Layout (top to bottom)

1. **Title**
2. **Controls panel**
   - **API address:** a text box, a Connect button and a status message ("Connected, 10 trips" in green, or
     "Not connected").
   - **Trip:** a dropdown filled from `GET /trips`, and a Run button. "Running…" shows while the API works.
3. **Error message:** a red box that names the problem (cannot reach the API, timeout, or the API's own error).
4. **Results panel:** accuracy, route continuity, route F1 and number of GPS fixes, with a note on how many
   fixes the accuracy is measured on and how long matching took.
5. **Map** (Leaflet + CARTO (OpenStreetMap data)): the predicted route as a blue line with green start and red end markers,
   and the GPS fixes as dark dots. The map zooms to fit the trip.

## API address

The address comes from the first of these that is set:

1. `?api=http://<laptop 1 IP>:8000` in the page URL
2. the last address that connected successfully (remembered in the browser)
3. `apiUrl` in `config.js` (default `http://localhost:8000`)

It can also be edited in the page at any time, followed by Connect.

## How it talks to the API

| When | Call |
|---|---|
| Page load or Connect | `GET <api>/trips` |
| Run | `POST <api>/predict` with `{"trip_id": "<id>"}` (timeout 120 s, set in `config.js`) |

The API already allows cross-origin requests, so the page can be served from anywhere, or opened straight from disk.

## Running it

On laptop 2, from the `ui/` folder:

```
python -m http.server 8080
```

Open `http://localhost:8080/?api=http://<laptop 1 IP>:8000`. Opening `index.html` directly also works. Map tiles are
loaded from CARTO (OpenStreetMap data), so laptop 2 needs internet access for the background map. The route and fixes draw
without it.

## Verification

- An earlier version of the page was opened in Chrome against the live API: it connected, loaded the trip list, and
  running trip `00000000` drew the route with accuracy 98.68%, continuity 100% and route F1 99.90%, matching the API
  log, with no console errors.
- The current layout and styling (top-to-bottom page, light styling) has not been checked in a browser yet. The
  Chrome extension disconnected, so only the JavaScript syntax check and the file serving were verified.
- Not tested: running it from a second laptop (laptop 1's firewall may need to allow the API port).
