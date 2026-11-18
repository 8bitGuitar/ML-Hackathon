// PathSnap evaluator UI: lists trips, asks the API to match one, draws the returned route and scores.
// The UI only ever sees what the API returns (predicted route, GPS track, scores).
(() => {
  const cfg = window.PATHSNAP_CONFIG || {};
  const $ = (id) => document.getElementById(id);
  const el = {
    api: $("api-url"), form: $("conn-form"), status: $("status"), trip: $("trip"), run: $("run"),
    busy: $("busy"), error: $("error"), results: $("results"), rTrip: $("r-trip"), note: $("r-note"),
    acc: $("m-acc"), cont: $("m-cont"), f1: $("m-f1"), fix: $("m-fix"),
  };

  // ---- API address: ?api=... > saved in browser > config.js
  const fromQuery = new URLSearchParams(location.search).get("api");
  const saved = (() => { try { return localStorage.getItem("pathsnap_api"); } catch { return null; } })();
  el.api.value = fromQuery || saved || cfg.apiUrl || "http://localhost:8000";
  const apiBase = () => el.api.value.trim().replace(/\/+$/, "");

  // ---- map
  const map = L.map("map", { preferCanvas: true }).setView([43.39, 77.12], 11);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  let layer = L.layerGroup().addTo(map);

  function drawResult(res) {
    layer.remove();
    layer = L.layerGroup().addTo(map);
    const canvas = L.canvas({ padding: 0.5 });
    res.track.forEach((p) => L.circleMarker(p, { renderer: canvas, radius: 2, stroke: false,
      fillColor: "#222", fillOpacity: 0.7 }).addTo(layer));
    if (res.route.length) {
      L.polyline(res.route, { color: "#1565c0", weight: 4, opacity: 0.9 }).addTo(layer);
      L.circleMarker(res.route[0], { radius: 6, color: "#1a7f37", fillColor: "#1a7f37", fillOpacity: 1 })
        .bindTooltip("Start").addTo(layer);
      L.circleMarker(res.route[res.route.length - 1], { radius: 6, color: "#b00020", fillColor: "#b00020", fillOpacity: 1 })
        .bindTooltip("End").addTo(layer);
    }
    map.fitBounds(L.latLngBounds([...res.track, ...res.route]), { padding: [30, 30] });
  }

  // ---- helpers
  const pct = (x) => (x * 100).toFixed(2) + "%";
  function setStatus(text, ok) { el.status.textContent = text; el.status.className = "status" + (ok ? " ok" : ""); }
  function showError(msg) { el.error.textContent = msg; el.error.hidden = !msg; }

  async function api(path, options) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), (cfg.requestTimeoutSeconds || 120) * 1000);
    try {
      const r = await fetch(apiBase() + path, { ...options, signal: ctrl.signal });
      if (!r.ok) {
        let detail = "";
        try { detail = (await r.json()).detail || ""; } catch { /* not JSON */ }
        throw new Error(`API returned ${r.status}${detail ? ": " + detail : ""}`);
      }
      return await r.json();
    } catch (e) {
      if (e.name === "AbortError") throw new Error("The API did not answer in time");
      if (e instanceof TypeError) throw new Error(`Cannot reach the API at ${apiBase()}`);
      throw e;
    } finally { clearTimeout(timer); }
  }

  async function connect() {
    showError("");
    setStatus("Connecting…");
    el.run.disabled = true;
    try {
      const { trips } = await api("/trips");
      el.trip.innerHTML = trips.map((t) => `<option value="${t}">${t}</option>`).join("");
      el.trip.disabled = el.run.disabled = !trips.length;
      setStatus(`Connected, ${trips.length} trips`, true);
      try { localStorage.setItem("pathsnap_api", apiBase()); } catch { /* storage unavailable */ }
    } catch (e) {
      el.trip.innerHTML = "";
      el.trip.disabled = true;
      setStatus("Not connected");
      showError(e.message);
    }
  }

  async function run() {
    showError("");
    el.run.disabled = true;
    el.busy.hidden = false;
    try {
      const res = await api("/predict", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trip_id: el.trip.value }),
      });
      const m = res.metrics;
      el.rTrip.textContent = res.trip_id;
      el.acc.textContent = pct(m.accuracy);
      el.cont.textContent = pct(m.route_continuity);
      el.f1.textContent = pct(m.route_f1);
      el.fix.textContent = m.n_fixes.toLocaleString();
      el.note.textContent = `Accuracy is measured on ${m.n_reliable_fixes.toLocaleString()} reliable fixes. Matched in ${res.seconds} s.`;
      el.results.hidden = false;
      drawResult(res);
    } catch (e) {
      showError(e.message);
    } finally {
      el.busy.hidden = true;
      el.run.disabled = false;
    }
  }

  el.form.addEventListener("submit", (e) => { e.preventDefault(); connect(); });
  el.run.addEventListener("click", run);
  connect();
})();
