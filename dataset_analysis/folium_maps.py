"""Interactive Folium maps: GPS fixes, nearby road network, reference route.
Usage: python folium_maps.py "<Student folder>" out train/00000045 train/00000033 ...
"""
import os
import sys

import folium
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load_trip  # noqa: E402


def trip_map(base, rel, out):
    t, tr, nd, ar, rt = load_trip(f"{base}/{rel}")
    lon, lat = tr[:, 0], tr[:, 1]
    m = folium.Map(location=[np.median(lat), np.median(lon)], zoom_start=14, tiles="OpenStreetMap")
    pad = 0.003
    fr, to = ar[:, 0], ar[:, 1]
    near = ((np.maximum(nd[fr, 0], nd[to, 0]) > lon.min() - pad) & (np.minimum(nd[fr, 0], nd[to, 0]) < lon.max() + pad) &
            (np.maximum(nd[fr, 1], nd[to, 1]) > lat.min() - pad) & (np.minimum(nd[fr, 1], nd[to, 1]) < lat.max() + pad))
    net = folium.FeatureGroup("Road network (nearby arcs)", show=True)
    ids = np.flatnonzero(near)
    # draw each two-way road once, but keep one-way arcs whatever their direction
    nn = len(nd)
    has_rev = np.isin(to[ids] * nn + fr[ids], fr * nn + to)
    ids = ids[(fr[ids] < to[ids]) | ~has_rev]
    segs = [[(round(nd[fr[a], 1], 6), round(nd[fr[a], 0], 6)), (round(nd[to[a], 1], 6), round(nd[to[a], 0], 6))]
            for a in ids]
    folium.PolyLine(segs, color="#6250d6", weight=1.5, opacity=0.55).add_to(net)
    net.add_to(m)
    gps = folium.FeatureGroup("GPS fixes", show=True)
    step = max(1, len(tr) // 1500)  # keep the HTML light for long trips
    for i in range(0, len(tr), step):
        folium.CircleMarker((lat[i], lon[i]), radius=2.5, color="#2a78d6", fill=True, fill_opacity=0.6, weight=0,
                            tooltip=f"t={tr[i, 2]:.0f}s").add_to(gps)
    gps.add_to(m)
    route = folium.FeatureGroup("Reference route", show=True)
    folium.PolyLine([[(nd[fr[a], 1], nd[fr[a], 0]), (nd[to[a], 1], nd[to[a], 0])] for a in rt],
                    color="#008300", weight=4, opacity=0.75, tooltip="reference route").add_to(route)
    route.add_to(m)
    folium.Marker((lat[0], lon[0]), tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((lat[-1], lon[-1]), tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    os.makedirs(f"{out}/maps", exist_ok=True)
    path = f"{out}/maps/trip_{t}.html"
    m.save(path)
    print("wrote", path)


if __name__ == "__main__":
    base, out, *trips = sys.argv[1:]
    for rel in trips:
        trip_map(base, rel, out)
