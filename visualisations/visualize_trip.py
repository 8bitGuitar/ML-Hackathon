#!/usr/bin/env python3
"""Visualize a PathSnap trip: road network, reference route, and GPS track.

Usage:
    python visualize_trip.py <trip_dir_or_id> [--out fig.png] [--margin 0.15] [--full]

<trip_dir_or_id> is either a directory containing <id>.{nodes,arcs,route,track}
or a bare trip id (e.g. 00000001), searched under "Student folder/{train,student_test}".
Add --html map.html for an interactive Folium map (needs `pip install folium`; tiles need internet).
Requires numpy + matplotlib.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

DATA_ROOT = Path(__file__).resolve().parent.parent / "Student folder"


def resolve_trip(arg):
    p = Path(arg)
    if p.is_dir():
        return p
    for part in ("train", "student_test"):
        cand = DATA_ROOT / part / arg
        if cand.is_dir():
            return cand
    raise SystemExit(f"Trip not found: {arg}")


def load(trip_dir):
    tid = trip_dir.name
    nodes = np.loadtxt(trip_dir / f"{tid}.nodes")             # lon, lat
    arcs = np.loadtxt(trip_dir / f"{tid}.arcs", dtype=int)    # from, to (row = arc_id)
    route = np.loadtxt(trip_dir / f"{tid}.route", dtype=int, ndmin=1)
    track = np.loadtxt(trip_dir / f"{tid}.track")             # lon, lat, t
    return tid, nodes, arcs, route, track


def near_track_mask(nodes, track, radius_m):
    """Boolean mask of nodes within ~radius_m of any track point (grid-hash, approximate)."""
    lat0 = np.radians(track[:, 1].mean())
    scale = np.array([111320 * np.cos(lat0), 110540])    # deg -> m
    cell = lambda ll: np.floor(ll * scale / radius_m).astype(np.int64)
    tc = np.unique(cell(track[:, :2]), axis=0)
    near = {(x + dx, y + dy) for x, y in tc for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    return np.fromiter((tuple(c) in near for c in cell(nodes)), bool, len(nodes))


def make_folium_map(tid, nodes, arcs, route, track, radius_m, out):
    import folium
    segs = np.stack([nodes[arcs[:, 0]], nodes[arcs[:, 1]]], axis=1)
    keep = near_track_mask(nodes, track, radius_m)[arcs[:, 0]]
    flip = lambda a: [[float(y), float(x)] for x, y in a]          # (lon,lat) -> (lat,lon)

    m = folium.Map(location=[track[:, 1].mean(), track[:, 0].mean()],
                   zoom_start=14, tiles=None, control_scale=True)
    esri = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
    folium.TileLayer(esri.format("World_Street_Map"), attr="Esri", name="streets").add_to(m)
    folium.TileLayer(esri.format("World_Imagery"), attr="Esri", name="satellite", show=False).add_to(m)
    m.fit_bounds([[track[:, 1].min(), track[:, 0].min()], [track[:, 1].max(), track[:, 0].max()]])

    net = folium.FeatureGroup(name=f"network within {radius_m:g} m ({keep.sum()} arcs)", show=False)
    folium.PolyLine([flip(sg) for sg in segs[keep]], color="#808080", weight=1, opacity=0.7).add_to(net)
    net.add_to(m)

    rt = folium.FeatureGroup(name=f"route ({len(route)} arcs)")
    for i, a in enumerate(route):
        folium.PolyLine(flip(segs[a]), color="blue", weight=5, opacity=0.8,
                        tooltip=f"route step {i}, arc {a}").add_to(rt)
    rt.add_to(m)

    gps = folium.FeatureGroup(name=f"GPS track ({len(track)} pts)")
    folium.PolyLine(flip(track[:, :2]), color="red", weight=1, opacity=0.5).add_to(gps)
    for i, (lon, lat, t) in enumerate(track):
        folium.CircleMarker([lat, lon], radius=2, color="red", fill=True, fill_opacity=0.9,
                            weight=0, tooltip=f"#{i}  t={t:g}s").add_to(gps)
    gps.add_to(m)

    folium.Marker([track[0, 1], track[0, 0]], tooltip="start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([track[-1, 1], track[-1, 0]], tooltip="end",
                  icon=folium.Icon(color="black", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trip")
    ap.add_argument("--out", help="save figure to this path instead of showing")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="padding around track/route bbox, as fraction of its size")
    ap.add_argument("--zoom", type=int, nargs=2, metavar=("I0", "I1"),
                    help="zoom on track points I0..I1 (row indices) and the route arcs near them")
    ap.add_argument("--html", help="write an interactive Folium map to this .html file")
    ap.add_argument("--radius", type=float, default=300,
                    help="(--html) include network arcs within this many metres of the track")
    ap.add_argument("--full", action="store_true", help="show the entire network")
    args = ap.parse_args()

    tid, nodes, arcs, route, track = load(resolve_trip(args.trip))

    if args.html:
        make_folium_map(tid, nodes, arcs, route, track, args.radius, args.html)
        return

    # arc -> segment endpoints
    segs = np.stack([nodes[arcs[:, 0]], nodes[arcs[:, 1]]], axis=1)   # (A, 2, 2)
    route_segs = segs[route]

    # view window around the trip
    if args.zoom:
        pts = track[args.zoom[0]:args.zoom[1] + 1, :2]
    else:
        pts = np.vstack([track[:, :2], route_segs.reshape(-1, 2)])
    lo, hi = pts.min(0), pts.max(0)
    pad = (hi - lo) * args.margin
    lo, hi = lo - pad, hi + pad
    if args.full:
        lo, hi = nodes.min(0), nodes.max(0)

    # draw only arcs touching the window (network can be ~150k arcs)
    inside = ((segs[:, :, 0] >= lo[0]) & (segs[:, :, 0] <= hi[0]) &
              (segs[:, :, 1] >= lo[1]) & (segs[:, :, 1] <= hi[1])).any(axis=1)
    vis = nodes[(nodes[:, 0] >= lo[0]) & (nodes[:, 0] <= hi[0]) &
                (nodes[:, 1] >= lo[1]) & (nodes[:, 1] <= hi[1])]

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.add_collection(LineCollection(segs[inside], colors="#b0b0b0", linewidths=0.5,
                                     label=f"network ({inside.sum()} arcs shown)"))
    ax.scatter(vis[:, 0], vis[:, 1], s=1, c="#808080", zorder=2)
    ax.add_collection(LineCollection(route_segs, colors="tab:blue", linewidths=4,
                                     zorder=3, label=f"route ({len(route)} arcs)"))
    ax.scatter(track[:, 0], track[:, 1], s=5, c="tab:red", zorder=4,
               label=f"GPS track ({len(track)} pts)")
    ax.plot(track[0, 0], track[0, 1], "g^", ms=12, zorder=5, label="track start")
    ax.plot(track[-1, 0], track[-1, 1], "ks", ms=10, zorder=5, label="track end")

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect(1 / np.cos(np.radians((lo[1] + hi[1]) / 2)))  # lon/lat -> ~equal distance
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"Trip {tid}: {len(nodes)} nodes, {len(arcs)} arcs")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=2)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
