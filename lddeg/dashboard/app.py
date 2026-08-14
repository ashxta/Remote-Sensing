"""Minimal results viewer for a completed experiment.

Deliberately kept as a thin read-only viewer over results/<experiment_id>/.
This is a research repository, not a product: the dashboard displays what
the pipeline already wrote and computes nothing of its own.

    streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import Config  # noqa: E402

st.set_page_config(page_title="Land Degradation — Results Viewer",
                   layout="wide")
st.title("Land Degradation & Vegetation Dynamics — Results Viewer")

results_root = Path(Config().paths.results)
runs = sorted([p for p in results_root.glob("*") if p.is_dir()], reverse=True)
if not runs:
    st.warning("No experiments found. Run `python run_pipeline.py` first.")
    st.stop()

run = st.sidebar.selectbox("Experiment", runs, format_func=lambda p: p.name)
st.caption(f"Viewing `{run.name}`")

st.info(
    "Results shown here are from the SYNTHETIC development dataset unless "
    "the configuration states otherwise. They are not measurements of any "
    "real location.", icon="!")

res_file = run / "metrics" / "results.json"
res = json.loads(res_file.read_text()) if res_file.exists() else {}

tabs = st.tabs(["Summary", "Figures", "Metrics", "Config", "Log"])

with tabs[0]:
    q = res.get("data_quality", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Usable pixels", f"{q.get('n_usable', 0):,}")
    c2.metric("Usable fraction", f"{100 * q.get('usable_fraction', 0):.1f}%")
    c3.metric("Observations", q.get("n_obs_per_pixel", "-"))
    if "method_comparison" in res:
        st.subheader("Method comparison by true class")
        st.dataframe(pd.DataFrame(res["method_comparison"]).T,
                     use_container_width=True)
        st.caption("MK/RESTREND columns show the share flagged as a "
                   "significant decline; RF shows classification accuracy.")
    if "recovery" in res:
        st.subheader("Recovery status")
        st.bar_chart(pd.Series(res["recovery"]["status_counts"]))
    if "trajectory" in res:
        st.subheader("Analytical trajectory classes")
        st.bar_chart(pd.Series(res["trajectory"]["counts"]))
        st.caption("Analytical signal categories derived from NDVI shape. "
                   "They are not verified land-cover classes: cyclic does "
                   "not mean jhum and a negative trend does not prove "
                   "degradation.")
    if "spatial_block_cv" in res:
        st.subheader("Validation (spatial block CV is primary)")
        rows = {"spatial block CV": res["spatial_block_cv"]}
        if "random_split_baseline" in res:
            rows["random split (baseline, optimistic)"] = \
                res["random_split_baseline"]
        st.dataframe(pd.DataFrame({
            name: {k: metrics.get(k) for k in
                   ("accuracy", "f1_macro", "f1_weighted", "cohen_kappa")}
            for name, metrics in rows.items()}).T,
            use_container_width=True)

with tabs[1]:
    figs = sorted((run / "figures").glob("*.png"))
    for f in figs:
        st.image(str(f), caption=f.stem, use_container_width=True)
    if not figs:
        st.write("No figures in this run.")

with tabs[2]:
    for f in sorted((run / "metrics").glob("*")):
        # M2 groups per-experiment artifacts into sub-directories; the viewer
        # shows the top-level tables and leaves those to the file system.
        if f.is_dir():
            st.caption(f"{f.name}/ — {len(list(f.glob('*')))} experiment "
                       "artifact(s) on disk")
            continue
        st.markdown(f"**{f.name}**")
        if f.suffix == ".csv":
            st.dataframe(pd.read_csv(f, index_col=0), use_container_width=True)
        elif f.suffix == ".json":
            st.json(json.loads(f.read_text()))
        else:
            st.code(f.read_text())

with tabs[3]:
    for name in ("config.json", "environment.json", "georeference.json"):
        p = run / "config" / name
        if p.exists():
            st.markdown(f"**{name}**")
            st.json(json.loads(p.read_text()))

with tabs[4]:
    log = run / "logs" / "run.log"
    if log.exists():
        st.code(log.read_text())
