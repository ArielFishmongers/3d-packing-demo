"""
dashboard.py — Interactive Streamlit dashboard for warehouse packing comparison.

Run from the project root:
    streamlit run examples/dashboard.py

Install dependencies first:
    pip install -e ".[dashboard]"
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from bin_packing.visualisation import animate_packing, plot_packing  # noqa: E402
from bin_packing.warehouse_algorithms import AlgorithmConfig, PACKING_MODES
from bin_packing.warehouse_compare import (  # noqa: E402
    MAX_WORKERS_LIMIT,
    ComparisonConfig,
    aggregate_across_jobs,
    build_cuboids,
    run_jobs_parallel,
)
from bin_packing.warehouse_io import (  # noqa: E402
    DEFAULT_WAREHOUSE_DATA_PATHS,
    load_events_by_job,
    load_item_master,
    parse_container_dims,
)

# ── Page config — must be the first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="3D Packing Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Guard: warehouse CSV inputs must exist.
if (
    not DEFAULT_WAREHOUSE_DATA_PATHS.activity_file.exists()
    or not DEFAULT_WAREHOUSE_DATA_PATHS.item_file.exists()
):
    st.error(
        "**Warehouse CSV files not found.** "
        "Expected files under the repository `data/` directory."
    )
    st.stop()


# ── Cached data-loading helpers ───────────────────────────────────────────────

@st.cache_data(show_spinner="Loading item master...")
def cached_load_item_master() -> dict:
    return load_item_master()


@st.cache_data(show_spinner="Loading picking events...")
def cached_load_events_by_job(limit: int | None) -> dict:
    return load_events_by_job(limit=limit)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📦 3D Packing Demo")
    st.divider()

    container_str = st.text_input(
        "Container dims (L×W×H cm)",
        value="27x32x50",
        help="e.g. 27x32x50",
    )
    limit_raw = st.number_input(
        "Max items to load (0 = all)",
        min_value=0,
        value=0,
        step=100,
        help="Limit the dataset size for faster results.",
    )
    limit = int(limit_raw) if limit_raw > 0 else None

    algorithm_mode = st.selectbox(
        "Algorithm",
        options=PACKING_MODES,
        index=PACKING_MODES.index("sequential"),
        format_func=lambda mode: mode.upper() if mode == "ga" else mode.title(),
    )

    with st.expander("Advanced", expanded=False):
        n_workers = st.number_input(
            "Parallel processes",
            min_value=1,
            max_value=MAX_WORKERS_LIMIT,
            value=1,
            step=1,
            help=f"Number of parallel processes used by the packing run (max {MAX_WORKERS_LIMIT}).",
        )

    st.divider()
    run_clicked = st.button("Run Analysis", type="primary", width="stretch")

# Validate container dims immediately so we can show errors before running.
container_dims = parse_container_dims(container_str)
if container_dims is None:
    st.sidebar.error("Invalid format — use LxWxH, e.g. 27x32x50")

# ── Trigger analysis ──────────────────────────────────────────────────────────

# Auto-run a small demo on first visit so cold-start visitors see results
# without having to click anything. Subsequent runs honour the user's
# sidebar choices.
if (
    "job_results" not in st.session_state
    and "auto_run_done" not in st.session_state
    and container_dims is not None
):
    st.session_state["auto_run_done"] = True
    run_clicked = True
    limit = 500
    algorithm_mode = "sequential"
    n_workers = 1

if run_clicked and container_dims is not None:
    item_dims = cached_load_item_master()
    events_by_job = cached_load_events_by_job(limit)
    comparison_config = ComparisonConfig(
        container_dims=container_dims,
        max_workers=int(n_workers),
        algorithm=AlgorithmConfig(mode=algorithm_mode),
    )

    if not events_by_job:
        st.sidebar.error("No picking events found in data file.")
    else:
        n_jobs = len(events_by_job)
        progress_bar = st.progress(0, text=f"Packing job 0 / {n_jobs}…")
        _last_pct: list[int] = [0]

        def _on_progress(completed: int, total: int) -> None:
            pct = int(completed / total * 100)
            if pct != _last_pct[0] or completed == total:
                _last_pct[0] = pct
                progress_bar.progress(
                    completed / total,
                    text=f"Packing job {completed} / {total}…",
                )

        job_results = run_jobs_parallel(
            events_by_job, item_dims, comparison_config,
            progress_callback=_on_progress,
        )
        progress_bar.empty()

        st.session_state["job_results"]   = job_results
        st.session_state["events_by_job"] = events_by_job
        st.session_state["item_dims"]     = item_dims
        st.session_state["run_params"]    = {
            "container": container_str,
            "limit":     limit,
            "workers":   int(n_workers),
            "mode":      algorithm_mode,
        }
        st.rerun()

# ── Stale-params warning ──────────────────────────────────────────────────────

results_ready = "job_results" in st.session_state and st.session_state["job_results"]
active_run_params = st.session_state.get("run_params", {}) if results_ready else {}
active_container_str = active_run_params.get("container", container_str)
active_container_dims = parse_container_dims(active_container_str) if active_container_str else container_dims
active_mode = active_run_params.get("mode", "sequential")

if results_ready:
    prev = active_run_params
    if (
        prev.get("container") != container_str
        or prev.get("limit") != limit
        or prev.get("mode") != algorithm_mode
    ):
        st.warning(
            "Parameters have changed since the last run — "
            "click **Run Analysis** to update."
        )

# ── Main title ────────────────────────────────────────────────────────────────

st.title("Warehouse Packing Dashboard")
st.caption(
    "Compare real-world manual packing against the selected algorithm."
)

st.markdown(
    """
    This demo replays a sample of real warehouse picking jobs through a 3D bin
    packing algorithm and compares the result against how the items were actually
    packed by hand. Pick an algorithm in the sidebar — **Sequential** packs items
    in the order they were picked, **Heuristic** is a fast extreme-point greedy
    pass, and **GA** refines the heuristic with a genetic search — then click
    **Run Analysis**. The **Animation** tab lets you watch any single container
    being packed step-by-step.
    """
)

tab_overview, tab_perjob, tab_drilldown, tab_animation = st.tabs(
    ["Overview", "Per-Job", "Drill-Down", "Animation"]
)

_NO_DATA_MSG = "Configure parameters in the sidebar and click **Run Analysis** to begin."

# ── Tab 1 — Overview ──────────────────────────────────────────────────────────

with tab_overview:
    if not results_ready:
        st.info(_NO_DATA_MSG)
    else:
        job_results = st.session_state["job_results"]
        ws_agg, als_agg = aggregate_across_jobs(job_results)

        # Metric cards
        col1, col2, col3 = st.columns(3)
        saved = ws_agg["total_containers"] - als_agg["total_containers"]
        util_delta = als_agg["avg_util"] - ws_agg["avg_util"]

        col1.metric(
            "Containers Saved",
            value=saved,
            delta=f"{saved / ws_agg['total_containers'] * 100:.0f}% reduction"
            if ws_agg["total_containers"] else None,
            delta_color="normal",
        )
        col2.metric(
            "Fill % Improvement",
            value=f"{util_delta:+.1f} pp",
            delta=None,
        )
        col3.metric("Jobs Analysed", value=len(job_results))

        # Warnings
        if ws_agg["missing_dims"]:
            st.warning(
                f"{ws_agg['missing_dims']:,} pick line(s) had no dimension data "
                f"and were excluded from volume calculations."
            )
        all_warnings = [w for jr in job_results for w in jr["skip_warnings"]]
        if all_warnings:
            with st.expander(f"Packing warnings ({len(all_warnings)})", expanded=False):
                for w in all_warnings:
                    st.text(w)

        st.divider()

        # Grouped bar chart — aggregate metrics
        L, W, H = active_container_dims
        metrics = ["Total Containers", "Avg Fill %", "Avg Items / Container"]
        w_vals  = [ws_agg["total_containers"],  ws_agg["avg_util"],  ws_agg["avg_items"]]
        a_vals  = [als_agg["total_containers"], als_agg["avg_util"], als_agg["avg_items"]]

        df_bar = pd.DataFrame({
            "Metric": metrics * 2,
            "Value":  w_vals + a_vals,
            "Source": ["Manual"] * 3 + ["Algorithm"] * 3,
        })
        fig_bar = px.bar(
            df_bar,
            x="Metric", y="Value", color="Source", barmode="group",
            title=f"Aggregate Comparison — {L}×{W}×{H} cm container",
            color_discrete_map={"Manual": "#636EFA", "Algorithm": "#EF553B"},
        )
        st.plotly_chart(fig_bar, width="stretch")

        # Fill % distribution box plot
        n = len(job_results)
        df_util = pd.DataFrame({
            "Fill %": [jr["ws"]["avg_util"]  for jr in job_results]
                    + [jr["als"]["avg_util"] for jr in job_results],
            "Source": ["Manual"] * n + ["Algorithm"] * n,
        })
        fig_box = px.box(
            df_util,
            x="Source", y="Fill %", points="all",
            color="Source",
            color_discrete_map={"Manual": "#636EFA", "Algorithm": "#EF553B"},
            title="Fill % Distribution Across Jobs",
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, width="stretch")

# ── Tab 2 — Per-Job ───────────────────────────────────────────────────────────

with tab_perjob:
    if not results_ready:
        st.info(_NO_DATA_MSG)
    else:
        job_results = st.session_state["job_results"]

        # Build DataFrame
        rows = []
        for jr in job_results:
            w_c = jr["ws"]["total_containers"]
            a_c = jr["als"]["total_containers"]
            rows.append({
                "Job ID":             jr["job_id"],
                "Manual Containers":  w_c,
                "Manual Fill %":      round(jr["ws"]["avg_util"], 1),
                "Algo Containers":    a_c,
                "Algo Fill %":        round(jr["als"]["avg_util"], 1),
                "Saved":              w_c - a_c,
                "Items":              jr["n_cuboids"],
                "Skipped":            jr["skipped"],
            })
        df_jobs = pd.DataFrame(rows)

        # Colour-coded dataframe
        def _colour_saved(val):
            if val > 0:
                return "background-color: #d4edda; color: #155724"
            if val < 0:
                return "background-color: #f8d7da; color: #721c24"
            return ""

        styled = df_jobs.style.map(_colour_saved, subset=["Saved"])

        st.dataframe(
            styled,
            width="stretch",
            height=min(400, 38 + 35 * len(df_jobs)),
            column_config={
                "Manual Fill %":  st.column_config.NumberColumn(format="%.1f%%"),
                "Algo Fill %":    st.column_config.NumberColumn(format="%.1f%%"),
                "Saved":          st.column_config.NumberColumn(format="%+d"),
            },
            hide_index=True,
        )

        st.divider()

        # Containers saved bar chart
        df_saved = df_jobs[["Job ID", "Saved"]].copy()
        df_saved["Colour"] = df_saved["Saved"].apply(
            lambda v: "Saved" if v > 0 else ("Extra needed" if v < 0 else "No change")
        )
        fig_saved = px.bar(
            df_saved, x="Job ID", y="Saved", color="Colour",
            color_discrete_map={
                "Saved":       "#28a745",
                "Extra needed": "#dc3545",
                "No change":   "#adb5bd",
            },
            title="Containers Saved Per Job  (positive = algorithm wins)",
        )
        fig_saved.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
        fig_saved.update_layout(legend_title_text="")
        st.plotly_chart(fig_saved, width="stretch")

        # Fill % line chart
        fig_fill = go.Figure()
        job_ids = df_jobs["Job ID"].tolist()
        fig_fill.add_trace(go.Scatter(
            x=job_ids, y=df_jobs["Manual Fill %"].tolist(),
            mode="markers+lines", name="Manual",
            line={"color": "#636EFA"},
        ))
        fig_fill.add_trace(go.Scatter(
            x=job_ids, y=df_jobs["Algo Fill %"].tolist(),
            mode="markers+lines", name="Algorithm",
            line={"color": "#EF553B"},
        ))
        fig_fill.update_layout(
            title="Avg Fill % Per Job",
            xaxis_title="Job ID",
            yaxis_title="Fill %",
        )
        st.plotly_chart(fig_fill, width="stretch")

# ── Tab 3 — Drill-Down ────────────────────────────────────────────────────────

with tab_drilldown:
    if not results_ready:
        st.info(_NO_DATA_MSG)
    else:
        job_results = st.session_state["job_results"]

        job_ids = [jr["job_id"] for jr in job_results]
        selected_job = st.selectbox("Select job to inspect:", options=job_ids)

        jr = next(r for r in job_results if r["job_id"] == selected_job)

        # Per-container fill % grouped bar
        w_ctrs = jr["w_containers"]
        a_ctrs = jr["a_containers"]
        max_rank = max(len(w_ctrs), len(a_ctrs), 1)
        ranks = list(range(1, max_rank + 1))

        w_map = {c["rank"]: c["util_pct"] for c in w_ctrs}
        a_map = {c["rank"]: c["util_pct"] for c in a_ctrs}

        df_ctr = pd.DataFrame({
            "Container #": ranks * 2,
            "Fill %": [w_map.get(r) for r in ranks] + [a_map.get(r) for r in ranks],
            "Source": ["Manual"] * max_rank + ["Algorithm"] * max_rank,
        })
        fig_ctr = px.bar(
            df_ctr, x="Container #", y="Fill %", color="Source", barmode="group",
            title=f"Per-Container Fill % — Job {selected_job}",
            color_discrete_map={"Manual": "#636EFA", "Algorithm": "#EF553B"},
        )
        fig_ctr.add_hline(
            y=100, line_dash="dot", line_color="grey",
            annotation_text="100% capacity", annotation_position="top right",
        )
        fig_ctr.update_layout(xaxis={"dtick": 1})
        st.plotly_chart(fig_ctr, width="stretch")

        # 3D packing view
        st.divider()
        st.subheader("3D Packing View (Algorithm)")

        n_algo_ctrs = len(a_ctrs)
        if n_algo_ctrs == 0:
            st.info("No algorithm containers to visualise for this job.")
        else:
            ctr_options = [f"Container {i + 1}" for i in range(n_algo_ctrs)]
            selected_ctr = st.selectbox("Algorithm container:", options=ctr_options)
            ctr_idx = int(selected_ctr.split()[-1]) - 1

            pack_results = jr["pack_results"]
            id_to_dims   = jr["id_to_dims"]

            if ctr_idx < len(pack_results):
                result = pack_results[ctr_idx]
                n_items = len(result.placements)
                packed_vol = sum(
                    p.orientation.l * p.orientation.w * p.orientation.h
                    for p in result.placements
                )
                cvol = active_container_dims[0] * active_container_dims[1] * active_container_dims[2]
                util_pct = packed_vol / cvol * 100 if cvol else 0.0

                fig_3d = plot_packing(
                    result,
                    title=f"Job {selected_job} — Algorithm Container {ctr_idx + 1}",
                    opacity=1.0,
                    show_labels=True,
                )

                # If this container has a rejected item, render it just outside.
                L, W, H = active_container_dims
                rejected_id = result.unpacked_ids[0] if result.unpacked_ids else None
                if rejected_id and rejected_id in id_to_dims:
                    rl, rw, rh = id_to_dims[rejected_id]
                    gap = max(L * 0.1, 2)   # small gap between container and box
                    rx = L + gap            # placed immediately to the right

                    # 8 corner vertices
                    xs = [rx,      rx + rl, rx + rl, rx,      rx,      rx + rl, rx + rl, rx     ]
                    ys = [0,       0,       rw,      rw,      0,       0,       rw,      rw     ]
                    zs = [0,       0,       0,       0,       rh,      rh,      rh,      rh     ]
                    fi = [0, 0,  4, 4,  0, 0,  3, 3,  0, 0,  1, 1]
                    fj = [1, 2,  5, 6,  1, 5,  2, 6,  3, 7,  2, 6]
                    fk = [2, 3,  6, 7,  5, 4,  6, 7,  7, 4,  6, 5]

                    hover = (
                        f"<b>{rejected_id}</b> (did not fit)<br>"
                        f"Dims: {rl:.4g} × {rw:.4g} × {rh:.4g}"
                    )
                    fig_3d.add_trace(go.Mesh3d(
                        x=xs, y=ys, z=zs,
                        i=fi, j=fj, k=fk,
                        color="#dc3545",
                        opacity=1.0,
                        name=f"{rejected_id} (rejected)",
                        hovertemplate=hover + "<extra></extra>",
                        showlegend=True,
                        flatshading=True,
                        lighting=dict(diffuse=0.8, specular=0.2, ambient=0.4),
                    ))

                    # Extend the x-axis range to show the rejected box fully
                    fig_3d.update_layout(scene=dict(
                        xaxis=dict(range=[0, rx + rl + gap]),
                        yaxis=dict(range=[0, max(W, rw)]),
                        zaxis=dict(range=[0, max(H, rh)]),
                        aspectmode="data",
                    ))

                st.plotly_chart(fig_3d, width="stretch")

                caption = (
                    f"{n_items} item(s) packed · "
                    f"{util_pct:.1f}% fill · "
                    f"container {ctr_idx + 1} of {len(pack_results)}"
                )
                if rejected_id:
                    caption += f" · red box = first item that didn't fit"
                st.caption(caption)
            else:
                st.warning(
                    f"Container {ctr_idx + 1} is beyond the {len(pack_results)} "
                    f"container(s) produced for this job."
                )

# ── Tab 4 — Animation ─────────────────────────────────────────────────────────

with tab_animation:
    if not results_ready:
        st.info(_NO_DATA_MSG)
    else:
        job_results = st.session_state["job_results"]
        job_ids = [jr["job_id"] for jr in job_results]
        anim_job = st.selectbox(
            "Select job to animate:", options=job_ids, key="anim_job"
        )
        jr = next(r for r in job_results if r["job_id"] == anim_job)

        pack_results = jr["pack_results"]
        n_algo_ctrs = len(pack_results)
        if n_algo_ctrs == 0:
            st.info("No algorithm containers to animate for this job.")
        else:
            ctr_options = [f"Container {i + 1}" for i in range(n_algo_ctrs)]
            anim_ctr = st.selectbox(
                "Algorithm container:", options=ctr_options, key="anim_ctr"
            )
            ctr_idx = int(anim_ctr.split()[-1]) - 1

            col_speed, col_smooth = st.columns(2)
            with col_speed:
                frame_duration_ms = st.slider(
                    "Frame duration (ms)", min_value=10, max_value=100,
                    value=30, step=5,
                    help="Lower = faster playback.",
                )
            with col_smooth:
                n_steps = st.slider(
                    "Smoothness (frames per box)", min_value=5, max_value=30,
                    value=15, step=1,
                    help="Higher = smoother drop, longer animation.",
                )

            result = pack_results[ctr_idx]
            fig_anim = animate_packing(
                result,
                title=f"Job {anim_job} — Container {ctr_idx + 1}",
                n_steps=n_steps,
                frame_duration_ms=frame_duration_ms,
            )
            st.plotly_chart(fig_anim, width="stretch")
            st.caption(
                f"{len(result.placements)} item(s) packed · "
                f"press ▶ Play to watch each box drop into place."
            )
