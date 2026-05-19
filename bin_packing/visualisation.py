"""
3D visualisation of a PackingResult using Plotly.

Usage::

    from bin_packing.visualisation import plot_packing, animate_packing

    # Static snapshot
    fig = plot_packing(result)
    fig.show()

    # Drop animation (boxes fall into place in packing order)
    fig = animate_packing(result)
    fig.show()                      # opens a browser tab
    fig.write_html("packing.html")  # save to file

Requires plotly: pip install plotly
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go

from .models import PackingResult


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _box_vertices(
    x: float, y: float, z: float,
    l: float, w: float, h: float,
) -> tuple[list[float], list[float], list[float]]:
    """
    Return the 8 corner coordinates of a cuboid as (xs, ys, zs) lists.

    Vertex index layout:
      v0=(x,   y,   z)    v1=(x+l, y,   z)
      v2=(x+l, y+w, z)    v3=(x,   y+w, z)
      v4=(x,   y,   z+h)  v5=(x+l, y,   z+h)
      v6=(x+l, y+w, z+h)  v7=(x,   y+w, z+h)
    """
    xs = [x,     x + l, x + l, x,     x,     x + l, x + l, x    ]
    ys = [y,     y,     y + w, y + w, y,     y,     y + w, y + w ]
    zs = [z,     z,     z,     z,     z + h, z + h, z + h, z + h ]
    return xs, ys, zs


def _box_faces() -> tuple[list[int], list[int], list[int]]:
    """
    Return the (i, j, k) triangle vertex indices for the 6 faces of a box
    (12 triangles total, 2 per face). These are the same for every box.
    """
    i = [0, 0,  4, 4,  0, 0,  3, 3,  0, 0,  1, 1]
    j = [1, 2,  5, 6,  1, 5,  2, 6,  3, 7,  2, 6]
    k = [2, 3,  6, 7,  5, 4,  6, 7,  7, 4,  6, 5]
    return i, j, k


def _container_wireframe(L: float, W: float, H: float):
    """
    Return a Scatter3d trace drawing the 12 edges of the container box.
    None values in the coordinate lists act as pen-up breaks.
    """
    import plotly.graph_objects as go

    # Define 8 corners
    corners = [
        (0, 0, 0), (L, 0, 0), (L, W, 0), (0, W, 0),
        (0, 0, H), (L, 0, H), (L, W, H), (0, W, H),
    ]
    # 12 edges as pairs of corner indices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [corners[a][0], corners[b][0], None]
        ys += [corners[a][1], corners[b][1], None]
        zs += [corners[a][2], corners[b][2], None]

    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color="black", width=3),
        name="Container",
        hoverinfo="skip",
        showlegend=True,
    )


def _color_palette(n: int) -> list[str]:
    """
    Return n distinct hex colour strings, cycling if n exceeds palette length.
    Uses Plotly's built-in qualitative palette.
    """
    import plotly.colors as pc
    palette = pc.qualitative.Plotly  # 10 colours
    return [palette[i % len(palette)] for i in range(n)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_packing(
    result: PackingResult,
    *,
    title: str = "3D Bin Packing Result",
    opacity: float = 1.0,
    show_labels: bool = True,
) -> "go.Figure":
    """
    Build an interactive 3D Plotly figure of a PackingResult.

    Args:
        result:      The PackingResult returned by Packer.pack().
        title:       Figure title shown above the plot.
        opacity:     Transparency of box faces (0.0 = invisible, 1.0 = solid).
                     Values around 0.6–0.7 let you see boxes behind others.
        show_labels: If True, item IDs appear in the legend and hover tooltips.

    Returns:
        A plotly.graph_objects.Figure. Call fig.show() to open in a browser,
        or fig.write_html("out.html") to save.

    Animation extension (future):
        Build per-box traces the same way, then wrap in frames:
            frames = [go.Frame(data=traces[:k]) for k in range(1, len(traces)+1)]
            fig = go.Figure(data=traces, frames=frames)
    """
    import plotly.graph_objects as go

    L, W, H = result.container_dims
    placements = result.placements
    colors = _color_palette(len(placements))
    face_i, face_j, face_k = _box_faces()

    traces = []

    for placement, color in zip(placements, colors):
        x, y, z = placement.position
        l = placement.orientation.l
        w = placement.orientation.w
        h = placement.orientation.h

        xs, ys, zs = _box_vertices(x, y, z, l, w, h)

        hover = (
            f"<b>{placement.item_id}</b><br>"
            f"Dims: {l:.4g} × {w:.4g} × {h:.4g}<br>"
            f"Pos: ({x:.4g}, {y:.4g}, {z:.4g})"
        )

        traces.append(go.Mesh3d(
            x=xs, y=ys, z=zs,
            i=face_i, j=face_j, k=face_k,
            color=color,
            opacity=opacity,
            name=placement.item_id if show_labels else "",
            hovertemplate=hover + "<extra></extra>",
            showlegend=show_labels,
            flatshading=True,
            lighting=dict(diffuse=0.8, specular=0.2, ambient=0.4),
        ))

    # Container wireframe last so it renders on top
    traces.append(_container_wireframe(L, W, H))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(title="X", range=[0, L]),
            yaxis=dict(title="Y", range=[0, W]),
            zaxis=dict(title="Z", range=[0, H]),
            aspectmode="data",  # preserves real-world proportions
        ),
        legend=dict(title="Items"),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    if result.unpacked_ids:
        annotation_text = f"Unplaced: {', '.join(result.unpacked_ids)}"
        fig.add_annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0.01, y=0.01,
            showarrow=False,
            font=dict(color="red", size=12),
        )

    return fig


def animate_packing(
    result: PackingResult,
    *,
    title: str = "3D Bin Packing — Drop Animation",
    opacity: float = 1.0,
    show_labels: bool = True,
    n_steps: int = 15,
    frame_duration_ms: int = 30,
) -> "go.Figure":
    """
    Build an animated 3D Plotly figure where each box drops vertically into its
    final position, in packing order (canon order).

    Each box starts at the container rim (z = H) in its correct x/y position
    and orientation, then falls with quadratic ease-in (gravity acceleration)
    to its resting position.  Previously settled boxes stay in place; upcoming
    boxes are hidden above the container until their turn.

    Args:
        result:            PackingResult from Packer.pack().
        title:             Figure title.
        opacity:           Face opacity (0 = invisible, 1 = solid).
        show_labels:       Show item IDs in legend and hover tooltips.
        n_steps:           Animation frames per box drop — more = smoother.
        frame_duration_ms: Milliseconds per frame (lower = faster drops).

    Returns:
        A plotly.graph_objects.Figure with Play / Pause controls.
        Call fig.show() to open in a browser, or fig.write_html() to save.
    """
    import plotly.graph_objects as go

    L, W, H = result.container_dims
    placements = result.placements
    n = len(placements)
    colors = _color_palette(n)
    face_i, face_j, face_k_idx = _box_faces()

    # Extend the visible z range above the container so boxes are fully
    # visible while falling.  z_vis is the total visible height.
    max_box_h = max((p.orientation.h for p in placements), default=0.0)
    z_vis = H + max_box_h

    def _mesh(placement, color, z_bottom: float) -> "go.Mesh3d":
        """Build a Mesh3d trace with the box bottom at z_bottom."""
        x, y = placement.position[0], placement.position[1]
        l = placement.orientation.l
        w = placement.orientation.w
        h = placement.orientation.h
        xs, ys, zs = _box_vertices(x, y, z_bottom, l, w, h)
        hover = (
            f"<b>{placement.item_id}</b><br>"
            f"Dims: {l:.4g} × {w:.4g} × {h:.4g}<br>"
            f"Pos: ({x:.4g}, {y:.4g}, {placement.position[2]:.4g})"
        )
        return go.Mesh3d(
            x=xs, y=ys, z=zs,
            i=face_i, j=face_j, k=face_k_idx,
            color=color,
            opacity=opacity,
            name=placement.item_id if show_labels else "",
            hovertemplate=hover + "<extra></extra>",
            showlegend=show_labels,
            flatshading=True,
            lighting=dict(diffuse=0.9, specular=0.1, ambient=0.5),
        )

    # Initial state: waiting boxes parked well above z_vis so they are hidden
    # until their turn.  The active falling box starts exactly at z=H.
    init_traces = []
    for p, c in zip(placements, colors):
        init_traces.append(_mesh(p, c, z_bottom=z_vis * 4))
    init_traces.append(_container_wireframe(L, W, H))

    # Build one frame per (box, step) pair.
    # Frame for box k at drop-step s only updates trace index k.
    # All earlier boxes (0..k-1) are already at their final z from previous
    # frames; later boxes (k+1..n-1) remain hidden above.
    frames = []
    for k, (placement, color) in enumerate(zip(placements, colors)):
        z_final = placement.position[2]
        z_start = float(H)   # box bottom enters from the container rim

        for step in range(n_steps):
            t = (step + 1) / n_steps        # 0 < t <= 1  (never exactly 0)
            t_eased = t * t                  # quadratic ease-in: accelerates like gravity
            z_current = z_start + (z_final - z_start) * t_eased

            frames.append(go.Frame(
                data=[_mesh(placement, color, z_bottom=z_current)],
                traces=[k],                  # only update this box's trace
                name=f"b{k}s{step}",
            ))

    fig = go.Figure(data=init_traces, frames=frames)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(title="X", range=[0, L], autorange=False),
            yaxis=dict(title="Y", range=[0, W], autorange=False),
            # z range extends above H so the falling box is visible on entry.
            zaxis=dict(title="Z", range=[0, z_vis], autorange=False),
            # Normalise aspect ratios to [0,1] so they stay in Plotly's
            # standard camera-unit space.  Raw dimension values (e.g. 10,10,5)
            # would scale the scene that many units, putting the camera inside.
            aspectmode="manual",
            aspectratio=dict(x=L / max(L, W, z_vis),
                             y=W / max(L, W, z_vis),
                             z=z_vis / max(L, W, z_vis)),
            # Camera eye in the same [0,1]-normalised space.  Values >1.25
            # are outside the scene; ~2.0 gives a comfortable outside view.
            camera=dict(eye=dict(x=2.0, y=2.0, z=1.5)),
        ),
        legend=dict(title="Items"),
        margin=dict(l=0, r=0, t=80, b=0),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.1,
            x=0.0,
            xanchor="left",
            buttons=[
                dict(
                    label="&#9654; Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=frame_duration_ms, redraw=True),
                            fromcurrent=True,
                            mode="immediate",
                        ),
                    ],
                ),
                dict(
                    label="&#9646;&#9646; Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                    ],
                ),
            ],
        )],
    )

    if result.unpacked_ids:
        fig.add_annotation(
            text=f"Unplaced: {', '.join(result.unpacked_ids)}",
            xref="paper", yref="paper",
            x=0.01, y=0.01,
            showarrow=False,
            font=dict(color="red", size=12),
        )

    return fig
