"""
Streamlit app — Tackler-ID Probability Explorer

Visualizes STGNN_TS's per-frame tackler probability predictions on a football
field. Hover over any defender to see their probability of making the tackle;
the model's top pick and the actual tackler are highlighted directly on the
field. Uses true (standardized) field coordinates, not play-relative ones, so
positions can be drawn on an actual football field background.

Animation (play/pause/scrub) runs entirely client-side via Plotly's native
frame animation -- no Streamlit rerun happens while scrubbing through a play,
only when you pick a different play from the sidebar.

Runs entirely offline from demo_plays/demo_plays.parquet -- a small, self-
contained bundle built once (on the full-dataset machine) via build_demo_plays.py.
No connection to the full dataset is required to run this app.

Run:
    streamlit run src/tackler_app.py
"""

from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

DEMO_PATH = Path("demo_plays/demo_plays.parquet")

st.set_page_config(page_title="Tackler-ID Explorer", layout="wide")


@st.cache_data
def load_demo() -> pl.DataFrame:
    return pl.read_parquet(DEMO_PATH)


def positions_df_from_row(row: dict) -> pl.DataFrame:
    """Rebuild a small per-frame positions table from the row's bundled list columns.
    Uses absolute (standardized) field coordinates -- x in [0, 120], y in [0, 53.3]."""
    return pl.DataFrame(
        {
            "nflId": row["pos_nfl_ids"],
            "x": row["xs"],
            "y": row["ys"],
            "side": row["sides"],
            "is_ball_carrier": row["is_ball_carriers"],
        }
    )


def add_field_background(fig: go.Figure) -> None:
    """Draw a football field: green turf, darker end zones, yard lines every 5 yards."""
    fig.add_shape(type="rect", x0=0, y0=0, x1=120, y1=53.3, line=dict(color="white"), fillcolor="#2e7d32", layer="below")
    fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=53.3, fillcolor="#1b5e20", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=110, y0=0, x1=120, y1=53.3, fillcolor="#1b5e20", line=dict(width=0), layer="below")
    for x in range(10, 111, 5):
        fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=53.3, line=dict(color="white", width=1 if x % 10 else 2), layer="below")
    fig.update_xaxes(range=[0, 120], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[0, 53.3], showgrid=False, zeroline=False, showticklabels=False)


def _traces_for_row(row: dict) -> tuple[list[go.Scatter], str]:
    """Build the 3 scatter traces (offense/ball carrier/defense) for one frame,
    plus an annotation string summarizing that frame."""
    positions_df = positions_df_from_row(row)
    prob_by_id = dict(zip(row["nfl_ids"], row["probs"]))
    pred_id = int(row["pred_tackler_nflId"])
    true_id = int(row["tacklerNflId"])

    offense = positions_df.filter(pl.col("side") > 0).filter(pl.col("is_ball_carrier") == 0)
    ball_carrier = positions_df.filter(pl.col("is_ball_carrier") == 1)
    defense = positions_df.filter(pl.col("side") < 0)

    offense_trace = go.Scatter(
        x=offense["x"], y=offense["y"], mode="markers",
        marker=dict(color="white", size=14, line=dict(width=1, color="black")),
        name="Offense",
        text=[f"nflId {r['nflId']} (offense)" for r in offense.iter_rows(named=True)],
        hoverinfo="text",
    )

    bc_trace = go.Scatter(
        x=ball_carrier["x"], y=ball_carrier["y"], mode="markers",
        marker=dict(color="black", size=18, symbol="diamond", line=dict(width=2, color="white")),
        name="Ball carrier",
        text=[f"nflId {r['nflId']} (BALL CARRIER)" for r in ball_carrier.iter_rows(named=True)],
        hoverinfo="text",
    )

    def_ids = defense["nflId"].to_list()
    def_probs = [prob_by_id.get(i, 0.0) for i in def_ids]
    line_colors, line_widths = [], []
    for i in def_ids:
        if i == pred_id and i == true_id:
            line_colors.append("limegreen")
            line_widths.append(5)
        elif i == pred_id:
            line_colors.append("red")
            line_widths.append(5)
        elif i == true_id:
            line_colors.append("gold")
            line_widths.append(5)
        else:
            line_colors.append("black")
            line_widths.append(1)

    hover_text = []
    for i, p in zip(def_ids, def_probs):
        tag = []
        if i == pred_id:
            tag.append("MODEL PICK")
        if i == true_id:
            tag.append("ACTUAL TACKLER")
        tag_str = f" ({', '.join(tag)})" if tag else ""
        hover_text.append(f"nflId {i}{tag_str}<br>Tackle probability: {p*100:.1f}%")

    def_trace = go.Scatter(
        x=defense["x"], y=defense["y"], mode="markers",
        marker=dict(
            color=def_probs, colorscale="OrRd", cmin=0, cmax=max(0.3, max(def_probs) if def_probs else 0.3),
            size=[16 + 30 * p for p in def_probs],
            line=dict(color=line_colors, width=line_widths),
            colorbar=dict(title="Tackle Prob"),
        ),
        name="Defense",
        text=hover_text,
        hoverinfo="text",
    )

    frames_to_tackle = row["tackle_frameId"] - row["frameId"]
    seconds_to_tackle = frames_to_tackle / 10.0
    top3 = sorted(prob_by_id.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top3_str = " | ".join(f"nflId {i}: {p*100:.0f}%" + (" ✅" if i == true_id else "") for i, p in top3)
    summary = (
        f"Frame {row['frameId']}  |  {seconds_to_tackle:.1f}s before tackle  |  "
        f"{'✅ Model correct' if row['correct'] else '❌ Model incorrect'}<br>"
        f"Top 3: {top3_str}"
    )

    return [offense_trace, bc_trace, def_trace], summary


def build_animated_field_figure(play_df: pl.DataFrame) -> go.Figure:
    """One self-contained Plotly figure covering the whole play, with native
    play/pause/slider animation. All frame-to-frame animation happens in the
    browser -- no Streamlit rerun is triggered while scrubbing or playing."""
    frame_rows = list(play_df.iter_rows(named=True))

    first_traces, first_summary = _traces_for_row(frame_rows[0])
    fig = go.Figure(data=first_traces)
    add_field_background(fig)

    frames = []
    for i, row in enumerate(frame_rows):
        traces, summary = _traces_for_row(row)
        frames.append(
            go.Frame(
                data=traces,
                name=str(i),
                layout=go.Layout(annotations=[dict(
                    text=summary, xref="paper", yref="paper", x=0.5, y=1.28,
                    showarrow=False, font=dict(size=13), align="center",
                )]),
            )
        )
    fig.frames = frames

    fig.update_layout(
            annotations=[dict(
                text=first_summary, xref="paper", yref="paper", x=0.5, y=1.22,
                showarrow=False, font=dict(size=13), align="center",
            )],
            xaxis_title=None, yaxis_title=None,
            yaxis=dict(scaleanchor="x", scaleratio=1, constrain="domain"),
            xaxis=dict(constrain="domain"),
            width=1600, height=711,
            margin=dict(l=10, r=10, t=90, b=90),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="#2e7d32",
            updatemenus=[dict(
                type="buttons", showactive=False, x=0.0, y=1.30, xanchor="left",
                buttons=[
                    dict(label="▶ Play", method="animate", args=[
                        None, dict(frame=dict(duration=400, redraw=True), fromcurrent=True, transition=dict(duration=0)),
                    ]),
                    dict(label="⏸ Pause", method="animate", args=[
                        [None], dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                    ]),
                ],
            )],
            sliders=[dict(
                x=0.1, y=-0.12, len=0.9, pad=dict(t=30),
                currentvalue=dict(prefix="Frame index: "),
                steps=[
                    dict(
                        method="animate", label=str(i),
                        args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                    )
                    for i in range(len(frame_rows))
                ],
            )],
        )
    return fig


def main():
    st.title("Tackler-ID Probability Explorer")

    if not DEMO_PATH.exists():
        st.error(
            f"{DEMO_PATH} not found.\n\n"
            "On Hopper4, run `python src/build_demo_plays.py`, then copy the output "
            f"file to `{DEMO_PATH}` on this machine."
        )
        return

    demo_df = load_demo()

    plays = (
        demo_df.select(["gameId", "playId", "mirrored"]).unique()
        .sort(["gameId", "playId", "mirrored"])
    )
    play_labels = [
        f"Game {r['gameId']} / Play {r['playId']}{' (mirrored)' if r['mirrored'] else ''}"
        for r in plays.iter_rows(named=True)
    ]
    selected_idx = st.sidebar.selectbox("Play", range(len(play_labels)), format_func=lambda i: play_labels[i])
    game_id, play_id, mirrored = plays.row(selected_idx)

    play_df = (
        demo_df.filter(
            (pl.col("gameId") == game_id) & (pl.col("playId") == play_id) & (pl.col("mirrored") == mirrored)
        )
        .sort("frameId")
    )
    if play_df.height == 0:
        st.warning("No frames found for this play.")
        return

    fig = build_animated_field_figure(play_df)
    st.plotly_chart(fig, use_container_width=False)


if __name__ == "__main__":
    main()