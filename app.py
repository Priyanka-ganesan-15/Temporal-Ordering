"""ChronoLogic · Research Dashboard — Streamlit front-end."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChronoLogic · Temporal Ordering Research",
    page_icon="🕰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA = ROOT / "Data"
ANALYSIS = DATA / "analysis"
ORDERING = ANALYSIS / "ordering"
SIMILARITY = ANALYSIS / "similarity"
DIAGNOSTICS = ORDERING / "diagnostics"
EMBEDDINGS = DATA / "embeddings" / "openclip"
MANIFEST = DATA / "manifests" / "sequences.json"

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE = {
    "random": "#9e9e9e",
    "greedy_nearest_neighbor": "#1976d2",
    "continuity": "#e65100",
}
METHOD_LABELS = {
    "random": "Random",
    "greedy_nearest_neighbor": "Greedy NN",
    "continuity": "Continuity",
}
DIFFICULTY_COLORS = {"easy": "#4caf50", "medium": "#ff9800", "hard": "#f44336"}

# ── Data loaders (cached) ──────────────────────────────────────────────────────
@st.cache_data
def load_manifest() -> list[dict]:
    with open(MANIFEST) as f:
        return json.load(f)

@st.cache_data
def load_results() -> pd.DataFrame:
    return pd.read_csv(ORDERING / "ordering_results.csv")

@st.cache_data
def load_by_method() -> pd.DataFrame:
    return pd.read_csv(ORDERING / "ordering_summary_by_method.csv")

@st.cache_data
def load_by_sequence() -> pd.DataFrame:
    return pd.read_csv(ORDERING / "ordering_summary_by_sequence.csv")

@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(DIAGNOSTICS / "predictions.csv")

@st.cache_data
def load_taxonomy_summary() -> pd.DataFrame:
    return pd.read_csv(DIAGNOSTICS / "error_taxonomy" / "taxonomy_summary.csv")

@st.cache_data
def load_taxonomy_rows() -> pd.DataFrame:
    return pd.read_csv(DIAGNOSTICS / "error_taxonomy" / "taxonomy_rows.csv")

@st.cache_data
def load_forward_reverse() -> pd.DataFrame:
    return pd.read_csv(DIAGNOSTICS / "forward_reverse" / "forward_reverse_scores.csv")

@st.cache_data
def load_endpoint() -> pd.DataFrame:
    return pd.read_csv(DIAGNOSTICS / "endpoint" / "endpoint_distinctiveness.csv")

@st.cache_data
def load_pairwise_errors() -> pd.DataFrame:
    return pd.read_csv(DIAGNOSTICS / "pairwise_errors" / "pairwise_error_rows.csv")

@st.cache_data
def load_embedding(seq_id: str) -> np.ndarray | None:
    path = EMBEDDINGS / f"{seq_id}.npy"
    return np.load(path) if path.exists() else None

# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine_sim_matrix(emb: np.ndarray) -> np.ndarray:
    normed = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return normed @ normed.T

def seq_label(seq_id: str) -> str:
    return seq_id.replace("sequence_", "Seq ")

def method_label(m: str) -> str:
    return METHOD_LABELS.get(m, m)

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🕰️ ChronoLogic")
    st.caption("Temporal Image Ordering Research")
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "Executive Summary",
            "Dataset Overview",
            "Embedding Pipeline",
            "Similarity Analysis",
            "Ordering Experiments",
            "Metrics & Results",
            "Diagnostics",
            "Future Directions",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Model: OpenCLIP RN50 (openai)")
    st.caption("Sequences: 7  ·  Frames: 8 per seq")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Executive Summary":
    st.title("ChronoLogic: Temporal Image Sequence Ordering")
    st.subheader("Executive Summary")

    st.markdown(
        """
        **ChronoLogic** is a vision-language benchmark and algorithm suite for **unsupervised temporal
        ordering** of image sequences. Given a shuffled set of frames from a real-world process—plant
        growth, DIY construction, time-lapse—the goal is to recover the correct chronological order
        **without any supervision signal**.
        """
    )

    # ── KPI metrics ──
    by_method = load_by_method()
    gnn = by_method[by_method["method"] == "greedy_nearest_neighbor"].iloc[0]
    cont = by_method[by_method["method"] == "continuity"].iloc[0]
    rand = by_method[by_method["method"] == "random"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Pairwise Accuracy", f"{gnn['pairwise_order_accuracy']:.1%}",
              f"+{gnn['pairwise_order_accuracy'] - rand['pairwise_order_accuracy']:.1%} vs. random",
              delta_color="normal")
    c2.metric("Best Kendall τ", f"{gnn['kendall_tau']:+.3f}", "Greedy Nearest Neighbor")
    c3.metric("Sequences Evaluated", "7", "8 frames each")
    c4.metric("Perfect Predictions", "1 / 7", "Sequence 4 — GNN exact match")

    st.divider()

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown("### Problem Statement")
        st.markdown(
            """
            Recovering temporal order from a disordered image collection is fundamental to:
            - Visual story understanding
            - Surgical training video analysis
            - Procedural task recognition
            - Time-lapse reconstruction

            This project explores purely **embedding-space** approaches using a frozen
            vision-language encoder (OpenCLIP), requiring zero supervision.
            """
        )
        st.markdown("### Engineering Contributions")
        st.markdown(
            """
            1. Modular `chronologic` Python package with clearly separated ordering, evaluation,
               and diagnostics modules
            2. Cached OpenCLIP embeddings (RN50 + OpenAI weights) — no re-computation needed
            3. Fully reproducible benchmark with fixed per-sequence shuffle seeds
            4. Rich diagnostic pipeline: error taxonomy, trajectory analysis, endpoint
               distinctiveness, and forward/reverse disambiguation
            """
        )

    with col_right:
        st.markdown("### Key Results at a Glance")
        results_df = load_results()
        summary = pd.DataFrame(
            [
                {"Method": method_label(m), "Pairwise Acc": f"{row['pairwise_order_accuracy']:.1%}",
                 "Kendall τ": f"{row['kendall_tau']:+.3f}", "Inversions (mean)": f"{row['inversion_count']:.1f}"}
                for m, row in by_method.set_index("method").iterrows()
            ]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("### Key Findings")
        st.success("**Greedy NN** beats random by **+11.2 pp** pairwise accuracy (61.2% vs 50%)")
        st.error("**Continuity exhaustive search** underperforms random at 29.1% — continuity is a poor proxy here")
        st.warning("**Forward/reverse ambiguity** is universal — all 7 sequences tie in both orientations")
        st.info("~57–86% of all method predictions fall into **'scrambled'** error — global ordering confusion")

    st.divider()
    st.markdown("### Dataset at a Glance")
    manifest = load_manifest()
    manifest_df = pd.DataFrame([
        {
            "Sequence": s["sequence_id"].replace("_", " ").title(),
            "Category": s["category"],
            "Difficulty": s["difficulty"].title(),
            "Type": s["sequence_type"].replace("_", "-").title(),
            "Caption": s["caption"],
        }
        for s in manifest
    ])
    st.dataframe(manifest_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset Overview":
    st.title("Dataset Overview")
    st.markdown(
        """
        Seven manually curated photo sequences each with **8 frames** representing a real-world
        temporal process. Frames are shuffled with a fixed per-sequence seed so all experiments
        are fully reproducible.
        """
    )
    manifest = load_manifest()
    manifest_df = pd.DataFrame([
        {
            "sequence_id": s["sequence_id"],
            "Category": s["category"],
            "Difficulty": s["difficulty"].title(),
            "Type": s["sequence_type"].replace("_", "-").title(),
            "Frames": s["num_frames"],
            "Caption": s["caption"],
        }
        for s in manifest
    ])

    c1, c2, c3 = st.columns(3)

    with c1:
        fig = px.bar(
            manifest_df.groupby("Difficulty", as_index=False).size().rename(columns={"size": "count"}),
            x="Difficulty", y="count",
            color="Difficulty",
            color_discrete_map={"Easy": "#4caf50", "Medium": "#ff9800", "Hard": "#f44336"},
            title="Sequences by Difficulty",
        )
        fig.update_layout(showlegend=False, height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            manifest_df.groupby("Category", as_index=False).size().rename(columns={"size": "count"}),
            x="Category", y="count",
            color="Category",
            title="Sequences by Category",
        )
        fig.update_layout(showlegend=False, height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        fig = px.pie(
            manifest_df.groupby("Type", as_index=False).size().rename(columns={"size": "count"}),
            names="Type", values="count",
            title="Sequence Type Split",
            color_discrete_sequence=["#5c6bc0", "#26c6da"],
        )
        fig.update_layout(height=320, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Full Manifest")
    st.dataframe(manifest_df.drop(columns="sequence_id"), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Sample Frames")
    seq_options = [s["sequence_id"] for s in manifest]
    selected_seq = st.selectbox("Choose a sequence", seq_options,
                                format_func=lambda x: x.replace("_", " ").title())
    seq_data = next(s for s in manifest if s["sequence_id"] == selected_seq)
    cols = st.columns(8)
    for i, (frame_path, col) in enumerate(zip(seq_data["frames"], cols)):
        full_path = ROOT / frame_path
        if full_path.exists():
            col.image(str(full_path), caption=f"Frame {i+1}", use_container_width=True)
        else:
            col.write(f"F{i+1}")
    st.caption(f"Caption: *{seq_data['caption']}* · Difficulty: {seq_data['difficulty'].title()} · Type: {seq_data['sequence_type'].replace('_','-').title()}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EMBEDDING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Embedding Pipeline":
    st.title("Embedding Pipeline")
    st.markdown(
        """
        All frames are encoded with **OpenCLIP (RN50, openai weights)** — a contrastively-trained
        vision-language model used *frozen*. No fine-tuning is performed; we test whether off-the-shelf
        vision-language representations contain sufficient temporal geometry for ordering.
        """
    )

    c1, c2 = st.columns([1, 1.3])

    with c1:
        st.markdown("### Why OpenCLIP?")
        st.markdown(
            """
            - **Joint image-text embedding space** — enables text-guided direction scoring
              (start / middle / end prompts)
            - **Massive pre-training** on internet-scale data with broad visual coverage
            - **Deterministic & cache-friendly**: each sequence's embeddings saved as a `.npy` file,
              eliminating re-computation
            - **Flexible**: `embed_texts()` exposes the text encoder for direction-aware ablations
            """
        )
        st.markdown("### Model Details")
        st.dataframe(pd.DataFrame([
            {"Parameter": "Architecture", "Value": "ResNet-50 (RN50)"},
            {"Parameter": "Weights", "Value": "openai"},
            {"Parameter": "Embedding dim", "Value": "1024"},
            {"Parameter": "Similarity", "Value": "Cosine"},
            {"Parameter": "Fine-tuned?", "Value": "No — frozen"},
            {"Parameter": "Batch size", "Value": "32"},
        ]), hide_index=True, use_container_width=True)

    with c2:
        st.markdown("### Pipeline")
        st.code(
            """Image Frames
    ↓
OpenCLIP Encoder  (RN50, frozen)
    ↓
1024-d L2-normalised embeddings  [8 × 1024]
    ↓  cached to Data/embeddings/openclip/
Cosine Similarity Matrix          [8 × 8]
    ↓
Ordering Algorithms         Diagnostics
• Greedy NN                 • Trajectory
• Continuity Exhaustive     • Endpoint
• Random Baseline           • Error Taxonomy
                            • Forward/Reverse""",
            language="text",
        )

    st.divider()
    st.markdown("### Cached Embedding Statistics")
    manifest = load_manifest()
    embed_rows = []
    for s in manifest:
        emb = load_embedding(s["sequence_id"])
        if emb is not None:
            norms = np.linalg.norm(emb, axis=1)
            embed_rows.append({
                "Sequence": s["sequence_id"].replace("_", " ").title(),
                "Shape": str(emb.shape),
                "Dtype": str(emb.dtype),
                "Norm (mean)": round(float(norms.mean()), 4),
                "Norm (std)": round(float(norms.std()), 4),
            })
    st.dataframe(pd.DataFrame(embed_rows), hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("### Embedding Norms per Frame (Ground-Truth Order)")
    fig = go.Figure()
    for s in manifest:
        emb = load_embedding(s["sequence_id"])
        if emb is not None:
            norms = np.linalg.norm(emb, axis=1)
            fig.add_trace(go.Scatter(
                x=list(range(1, len(norms)+1)),
                y=norms.tolist(),
                mode="lines+markers",
                name=s["sequence_id"].replace("_", " ").title(),
            ))
    fig.update_layout(
        xaxis_title="Frame Index",
        yaxis_title="L2 Norm",
        title="OpenCLIP Embedding Norms per Frame",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Embedding PCA Projection (2D)")
    st.markdown("Principal-component projection of all frame embeddings coloured by sequence.")
    try:
        from sklearn.decomposition import PCA
        all_embs, labels_seq, labels_frame = [], [], []
        for s in manifest:
            emb = load_embedding(s["sequence_id"])
            if emb is not None:
                all_embs.append(emb)
                labels_seq.extend([s["sequence_id"].replace("_", " ").title()] * len(emb))
                labels_frame.extend(list(range(1, len(emb)+1)))
        X = np.vstack(all_embs)
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X)
        pca_df = pd.DataFrame({"PC1": X2[:,0], "PC2": X2[:,1],
                               "Sequence": labels_seq, "Frame": labels_frame})
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="Sequence",
                         symbol="Frame", hover_data=["Frame"],
                         title=f"2D PCA of OpenCLIP Frame Embeddings (var: {pca.explained_variance_ratio_.sum()*100:.1f}%)")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("Install scikit-learn (`pip install scikit-learn`) to enable PCA projection.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SIMILARITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Similarity Analysis":
    st.title("Similarity Analysis")
    st.markdown(
        """
        For each sequence we compute the **pairwise cosine similarity matrix** over frames in
        ground-truth order. Two diagnostic statistics characterise the temporal structure:

        | Stat | Definition |
        |---|---|
        | **Near-mean** | Average similarity of adjacent frame pairs (\\|i − j\\| = 1) |
        | **Far-mean** | Average similarity of distant frame pairs (\\|i − j\\| ≥ 4) |
        | **TCS** (Temporal Contrast Score) | `near_mean − far_mean` — higher = stronger temporal signal |

        A high TCS means nearby frames look alike and distant frames look different — the property
        that makes greedy nearest-neighbour ordering tractable.
        """
    )

    results_df = load_results()
    sim_df = (
        results_df.drop_duplicates(subset="sequence_id")
        [["sequence_id", "category", "difficulty", "sequence_type",
          "temporal_near_mean", "temporal_far_mean", "temporal_contrast_score"]]
        .reset_index(drop=True)
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = px.scatter(
            sim_df,
            x="temporal_near_mean", y="temporal_far_mean",
            color="difficulty",
            color_discrete_map=DIFFICULTY_COLORS,
            size="temporal_contrast_score",
            size_max=35,
            text="sequence_id",
            hover_data=["category", "temporal_contrast_score"],
            title="Near vs. Far Similarity (bubble = TCS)",
        )
        fig.update_traces(textposition="top center", textfont_size=10)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            sim_df.sort_values("temporal_contrast_score", ascending=True),
            x="temporal_contrast_score", y="sequence_id",
            orientation="h",
            color="temporal_contrast_score",
            color_continuous_scale="RdYlGn",
            title="Temporal Contrast Score per Sequence",
            labels={"temporal_contrast_score": "TCS", "sequence_id": ""},
        )
        mean_tcs = sim_df["temporal_contrast_score"].mean()
        fig.add_vline(x=mean_tcs, line_dash="dash", line_color="#555",
                      annotation_text=f"Mean {mean_tcs:.3f}")
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### TCS Summary Table")
    display_sim = sim_df.copy()
    display_sim.columns = ["Sequence", "Category", "Difficulty", "Type", "Near Mean", "Far Mean", "TCS"]
    display_sim[["Near Mean", "Far Mean", "TCS"]] = display_sim[["Near Mean", "Far Mean", "TCS"]].round(4)
    st.dataframe(display_sim, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("### Similarity Heatmaps")
    col_seq, col_method = st.columns(2)

    with col_seq:
        manifest = load_manifest()
        seq_options = [s["sequence_id"] for s in manifest]
        selected_seq = st.selectbox("Sequence", seq_options,
                                    format_func=lambda x: x.replace("_", " ").title())

    heatmap_img_path = SIMILARITY / f"{selected_seq}_heatmap.png"
    emb = load_embedding(selected_seq)
    if emb is not None:
        sim_mat = cosine_sim_matrix(emb)
        labels = [f"F{i+1}" for i in range(len(emb))]
        fig = px.imshow(
            sim_mat,
            x=labels, y=labels,
            color_continuous_scale="viridis",
            zmin=0.6, zmax=1.0,
            text_auto=".2f",
            title=f"Pairwise Cosine Similarity — {selected_seq.replace('_',' ').title()}",
        )
        fig.update_layout(height=440)
        st.plotly_chart(fig, use_container_width=True)

        seq_meta = next(s for s in manifest if s["sequence_id"] == selected_seq)
        tcs_val = sim_df[sim_df["sequence_id"] == selected_seq]["temporal_contrast_score"].values[0]
        near_val = sim_df[sim_df["sequence_id"] == selected_seq]["temporal_near_mean"].values[0]
        far_val = sim_df[sim_df["sequence_id"] == selected_seq]["temporal_far_mean"].values[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Near Mean", f"{near_val:.4f}")
        m2.metric("Far Mean", f"{far_val:.4f}")
        m3.metric("TCS", f"{tcs_val:.4f}",
                  "High" if tcs_val > mean_tcs else "Below avg")

    st.divider()
    st.markdown(
        """
        ### Interpretation

        **Sequence 2 (Plant Growth)** has the highest TCS at **0.174** — monotonic biological growth
        creates a clear similarity gradient from early to late frames. **Sequence 7 (Coffee)**
        has the lowest at **0.036** — the coffee-making steps look visually similar throughout,
        leaving almost no geometric gradient in embedding space for ordering to exploit.

        > The low mean TCS ≈ 0.10 across all sequences explains why no method exceeds 61% pairwise
        > accuracy: the embedding space simply does not encode strong enough temporal separation
        > with a frozen RN50 model.
        """
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ORDERING EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Ordering Experiments":
    st.title("Ordering Experiments")
    st.markdown("Three ordering methods were evaluated across all 7 sequences.")

    tab1, tab2, tab3 = st.tabs(["Method Descriptions", "Predictions", "Path Scores"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 🎲 Random Baseline")
            st.markdown(
                """
                Draws a **uniformly random permutation** of frames using a fixed per-sequence seed.
                No embedding information is used.

                - Expected pairwise accuracy ≈ **50%** for 8 frames
                - Establishes the lower bound for all methods
                - Seed: 42 + sequence index
                """
            )
        with c2:
            st.markdown("#### 🔵 Greedy Nearest Neighbor (GNN)")
            st.markdown(
                """
                Builds a Hamiltonian path through the embedding similarity graph using a **greedy
                forward strategy**. Starting from every possible first frame, always visits the
                most similar unvisited frame next.

                **Objective**: maximise `path_score = Σ sim(fᵢ, fᵢ₊₁)`

                The best path across all 8 starting nodes is returned.
                """
            )
        with c3:
            st.markdown("#### 🟠 Continuity Exhaustive Search")
            st.markdown(
                """
                Enumerates all **8! = 40,320 permutations** and ranks by a composite score:
                """
            )
            st.latex(r"score = w_{adj}\cdot adj + w_{cont}\cdot cont + w_{dir}\cdot dir + w_{ep}\cdot ep")
            st.markdown(
                """
                Base config: `w_adj = w_cont = 1.0`, `w_dir = w_ep = 0.0`.

                Guarantees the **globally optimal** solution under this score function.
                """
            )

        st.divider()
        st.markdown("#### Advanced Ablations (Diagnostic Variants)")
        st.markdown(
            """
            | Variant | Change from base |
            |---|---|
            | `continuity_plus_direction` | Adds text-direction scoring `w_dir = 0.75` from start/middle/end CLIP prompts |
            | `continuity_plus_endpoint` | Adds endpoint distinctiveness weighting `w_ep = 0.75` |
            | `continuity_plus_reverse_disambiguation` | Applies forward/reverse tiebreaking after continuity ranking |
            """
        )

    with tab2:
        pred_df = load_predictions()
        pred_display = pred_df.copy()
        pred_display["predicted_order"] = pred_display["predicted_order"].apply(
            lambda x: " → ".join([str(int(i)+1) for i in str(x).split()])
        )
        pred_display["score"] = pred_display["score"].apply(
            lambda x: f"{float(x):.3f}" if str(x) != "nan" else "—"
        )
        pred_display.columns = ["Sequence", "Method", "Predicted Order (1-indexed)", "Path Score"]
        pred_display["Method"] = pred_display["Method"].map(METHOD_LABELS)

        seq_filter = st.multiselect(
            "Filter by sequence",
            options=pred_df["sequence_id"].unique().tolist(),
            default=pred_df["sequence_id"].unique().tolist(),
            format_func=lambda x: x.replace("_", " ").title(),
        )
        filtered = pred_display[pred_df["sequence_id"].isin(seq_filter)]
        st.caption("Ground truth for all sequences: Frame 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8")
        st.dataframe(filtered, hide_index=True, use_container_width=True)

    with tab3:
        results_df = load_results()
        path_df = results_df[results_df["path_score"].notna()].copy()
        path_df["seq_label"] = path_df["sequence_id"].str.replace("sequence_", "Seq ")
        path_df["method_label"] = path_df["method"].map(METHOD_LABELS)

        fig = px.bar(
            path_df,
            x="seq_label", y="path_score",
            color="method_label",
            barmode="group",
            color_discrete_map={v: PALETTE[k] for k, v in METHOD_LABELS.items() if k != "random"},
            title="Path Score (Σ Adjacent Cosine Similarity) per Sequence",
            labels={"seq_label": "Sequence", "path_score": "Path Score", "method_label": "Method"},
        )
        fig.update_layout(height=420, legend_title="Method")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            **GNN consistently achieves higher path scores than Continuity** because it directly
            optimises the adjacency objective. However, a higher path score does not always translate
            into better ranking accuracy — showing that the adjacency proxy has limits.
            """
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — METRICS & RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Metrics & Results":
    st.title("Metrics & Results")
    st.markdown(
        """
        Four complementary metrics are reported per (sequence, method) pair:

        | Metric | Range | Random Baseline | Interpretation |
        |---|---|---|---|
        | **Exact Match Accuracy** | {0, 1} | ≈ 0 | 1 only when order is perfectly correct |
        | **Pairwise Order Accuracy** | [0, 1] | 0.5 | Fraction of concordant frame pairs |
        | **Normalized Kendall Agreement** | [0, 1] | 0.5 | Same as pairwise accuracy |
        | **Kendall τ** | [−1, +1] | 0.0 | +1 = perfect, −1 = reversed, 0 = random |
        """
    )

    by_method = load_by_method()
    results_df = load_results()

    tab1, tab2, tab3 = st.tabs(["Method Comparison", "Per-Sequence Heatmap", "Raw Results"])

    with tab1:
        c1, c2 = st.columns(2)
        method_order = ["random", "greedy_nearest_neighbor", "continuity"]

        with c1:
            fig = go.Figure()
            for m in method_order:
                row = by_method[by_method["method"] == m].iloc[0]
                fig.add_trace(go.Bar(
                    name=METHOD_LABELS[m],
                    x=[METHOD_LABELS[m]],
                    y=[row["pairwise_order_accuracy"]],
                    marker_color=PALETTE[m],
                    text=f"{row['pairwise_order_accuracy']:.3f}",
                    textposition="outside",
                ))
            fig.add_hline(y=0.5, line_dash="dash", line_color="#888",
                          annotation_text="Random baseline (0.5)")
            fig.update_layout(
                title="Pairwise Order Accuracy",
                yaxis=dict(range=[0, 1.1]),
                showlegend=False, height=360,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure()
            for m in method_order:
                row = by_method[by_method["method"] == m].iloc[0]
                fig.add_trace(go.Bar(
                    name=METHOD_LABELS[m],
                    x=[METHOD_LABELS[m]],
                    y=[row["kendall_tau"]],
                    marker_color=PALETTE[m],
                    text=f"{row['kendall_tau']:+.3f}",
                    textposition="outside",
                ))
            fig.add_hline(y=0, line_dash="dash", line_color="#888",
                          annotation_text="Random baseline (0)")
            fig.update_layout(
                title="Kendall τ",
                yaxis=dict(range=[-0.7, 0.7]),
                showlegend=False, height=360,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        c3, c4 = st.columns(2)
        with c3:
            fig = px.bar(
                by_method, x="method", y="inversion_count",
                color="method",
                color_discrete_map=PALETTE,
                text="inversion_count",
                title="Mean Inversion Count (lower = better, max = 28)",
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.add_hline(y=14, line_dash="dash", line_color="#888",
                          annotation_text="Expected random (14)")
            fig.update_layout(showlegend=False, height=350,
                               xaxis=dict(ticktext=list(METHOD_LABELS.values()),
                                          tickvals=list(METHOD_LABELS.keys())))
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            fig = px.bar(
                by_method, x="method", y="exact_match_accuracy",
                color="method",
                color_discrete_map=PALETTE,
                text="exact_match_accuracy",
                title="Exact Match Accuracy (fraction of sequences perfectly ordered)",
            )
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(showlegend=False, height=350,
                               xaxis=dict(ticktext=list(METHOD_LABELS.values()),
                                          tickvals=list(METHOD_LABELS.keys())))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        metric = st.selectbox("Metric", ["kendall_tau", "pairwise_order_accuracy",
                                          "normalized_kendall_agreement", "inversion_count"])
        pivot = results_df.pivot_table(
            index="sequence_id", columns="method", values=metric
        ).reindex(columns=method_order)
        pivot.index = pivot.index.str.replace("sequence_", "Seq ")
        pivot.columns = [METHOD_LABELS[m] for m in method_order]

        colorscale = "RdYlGn_r" if metric == "inversion_count" else "RdYlGn"
        center = 14.0 if metric == "inversion_count" else (0.0 if "tau" in metric else 0.5)
        fig = px.imshow(
            pivot,
            text_auto=".2f",
            color_continuous_scale=colorscale,
            title=f"{metric.replace('_', ' ').title()} — Sequence × Method",
            aspect="auto",
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        col_sel = st.selectbox("Select sequence for per-frame detail", results_df["sequence_id"].unique(),
                               format_func=lambda x: x.replace("_", " ").title())
        sub = results_df[results_df["sequence_id"] == col_sel]
        st.dataframe(
            sub[["method", "predicted_order", "path_score", "kendall_tau",
                  "pairwise_order_accuracy", "inversion_count"]]
            .assign(method=sub["method"].map(METHOD_LABELS))
            .round(4),
            hide_index=True, use_container_width=True,
        )

    with tab3:
        st.dataframe(
            results_df.assign(method=results_df["method"].map(METHOD_LABELS))
            .round(4),
            hide_index=True, use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Diagnostics":
    st.title("Diagnostics")
    st.markdown(
        "Four diagnostic modules probe *why* methods succeed or fail beyond headline metrics."
    )

    diag_tab1, diag_tab2, diag_tab3, diag_tab4 = st.tabs(
        ["Error Taxonomy", "Forward / Reverse", "Endpoint Distinctiveness", "Pairwise Errors"]
    )

    # ── Error Taxonomy ────────────────────────────────────────────────────────
    with diag_tab1:
        st.markdown(
            """
            ### Error Taxonomy

            Each prediction is categorised into one of five failure modes:

            | Category | Definition |
            |---|---|
            | **exact** | Perfect match with ground truth |
            | **reversed** | Prediction is the exact reverse of ground truth |
            | **local_swap** | Only one adjacent pair swapped |
            | **endpoint_error** | Only the first or last frame is wrong |
            | **scrambled** | None of the above — global ordering confusion |
            """
        )

        taxonomy_summary = load_taxonomy_summary()
        taxonomy_rows = load_taxonomy_rows()
        tax_order = ["exact", "endpoint_error", "local_swap", "reversed", "scrambled"]
        method_order = ["random", "greedy_nearest_neighbor", "continuity"]

        c1, c2 = st.columns([1.5, 1])
        with c1:
            fig = px.bar(
                taxonomy_summary[taxonomy_summary["method"].isin(method_order)],
                x="method", y="fraction",
                color="taxonomy",
                color_discrete_sequence=px.colors.qualitative.Set2,
                barmode="stack",
                title="Error Taxonomy Breakdown by Method",
                labels={"method": "Method", "fraction": "Fraction of Sequences",
                        "taxonomy": "Error Type"},
                category_orders={"taxonomy": tax_order, "method": method_order},
            )
            fig.update_xaxes(
                ticktext=[METHOD_LABELS[m] for m in method_order],
                tickvals=method_order,
            )
            fig.update_layout(height=420, legend_title="Error Type")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            per_seq = taxonomy_rows.copy()
            per_seq["method"] = per_seq["method"].map(METHOD_LABELS)
            fig2 = px.scatter(
                per_seq,
                x="method", y="sequence_id",
                color="taxonomy",
                symbol="taxonomy",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Per-Sequence Error Categories",
                labels={"method": "", "sequence_id": "", "taxonomy": "Error"},
                size_max=14,
            )
            fig2.update_traces(marker_size=14)
            fig2.update_layout(height=420, legend_title="Error Type")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            """
            **Observation:** The _scrambled_ category dominates all methods. This tells us that
            failures are not systematic — no method consistently produces reversals or simple swaps.
            Instead, failures arise from fundamental confusion in embedding space where no clear
            temporal gradient exists.

            **GNN achieves 2 exact matches** (Seq 2 and Seq 4) while **Continuity gets 1** (Seq 4),
            confirming that GNN's greedy path traversal is a better proxy for temporal order than
            exhaustive continuity optimisation on this dataset.
            """
        )

    # ── Forward / Reverse ─────────────────────────────────────────────────────
    with diag_tab2:
        st.markdown(
            """
            ### Forward / Reverse Disambiguation

            The **forward/reverse ambiguity** arises because cosine similarity is symmetric:
            if a sequence looks visually smooth forward, the reversed sequence scores identically.
            We check whether the continuity score can distinguish orientation.
            """
        )

        fwd_rev = load_forward_reverse()
        c1, c2 = st.columns([1.4, 1])

        with c1:
            fig = go.Figure()
            labels = fwd_rev["sequence_id"].str.replace("sequence_", "Seq ")
            fig.add_trace(go.Bar(name="Forward", x=labels,
                                  y=fwd_rev["forward_score"], marker_color="#4caf50"))
            fig.add_trace(go.Bar(name="Reverse", x=labels,
                                  y=fwd_rev["reverse_score"], marker_color="#ef5350",
                                  opacity=0.6))
            fig.update_layout(
                barmode="group",
                title="Forward vs. Reverse Path Scores per Sequence",
                yaxis_title="Path Score",
                height=380, legend_title="Orientation",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.bar(
                fwd_rev,
                x="sequence_id", y="forward_minus_reverse",
                color="forward_minus_reverse",
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                title="Forward − Reverse Score Gap",
                labels={"sequence_id": "", "forward_minus_reverse": "Gap"},
            )
            fig2.add_hline(y=0, line_dash="solid", line_color="#333")
            fig2.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.error(
            "**Critical Finding:** All 7 sequences score **identically** in forward and reverse "
            "(gap = 0.0, winner = 'tie'). The continuity/adjacency objective has **zero discriminative "
            "power for temporal orientation**. Text-grounded direction scoring is essential to break this symmetry."
        )

        st.markdown(
            """
            #### Why Does This Happen?

            The cosine similarity matrix is by definition symmetric: `sim(A, B) = sim(B, A)`.
            The path score `Σ sim(fᵢ, fᵢ₊₁)` over a reverse path visits the same *pairs* in
            reverse order — but because addition is commutative the total is identical.

            **The only way to break this tie is an asymmetric signal**, such as:
            - Text-direction prompts: `"start of X"` vs. `"end of X"` using CLIP's language encoder
            - A learned pairwise ordering model
            - Optical flow or temporal motion cues
            """
        )

    # ── Endpoint Distinctiveness ───────────────────────────────────────────────
    with diag_tab3:
        st.markdown(
            """
            ### Endpoint Distinctiveness

            A well-structured temporal sequence should have **edge frames** (first and last) that
            are visually distinct from the cluster of middle frames — sitting farther from the
            embedding centroid. This property would allow anchor-finding heuristics to seed ordering.
            """
        )

        endpoint_df = load_endpoint()
        manifest = load_manifest()

        selected_seq = st.selectbox(
            "Select sequence",
            [s["sequence_id"] for s in manifest],
            format_func=lambda x: x.replace("_", " ").title(),
            key="ep_seq",
        )
        grp = endpoint_df[endpoint_df["sequence_id"] == selected_seq].copy()
        grp["frame_type"] = grp["is_true_endpoint"].map({1: "True Endpoint (F1 / F8)", 0: "Middle Frame"})

        fig = px.bar(
            grp, x="frame_index", y="distance_to_centroid",
            color="frame_type",
            color_discrete_map={"True Endpoint (F1 / F8)": "#f44336", "Middle Frame": "#1976d2"},
            title=f"Distance to Embedding Centroid — {selected_seq.replace('_',' ').title()}",
            labels={"frame_index": "Frame Index", "distance_to_centroid": "Distance to Centroid"},
        )
        fig.update_layout(height=350, legend_title="Frame Type")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### Aggregate: Are True Endpoints Most Distinct?")

        summary_rows = []
        for seq_id, grp in endpoint_df.groupby("sequence_id"):
            ep_dist = grp[grp["is_true_endpoint"] == 1]["distance_to_centroid"].mean()
            mid_dist = grp[grp["is_true_endpoint"] == 0]["distance_to_centroid"].mean()
            max_f = grp.loc[grp["distance_to_centroid"].idxmax(), "frame_index"]
            is_ep = max_f in grp[grp["is_true_endpoint"] == 1]["frame_index"].values
            summary_rows.append({
                "Sequence": seq_id.replace("sequence_", "Seq "),
                "Avg Endpoint Dist": round(ep_dist, 4),
                "Avg Middle Dist": round(mid_dist, 4),
                "Endpoint > Middle?": "✓" if ep_dist > mid_dist else "✗",
                "Most Distinct = Endpoint?": "✓" if is_ep else "✗",
            })
        ep_sum = pd.DataFrame(summary_rows)
        st.dataframe(ep_sum, hide_index=True, use_container_width=True)

        n_correct = ep_sum["Endpoint > Middle?"].value_counts().get("✓", 0)
        st.info(
            f"**{n_correct}/{len(ep_sum)} sequences** have true endpoints with higher mean distance "
            f"to centroid than middle frames — indicating partial but not universal endpoint signal."
        )

        # All sequences overview
        fig_all = px.box(
            endpoint_df,
            x="sequence_id", y="distance_to_centroid",
            color=endpoint_df["is_true_endpoint"].map({1: "Endpoint", 0: "Middle"}),
            color_discrete_map={"Endpoint": "#f44336", "Middle": "#1976d2"},
            title="Distribution of Centroid Distance — Endpoints vs. Middle Frames (all sequences)",
            labels={"sequence_id": "Sequence", "distance_to_centroid": "Distance to Centroid"},
        )
        fig_all.update_layout(height=380, legend_title="Frame Type")
        st.plotly_chart(fig_all, use_container_width=True)

    # ── Pairwise Errors ───────────────────────────────────────────────────────
    with diag_tab4:
        st.markdown(
            """
            ### Pairwise Error Analysis

            For each ordered pair of frames `(i, j)` where `i < j` in ground truth, we check
            whether the predicted ordering places `i` before `j`. This gives a per-pair binary
            correctness matrix, visualised as a heatmap.
            """
        )

        pw_df = load_pairwise_errors()
        manifest = load_manifest()

        col1, col2 = st.columns(2)
        with col1:
            selected_seq = st.selectbox(
                "Sequence", [s["sequence_id"] for s in manifest],
                format_func=lambda x: x.replace("_", " ").title(), key="pw_seq"
            )
        with col2:
            selected_method = st.selectbox(
                "Method", ["greedy_nearest_neighbor", "continuity", "random"],
                format_func=lambda m: METHOD_LABELS[m], key="pw_method"
            )

        sub = pw_df[(pw_df["sequence_id"] == selected_seq) & (pw_df["method"] == selected_method)]
        if not sub.empty:
            n = 8
            matrix = np.full((n, n), np.nan)
            for _, row in sub.iterrows():
                i, j = int(row["frame_i"]), int(row["frame_j"])
                matrix[i, j] = int(row["is_correct"])

            fig = px.imshow(
                matrix,
                color_continuous_scale=[[0, "#f44336"], [1, "#4caf50"]],
                zmin=0, zmax=1,
                title=f"Pairwise Correctness — {selected_seq.replace('_',' ').title()} · {METHOD_LABELS[selected_method]}",
                labels={"x": "Frame j", "y": "Frame i", "color": "Correct?"},
                x=[f"F{i+1}" for i in range(n)],
                y=[f"F{i+1}" for i in range(n)],
                text_auto=True,
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Green = pair ordered correctly, Red = pair inverted, Grey = not applicable (i ≥ j)")

        # Error rate summary
        st.divider()
        st.markdown("### Overall Pairwise Error Rate by Method")
        err_summary = (
            pw_df.groupby("method")["is_correct"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "accuracy", "count": "total_pairs"})
        )
        err_summary["error_rate"] = 1 - err_summary["accuracy"]
        err_summary["method_label"] = err_summary["method"].map(METHOD_LABELS)
        fig = px.bar(
            err_summary, x="method_label", y="error_rate",
            color="method_label",
            color_discrete_map={v: PALETTE[k] for k, v in METHOD_LABELS.items()},
            text=err_summary["error_rate"].round(3),
            title="Pairwise Error Rate (lower = better)",
            labels={"method_label": "Method", "error_rate": "Error Rate"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=340, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        # Trajectory images
        st.divider()
        st.markdown("### Trajectory & Alignment Plots")
        col_a, col_b = st.columns(2)
        with col_a:
            traj_seq = st.selectbox(
                "Select sequence for trajectory",
                [s["sequence_id"] for s in manifest],
                format_func=lambda x: x.replace("_", " ").title(), key="traj_seq"
            )
        traj_path = DIAGNOSTICS / "trajectory" / f"{traj_seq}_trajectory.png"
        align_path = DIAGNOSTICS / "alignment" / f"{traj_seq}_alignment.png"
        r1, r2 = st.columns(2)
        with r1:
            if traj_path.exists():
                st.image(str(traj_path), caption="Embedding Trajectory", use_container_width=True)
            else:
                st.info("Trajectory image not found.")
        with r2:
            if align_path.exists():
                st.image(str(align_path), caption="Alignment Plot", use_container_width=True)
            else:
                st.info("Alignment image not found.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — FUTURE DIRECTIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Future Directions":
    st.title("Future Optimizations & Research Directions")
    st.markdown(
        "Based on the diagnostic findings, six concrete improvement paths are identified, "
        "with estimated pairwise accuracy gains."
    )

    # ── Roadmap chart ──
    roadmap = {
        "Current Greedy NN": 0.612,
        "+ Text Direction\n(CLIP prompts)": 0.68,
        "+ TSP / Beam Search": 0.70,
        "+ ViT-L/14\nEmbeddings": 0.76,
        "+ Pairwise\nRanking Model": 0.82,
        "+ Full Dataset\n& Fine-tuning": 0.88,
    }
    roadmap_df = pd.DataFrame({"Step": list(roadmap.keys()), "Pairwise Accuracy": list(roadmap.values())})
    fig = px.bar(
        roadmap_df, x="Pairwise Accuracy", y="Step", orientation="h",
        text=roadmap_df["Pairwise Accuracy"].apply(lambda v: f"{v:.0%}"),
        color="Pairwise Accuracy",
        color_continuous_scale="Blues",
        title="Estimated Performance Roadmap (projected gains from future work)",
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="#aaa",
                  annotation_text="Random baseline")
    fig.add_vline(x=0.612, line_dash="dot", line_color="#1976d2",
                  annotation_text="Current GNN")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=380, coloraxis_showscale=False, xaxis_range=[0.3, 1.0])
    st.plotly_chart(fig, use_container_width=True)

    st.caption("⚠️ Projected values are illustrative estimates based on related-work benchmarks. Actual gains depend on implementation quality.")

    st.divider()

    # ── Individual directions ──
    d1, d2 = st.columns(2)

    with d1:
        with st.expander("🔤 1. Text-Grounded Direction Scoring", expanded=True):
            st.markdown(
                """
                The forward/reverse ambiguity (gap = 0 on all 7 sequences) is the most critical
                failure mode. The `build_directional_evidence()` function exists in the codebase
                but was inactive (`w_dir = 0.0`) in all reported experiments.

                **Fix**: enable direction scoring using CLIP's text encoder with captioned prompts:
                ```python
                weights = ContinuityScoreWeights(
                    adjacency=1.0, continuity=1.0,
                    direction=0.75, endpoint=0.5
                )
                ```
                Text prompts like `"start of building a house"` and `"end of building a house"`
                can break forward/reverse symmetry for captioned sequences.

                **Expected gain**: +5–8 pp pairwise accuracy on procedural sequences.
                """
            )

        with st.expander("🔢 2. Better Embeddings"):
            st.markdown(
                """
                | Model | Expected TCS Improvement | Notes |
                |---|---|---|
                | ViT-L/14 OpenCLIP (laion2b) | +50–100% | Richer visual features |
                | DINOv2 ViT-L | +30–70% | Better scene geometry |
                | VideoMAE / TimeSformer | High | Video-native temporal encoding |
                | Fine-tuned on temporal pairs | Very high | Would need labeled data |

                The current mean TCS ≈ 0.10 is very low. Even moving to ViT-L/14 should
                meaningfully increase the near/far similarity gap.
                """
            )

        with st.expander("🌐 3. Global Optimization (TSP / Beam Search)"):
            st.markdown(
                """
                The greedy path is a heuristic. For n=8 frames, the **exact TSP solution**
                is tractable:

                - **Exact**: enumerate all 8! = 40,320 paths (already done in Continuity)
                - **Beam search** with `k` beams for n > 15
                - **Christofides algorithm**: 3/2-approximation for general n
                - **OR-Tools**: off-the-shelf TSP solver

                The key issue is not the search strategy but the **objective function**.
                Replacing adjacency with a learned score would have the biggest impact.
                """
            )

    with d2:
        with st.expander("🤖 4. Learned Pairwise Ranking Model", expanded=True):
            st.markdown(
                """
                Train a **pairwise classifier** that predicts whether frame A comes before frame B:

                ```
                Input:  [embed(A), embed(B)]  (2048-d concat)
                Output: P(A before B)         (binary)
                ```

                Once trained:
                - Use **topological sort** or **Kemeny-optimal rank aggregation**
                - Pre-train on large video datasets (Kinetics, HowTo100M) and zero-shot transfer
                - No per-sequence labelling needed at inference time

                This converts ordering into a standard classification pipeline and avoids
                the symmetry problem entirely.

                **Expected gain**: +15–25 pp pairwise accuracy with good pre-training data.
                """
            )

        with st.expander("📦 5. Expand & Balance the Dataset"):
            st.markdown(
                """
                Current limitations:
                - All sequences have exactly **8 frames** — variable lengths (5–20) needed
                - Heavy DIY/procedural bias (5 of 7 sequences)
                - No train / val / test split — can only report leave-one-out estimates

                **Recommended expansion**:
                - 50+ sequences across 10+ categories (cooking, medical, nature, sports)
                - 3-way stratified split by difficulty and sequence type
                - Include **distractor frames** to test robustness
                - Human performance baseline (annotators on MTurk)

                With N=7, results are directional hypotheses — statistical significance
                requires at minimum N≥30 per condition.
                """
            )

        with st.expander("🔀 6. Ensemble & Hybrid Methods"):
            st.markdown(
                """
                Individual methods have complementary failure modes:
                - **GNN** fails on non-monotonic embedding spaces
                - **Continuity** fails globally but might succeed locally

                **Proposed 3-stage hybrid**:
                1. GNN generates k candidate paths from different start nodes
                2. Re-rank candidates with continuity + direction + endpoint composite score
                3. Apply text-grounded forward/reverse disambiguation as final tiebreaker

                This keeps GNN's global path structure while using continuity as a
                local refinement oracle.
                """
            )

    st.divider()
    st.markdown("### Evaluation Improvements")
    st.markdown(
        """
        | Current Gap | Proposed Fix |
        |---|---|
        | Only 4 metrics reported | Add Spearman ρ, Average Displacement Error (ADE), partial credit (within ±1 position) |
        | No confidence intervals | Bootstrap resampling over N sequences |
        | No human upper bound | Human MTurk annotation as ceiling |
        | Single fixed seed per sequence | Multi-seed evaluation with variance reporting |
        | All sequences same length | Variable n with difficulty stratification |
        """
    )

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### Reproducibility — Full Pipeline")
        st.code(
            """# 1. Install dependencies
pip install -e ".[dev]"

# 2. Embed all sequences
python embedder.py

# 3. Similarity analysis
python scripts/similarity_matrices.py

# 4. Ordering benchmark
python evaluate_ordering.py

# 5. Full diagnostics
python scripts/run_diagnostics.py

# 6. Launch this dashboard
streamlit run app.py""",
            language="bash",
        )
    with col_r:
        st.markdown("### Key Hyperparameters")
        st.dataframe(pd.DataFrame([
            {"Parameter": "Encoder", "Value": "OpenCLIP RN50 (openai)"},
            {"Parameter": "Embedding dim", "Value": "1024"},
            {"Parameter": "Similarity metric", "Value": "Cosine"},
            {"Parameter": "Near-gap", "Value": "1 (adjacent frames)"},
            {"Parameter": "Far-gap", "Value": "4 (distant frames)"},
            {"Parameter": "Random seeds", "Value": "42–48 (seq 1–7)"},
            {"Parameter": "Continuity w_adj", "Value": "1.0"},
            {"Parameter": "Continuity w_cont", "Value": "1.0"},
            {"Parameter": "Continuity w_dir", "Value": "0.0 (inactive)"},
            {"Parameter": "Continuity w_ep", "Value": "0.0 (inactive)"},
        ]), hide_index=True, use_container_width=True)
