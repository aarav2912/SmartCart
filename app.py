from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATHS = [
    APP_DIR / "smartcart_customers.csv",
    Path.cwd() / "smartcart_customers.csv",
    Path.home() / "Desktop" / "smartcart_customers.csv",
]

SPEND_COLS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]

PRODUCT_COLS = [
    "MntWines",
    "MntMeatProducts",
    "MntGoldProds",
    "MntFruits",
    "MntFishProducts",
    "MntSweetProducts",
]

PRODUCT_LABELS = ["Wine", "Meat", "Gold", "Fruits", "Fish", "Sweets"]

FEATURES = [
    "Income",
    "Total_Spending",
    "Total_Purchases",
    "Recency",
    "Age",
    "Total_Children",
    "NumWebVisitsMonth",
]

COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#8D5A97", "#F4A261", "#264653", "#6A994E"]

ALGORITHM_OPTIONS = {
    "K-Means": {"label_key": "labels_km", "score_key": "sil_km"},
    "Agglomerative (Ward)": {"label_key": "labels_agg", "score_key": "sil_agg"},
    "GMM": {"label_key": "labels_gmm", "score_key": "sil_gmm"},
}

STRATEGIES = {
    "The Elite": {
        "profile": "Highest income, spending, and purchases. They buy the most wine and meat and visit the website less often.",
        "tactics": [
            "Offer early access to premium products.",
            "Use loyalty rewards that protect retention.",
            "Recommend products from their past purchases.",
        ],
    },
    "Mainstream Regulars": {
        "profile": "Steady income, spending, and purchases across several product types.",
        "tactics": [
            "Test a spend-and-save bundle.",
            "Suggest a related product at checkout.",
            "Use simple loyalty points for repeat purchases.",
        ],
    },
    "Value Families": {
        "profile": "Largest households, lower spending, and frequent website visits.",
        "tactics": [
            "Test family bundles and larger packs.",
            "Run offers around school holidays.",
            "Show the saving per item clearly.",
        ],
    },
    "Digital Browsers": {
        "profile": "Most website visits but the lowest spending and purchases.",
        "tactics": [
            "Test a short checkout reminder.",
            "Offer free delivery above a clear order value.",
            "Use a short, time-limited offer to encourage checkout.",
        ],
    },
}


st.set_page_config(
    page_title="SmartCart Customer Segmentation",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_style("whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.55rem; }
    div[data-testid="stDataFrame"] { border: 1px solid rgba(49, 51, 63, 0.12); border-radius: 8px; }
    .segment-card {
        border: 1px solid rgba(49, 51, 63, 0.14);
        border-radius: 8px;
        padding: 1.05rem 1.15rem;
        min-height: 235px;
        background: #ffffff;
        color: #171923 !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    .segment-card h4,
    .segment-card p,
    .segment-card strong,
    .segment-card li {
        color: #171923 !important;
    }
    .segment-card h4 {
        font-size: 1.05rem;
        line-height: 1.25;
        margin: 0.2rem 0 0.45rem 0;
    }
    .segment-card p,
    .segment-card li {
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .segment-card ul {
        margin: 0.35rem 0 0 1.05rem;
        padding: 0;
    }
    .segment-pill {
        display: inline-block;
        padding: 0.18rem 0.5rem;
        border-radius: 999px;
        color: white;
        font-weight: 700;
        font-size: 0.8rem;
        margin-bottom: 0.35rem;
    }
    .muted { color: rgba(23, 25, 35, 0.72) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def fig_to_bytes(fig: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def show_fig(fig: plt.Figure) -> None:
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def currency(value: float) -> str:
    return f"Rs. {value:,.0f}"


def first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))


@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def expected_columns() -> list[str]:
    return [
        "ID",
        "Year_Birth",
        "Education",
        "Marital_Status",
        "Income",
        "Kidhome",
        "Teenhome",
        "Dt_Customer",
        "Recency",
        *SPEND_COLS,
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth",
        "Complain",
        "Response",
    ]


def validate_input(df: pd.DataFrame) -> list[str]:
    missing = [col for col in expected_columns() if col not in df.columns]
    return missing


@st.cache_data(show_spinner=False)
def prepare_data(raw_df: pd.DataFrame, reference_year: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    df = raw_df.copy()
    missing_before = int(df.isna().sum().sum())
    income_missing = int(df["Income"].isna().sum())
    df["Income"] = df["Income"].fillna(df["Income"].median())

    df["Age"] = reference_year - df["Year_Birth"]
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
    df["Customer_Tenure_Days"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days
    df["Total_Spending"] = df[SPEND_COLS].sum(axis=1)
    df["Total_Purchases"] = df[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].sum(axis=1)

    df["Education"] = df["Education"].replace(
        {
            "Basic": "Undergrad",
            "2n Cycle": "Undergrad",
            "Graduation": "Graduate",
            "Master": "Postgrad",
            "PhD": "Postgrad",
        }
    )
    df["Living_With"] = df["Marital_Status"].replace(
        {
            "Married": "Partner",
            "Together": "Partner",
            "Single": "Alone",
            "Divorced": "Alone",
            "Widow": "Alone",
            "Absurd": "Alone",
            "YOLO": "Alone",
        }
    )

    to_drop = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer", *SPEND_COLS]
    model_df = df.drop(columns=to_drop)
    before_outliers = len(model_df)
    keep_mask = (model_df["Age"] < 90) & (model_df["Income"] < 600_000)
    model_df = model_df.loc[keep_mask].copy().reset_index(drop=True)
    full_df = df.loc[keep_mask].copy().reset_index(drop=True)

    stats = {
        "missing_before": missing_before,
        "income_missing": income_missing,
        "missing_after": int(model_df.isna().sum().sum()),
        "outliers_removed": int(before_outliers - len(model_df)),
    }
    return model_df, full_df, stats


@st.cache_data(show_spinner=False)
def run_models(model_df: pd.DataFrame, k: int, max_k: int, sample_size: int) -> dict[str, object]:
    x = model_df[FEATURES]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    k_values = list(range(2, max_k + 1))
    wcss = [KMeans(n_clusters=1, random_state=42, n_init=10).fit(x_scaled).inertia_]
    silhouette_values = []
    for kk in k_values:
        km_loop = KMeans(n_clusters=kk, random_state=42, n_init=10)
        labels_loop = km_loop.fit_predict(x_scaled)
        wcss.append(km_loop.inertia_)
        silhouette_values.append(silhouette_score(x_scaled, labels_loop))

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_km = km.fit_predict(x_scaled)
    sil_km = silhouette_score(x_scaled, labels_km)

    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels_agg = agg.fit_predict(x_scaled)
    sil_agg = silhouette_score(x_scaled, labels_agg)

    gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
    labels_gmm = gmm.fit_predict(x_scaled)
    sil_gmm = silhouette_score(x_scaled, labels_gmm)

    rows = []
    for name, labels, score in [
        ("K-Means", labels_km, sil_km),
        ("Agglomerative (Ward)", labels_agg, sil_agg),
        ("GMM", labels_gmm, sil_gmm),
    ]:
        row = {"Algorithm": name, "Silhouette": score}
        for cluster_id in range(k):
            row[f"Cluster {cluster_id}"] = int((labels == cluster_id).sum())
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values("Silhouette", ascending=False).reset_index(drop=True)

    sample_count = min(sample_size, len(x_scaled))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(x_scaled), sample_count, replace=False)
    dendrogram_linkage = linkage(x_scaled[sample_idx], method="ward")

    pca = PCA(n_components=2, random_state=42)
    pca_points = pca.fit_transform(x_scaled)

    return {
        "x_scaled": x_scaled,
        "wcss": wcss,
        "k_values": k_values,
        "silhouette_values": silhouette_values,
        "labels_km": labels_km,
        "labels_agg": labels_agg,
        "labels_gmm": labels_gmm,
        "sil_km": sil_km,
        "sil_agg": sil_agg,
        "sil_gmm": sil_gmm,
        "comparison": comparison,
        "dendrogram_linkage": dendrogram_linkage,
        "pca_points": pca_points,
        "pca_explained": pca.explained_variance_ratio_,
    }


def build_profiled(model_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    profiled = model_df.copy().reset_index(drop=True)
    profiled["Cluster"] = labels
    return profiled


def assign_personas(summary: pd.DataFrame, k: int) -> dict[int, str]:
    spend_rank = summary["Total_Spending"].rank(ascending=False, method="first")
    child_rank = summary["Total_Children"].rank(ascending=False, method="first")
    web_rank = summary["NumWebVisitsMonth"].rank(ascending=False, method="first")
    income_rank = summary["Income"].rank(ascending=False, method="first")
    purchase_rank = summary["Total_Purchases"].rank(ascending=False, method="first")

    personas: dict[int, str] = {}
    used_names: set[str] = set()
    for cluster_id in range(k):
        if spend_rank[cluster_id] == 1:
            name = "The Elite"
        elif spend_rank[cluster_id] >= max(3, k - 1) and child_rank[cluster_id] <= 2:
            name = "Value Families"
        elif spend_rank[cluster_id] >= max(3, k - 1) and web_rank[cluster_id] <= 2:
            name = "Digital Browsers"
        elif income_rank[cluster_id] <= 2 and purchase_rank[cluster_id] <= 2:
            name = "Premium Regulars"
        elif purchase_rank[cluster_id] <= max(2, k // 2):
            name = "Frequent Shoppers"
        elif web_rank[cluster_id] <= max(2, k // 2):
            name = "Online Researchers"
        else:
            name = "Mainstream Regulars" if "Mainstream Regulars" not in used_names else "Occasional Buyers"

        if name in used_names:
            fallback_names = [
                "Balanced Buyers",
                "Selective Shoppers",
                "Low-Spend Regulars",
                "Deal Seekers",
                "Dormant Customers",
            ]
            name = next((fallback for fallback in fallback_names if fallback not in used_names), f"Segment {cluster_id}")
        personas[cluster_id] = name
        used_names.add(name)
    return personas


def build_summary(profiled: pd.DataFrame, k: int) -> tuple[pd.DataFrame, dict[int, str]]:
    summary = profiled.groupby("Cluster")[FEATURES].mean().round(1)
    summary.insert(0, "Count", profiled["Cluster"].value_counts().sort_index())
    summary = summary.reindex(range(k))
    personas = assign_personas(summary, k)
    summary.insert(1, "Persona", pd.Series(personas))
    return summary, personas


def draw_distributions(model_df: pd.DataFrame) -> plt.Figure:
    dist_feats = ["Income", "Age", "Total_Spending", "Total_Purchases", "Recency", "NumWebVisitsMonth"]
    dist_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col, clr in zip(axes.flatten(), dist_feats, dist_colors):
        sns.histplot(model_df[col], kde=True, color=clr, edgecolor="white", ax=ax)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Feature Distributions", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def draw_correlation(model_df: pd.DataFrame) -> plt.Figure:
    corr_cols = ["Income", "Total_Spending", "Total_Purchases", "Recency", "Age", "Total_Children", "NumWebVisitsMonth"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        model_df[corr_cols].corr(),
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        annot_kws={"size": 10},
        ax=ax,
    )
    ax.set_title("Feature Correlations", fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def draw_k_selection(model_result: dict[str, object], selected_k: int) -> plt.Figure:
    k_values = model_result["k_values"]
    wcss = model_result["wcss"]
    silhouette_values = model_result["silhouette_values"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(range(1, len(wcss) + 1), wcss, "o-", color="#457B9D", lw=2, markersize=6)
    ax1.axvline(selected_k, color="#E63946", ls="--", lw=1.5, label=f"K = {selected_k}")
    ax1.set_xlabel("K")
    ax1.set_ylabel("WCSS")
    ax1.set_title("Elbow Method", fontweight="bold")
    ax1.legend()

    ax2.plot(k_values, silhouette_values, "s-", color="#2A9D8F", lw=2, markersize=6)
    ax2.axvline(selected_k, color="#E63946", ls="--", lw=1.5, label=f"K = {selected_k}")
    ax2.set_xlabel("K")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis", fontweight="bold")
    ax2.legend()
    fig.tight_layout()
    return fig


def draw_dendrogram(z_matrix: np.ndarray, k: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 4.5))
    cut_index = min(k - 1, len(z_matrix) - 1)
    color_threshold = z_matrix[-cut_index, 2] if cut_index > 0 else None
    dendrogram(
        z_matrix,
        truncate_mode="lastp",
        p=30,
        show_leaf_counts=True,
        leaf_rotation=45,
        color_threshold=color_threshold,
        ax=ax,
    )
    if color_threshold:
        ax.axhline(color_threshold, color="black", ls="--", lw=1.5, label=f"Cut -> {k} clusters")
        ax.legend()
    ax.set_title("Dendrogram (Ward Linkage)", fontweight="bold", pad=10)
    ax.set_xlabel("Customers (sample)")
    ax.set_ylabel("Merge Distance")
    fig.tight_layout()
    return fig


def draw_algorithm_boundaries(model_df: pd.DataFrame, model_result: dict[str, object], k: int) -> plt.Figure:
    algos = [
        ("K-Means", model_result["labels_km"], model_result["sil_km"]),
        ("Agglomerative", model_result["labels_agg"], model_result["sil_agg"]),
        ("GMM", model_result["labels_gmm"], model_result["sil_gmm"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (name, labels, score) in zip(axes, algos):
        for cluster_id in range(k):
            mask = labels == cluster_id
            ax.scatter(
                model_df["Income"].values[mask],
                model_df["Total_Spending"].values[mask],
                c=COLORS[cluster_id],
                alpha=0.5,
                s=15,
                label=f"C{cluster_id}",
            )
        ax.set_title(f"{name}\nSilhouette = {score:.3f}", fontweight="bold")
        ax.set_xlabel("Income")
    axes[0].set_ylabel("Total Spending")
    axes[0].legend(markerscale=2, fontsize=9)
    fig.tight_layout()
    return fig


def draw_boxplots(profiled: pd.DataFrame, k: int) -> plt.Figure:
    box_feats = ["Income", "Total_Spending", "Total_Purchases", "Recency", "Age"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
    for ax, feat in zip(axes, box_feats):
        bp = ax.boxplot(
            [profiled.loc[profiled["Cluster"] == cluster_id, feat] for cluster_id in range(k)],
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="black", linewidth=2),
            flierprops=dict(marker=".", markersize=3, alpha=0.3),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )
        for patch, clr in zip(bp["boxes"], COLORS):
            patch.set_facecolor(clr)
            patch.set_alpha(0.75)
        ax.set_xticklabels([f"C{cluster_id}" for cluster_id in range(k)])
        ax.set_title(feat, fontweight="bold")
        if ax != axes[0]:
            ax.set_ylabel("")
    fig.suptitle("Feature Distributions by Cluster", fontweight="bold", fontsize=12, y=1.03)
    fig.tight_layout()
    return fig


def draw_radar(summary: pd.DataFrame, k: int) -> plt.Figure:
    radar_feats = ["Income", "Total_Spending", "Total_Purchases", "NumWebVisitsMonth", "Total_Children", "Recency"]
    radar_labels = ["Income", "Spending", "Purchases", "Web Visits", "Children", "Recency\n(higher = recent)"]
    norm = summary[radar_feats].copy()
    for col in radar_feats:
        lo, hi = norm[col].min(), norm[col].max()
        norm[col] = (norm[col] - lo) / (hi - lo + 1e-9)
    norm["Recency"] = 1 - norm["Recency"]

    count = len(radar_feats)
    angles = [n / count * 2 * np.pi for n in range(count)] + [0]
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=9.5, fontweight="bold")
    ax.set_yticklabels([])

    for cluster_id in range(k):
        vals = norm.loc[cluster_id, radar_feats].tolist() + [norm.loc[cluster_id, radar_feats[0]]]
        ax.plot(angles, vals, color=COLORS[cluster_id], linewidth=2.5, label=f"Cluster {cluster_id}")
        ax.fill(angles, vals, color=COLORS[cluster_id], alpha=0.08)
    ax.set_title("Cluster Fingerprints", fontweight="bold", size=13, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.1), fontsize=9)
    fig.tight_layout()
    return fig


def draw_income_spending(profiled: pd.DataFrame, k: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cluster_id in range(k):
        mask = profiled["Cluster"] == cluster_id
        ax.scatter(
            profiled.loc[mask, "Income"],
            profiled.loc[mask, "Total_Spending"],
            c=COLORS[cluster_id],
            s=20,
            alpha=0.55,
            label=f"Cluster {cluster_id}",
        )
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Total Spending")
    ax.set_title("Customer Segments - Income vs Spending", fontweight="bold")
    ax.legend(title="Cluster", fontsize=9)
    fig.tight_layout()
    return fig


def draw_product_mix(full_df: pd.DataFrame, labels: np.ndarray, k: int) -> plt.Figure:
    df_full = full_df.copy()
    df_full["Cluster"] = labels
    prod_mean = df_full.groupby("Cluster")[PRODUCT_COLS].mean().reindex(range(k))
    x = np.arange(len(PRODUCT_LABELS))
    width = min(0.8 / k, 0.18)
    fig, ax = plt.subplots(figsize=(13, 5))
    for cluster_id in range(k):
        ax.bar(
            x + cluster_id * width,
            prod_mean.iloc[cluster_id],
            width,
            color=COLORS[cluster_id],
            label=f"Cluster {cluster_id}",
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xticks(x + width * (k - 1) / 2)
    ax.set_xticklabels(PRODUCT_LABELS, fontsize=11)
    ax.set_ylabel("Average Spend")
    ax.set_title("What Does Each Cluster Buy?", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


def draw_response(full_df: pd.DataFrame, labels: np.ndarray, k: int) -> tuple[plt.Figure, pd.Series]:
    df_full = full_df.copy()
    df_full["Cluster"] = labels
    response_rate = (df_full.groupby("Cluster")["Response"].mean() * 100).round(1).reindex(range(k))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(k), response_rate, color=COLORS[:k], edgecolor="black", linewidth=1, width=0.55)
    ax.bar_label(bars, fmt="%.1f%%", padding=5, fontweight="bold", fontsize=11)
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"Cluster {cluster_id}" for cluster_id in range(k)], fontsize=11)
    ax.set_ylabel("Response Rate (%)")
    ax.set_title("Campaign Response Rate by Cluster", fontweight="bold")
    ax.set_ylim(0, max(float(response_rate.max()) * 1.25, 5))
    fig.tight_layout()
    return fig, response_rate


def draw_pca(model_result: dict[str, object], labels: np.ndarray, k: int, algorithm: str) -> plt.Figure:
    points = model_result["pca_points"]
    explained = model_result["pca_explained"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cluster_id in range(k):
        mask = labels == cluster_id
        ax.scatter(points[mask, 0], points[mask, 1], c=COLORS[cluster_id], s=18, alpha=0.55, label=f"Cluster {cluster_id}")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.set_title(f"PCA View of {algorithm} Segments", fontweight="bold")
    ax.legend(title="Cluster")
    fig.tight_layout()
    return fig


with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload smartcart_customers.csv", type=["csv"])
    default_path = first_existing_path(DEFAULT_DATA_PATHS)
    if uploaded is None and default_path:
        st.caption(f"Auto-loaded: {default_path}")

    st.header("Model controls")
    selected_k = st.slider("Number of clusters", min_value=2, max_value=8, value=4)
    selected_algorithm = st.selectbox("Clustering algorithm", list(ALGORITHM_OPTIONS.keys()), index=0)
    max_k = st.slider("K range upper bound", min_value=max(5, selected_k), max_value=12, value=max(10, selected_k))
    reference_year = st.number_input("Reference year for age", min_value=2014, max_value=2035, value=2026)
    dendrogram_sample = st.slider("Dendrogram sample size", min_value=100, max_value=600, value=250, step=50)

st.title("SmartCart Customer Segmentation")

if uploaded is not None:
    raw_df = load_csv_from_bytes(uploaded.getvalue())
elif default_path is not None:
    raw_df = load_csv_from_path(str(default_path))
else:
    st.info(
        "Upload the `smartcart_customers.csv` dataset to run the dashboard. "
        "You can also place it in this app folder and refresh the page."
    )
    st.subheader("Expected dataset columns")
    st.dataframe(pd.DataFrame({"Column": expected_columns()}), use_container_width=True, hide_index=True)
    st.stop()

missing_columns = validate_input(raw_df)
if missing_columns:
    st.error("The uploaded CSV is missing required columns.")
    st.write(missing_columns)
    st.stop()

with st.spinner("Preparing data and fitting clustering models..."):
    model_df, full_df, prep_stats = prepare_data(raw_df, reference_year)
    model_result = run_models(model_df, selected_k, max_k, dendrogram_sample)
    selected_model_config = ALGORITHM_OPTIONS[selected_algorithm]
    selected_labels = model_result[selected_model_config["label_key"]]
    selected_silhouette = float(model_result[selected_model_config["score_key"]])
    profiled = build_profiled(model_df, selected_labels)
    summary, personas = build_summary(profiled, selected_k)
    profiled["Persona"] = profiled["Cluster"].map(personas)

best_row = model_result["comparison"].iloc[0]
total_spend = float(profiled["Total_Spending"].sum())
avg_response = float(full_df["Response"].mean() * 100)

metric_cols = st.columns(5)
metric_cols[0].metric("Starting Customers", f"{len(raw_df):,}")
metric_cols[1].metric("After Cleaning", f"{len(profiled):,}", delta=f"-{prep_stats['outliers_removed']} data outliers")
metric_cols[2].metric("Missing Income Filled", f"{prep_stats['income_missing']:,}")
metric_cols[3].metric("Selected Model", selected_algorithm, delta=f"silhouette {selected_silhouette:.3f}")
metric_cols[4].metric("Total Spend", currency(total_spend))

tab_overview, tab_prep, tab_eda, tab_models, tab_profiles, tab_campaigns, tab_pca = st.tabs(
    ["Overview", "Data Prep", "EDA", "Models", "Segments", "Campaigns", "PCA"]
)

with tab_overview:
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Final customer groups")
        final_table = summary[["Persona", "Count", "Income", "Total_Spending", "Total_Purchases", "Recency"]].copy()
        final_table = final_table.rename(
            columns={
                "Persona": "Customer group",
                "Income": "Average income",
                "Total_Spending": "Average spending",
                "Total_Purchases": "Average purchases",
            }
        )
        st.dataframe(final_table, use_container_width=True)
    with right:
        show_fig(draw_income_spending(profiled, selected_k))

with tab_prep:
    st.subheader("Notebook data preparation pipeline")
    cols = st.columns(4)
    cols[0].metric("Original rows", f"{len(raw_df):,}")
    cols[1].metric("Original columns", f"{raw_df.shape[1]:,}")
    cols[2].metric("Missing before", f"{prep_stats['missing_before']:,}")
    cols[3].metric("Missing after", f"{prep_stats['missing_after']:,}")

    st.dataframe(model_df.head(20), use_container_width=True)

with tab_eda:
    st.subheader("Exploratory analysis")
    viz_col_1, viz_col_2 = st.columns([1.2, 1])
    with viz_col_1:
        show_fig(draw_distributions(model_df))
    with viz_col_2:
        show_fig(draw_correlation(model_df))

with tab_models:
    st.subheader("Choosing K and comparing algorithms")
    st.dataframe(
        model_result["comparison"].style.format({"Silhouette": "{:.4f}"}),
        use_container_width=True,
    )
    show_fig(draw_k_selection(model_result, selected_k))
    with st.expander("Ward dendrogram sample", expanded=False):
        show_fig(draw_dendrogram(model_result["dendrogram_linkage"], selected_k))
    show_fig(draw_algorithm_boundaries(model_df, model_result, selected_k))

with tab_profiles:
    st.subheader("Cluster profiles")
    show_fig(draw_boxplots(profiled, selected_k))
    lower_left, lower_right = st.columns([0.9, 1.1])
    with lower_left:
        show_fig(draw_radar(summary, selected_k))
    with lower_right:
        st.dataframe(summary, use_container_width=True)

    st.download_button(
        "Download profiled customers",
        data=profiled.to_csv(index=False).encode("utf-8"),
        file_name="smartcart_profiled_customers.csv",
        mime="text/csv",
    )

with tab_campaigns:
    st.subheader("External validation and marketing plans")
    campaign_left, campaign_right = st.columns([1.1, 0.9])
    with campaign_left:
        show_fig(draw_product_mix(full_df, selected_labels, selected_k))
    with campaign_right:
        response_fig, response_rate = draw_response(full_df, selected_labels, selected_k)
        show_fig(response_fig)
        st.metric("Overall response rate", f"{avg_response:.1f}%")

    st.subheader("Persona playbook")
    card_cols = st.columns(min(selected_k, 3))
    for cluster_id in range(selected_k):
        name = personas[cluster_id]
        base_name = name.rsplit(" ", 1)[0] if name.rsplit(" ", 1)[-1].isdigit() else name
        info = STRATEGIES.get(base_name, STRATEGIES["Mainstream Regulars"])
        card = card_cols[cluster_id % len(card_cols)]
        with card:
            color = COLORS[cluster_id]
            count = int(summary.loc[cluster_id, "Count"])
            pct = count / len(profiled) * 100
            st.markdown(
                f"""
                <div class="segment-card">
                    <span class="segment-pill" style="background:{color};">Cluster {cluster_id}</span>
                    <h4>{name}</h4>
                    <p class="muted" style="margin-top:0;">{count:,} customers ({pct:.0f}%)</p>
                    <p>{info["profile"]}</p>
                    <strong>Ideas to test</strong>
                    <ul>
                        <li>{info["tactics"][0]}</li>
                        <li>{info["tactics"][1]}</li>
                        <li>{info["tactics"][2]}</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab_pca:
    st.subheader("PCA check")
    explained = model_result["pca_explained"]
    pca_cols = st.columns(3)
    pca_cols[0].metric("PC1 variance", f"{explained[0] * 100:.1f}%")
    pca_cols[1].metric("PC2 variance", f"{explained[1] * 100:.1f}%")
    pca_cols[2].metric("Combined", f"{explained.sum() * 100:.1f}%")
    show_fig(draw_pca(model_result, selected_labels, selected_k, selected_algorithm))
