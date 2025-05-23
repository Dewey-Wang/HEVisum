import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from matplotlib.lines import Line2D
from matplotlib import cm

def plot_all_rank_umaps(
    df,
    rank_cols=None,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
    cmap="viridis",
    point_size=5,
    alpha=0.8,
    ncols=7,
    uniform_cmap=True  # 是否所有子图统一色标
):
    if rank_cols is None:
        rank_cols = [f"rank_C{i}" for i in range(1,36)]
    # 1) UMAP 降维
    X = df[rank_cols].values.astype(float)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    emb = reducer.fit_transform(X)
    df = df.copy()
    df["UMAP1"], df["UMAP2"] = emb[:,0], emb[:,1]

    # ————————————
    # A) 按 slide 上色的全图
    unique_slides = df["slide_name"].unique()
    # map slide_name → category code 0..n-1
    codes = pd.Categorical(df["slide_name"], categories=unique_slides).codes
    cmap_cat = cm.get_cmap("tab20", len(unique_slides))

    plt.figure(figsize=(6,5))
    sc = plt.scatter(
        df["UMAP1"], df["UMAP2"],
        c=codes, cmap=cmap_cat,
        s=point_size, alpha=alpha
    )
    # build legend
    legend_elems = [
        Line2D([0],[0], marker='o', color=cmap_cat(i), label=sl,
               linestyle="", markersize=6)
        for i, sl in enumerate(unique_slides)
    ]
    plt.legend(handles=legend_elems, title="Slide", bbox_to_anchor=(1.05,1), loc="upper left")
    plt.title("UMAP colored by slide_name")
    plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.show()
    # ————————————

    # 2) 量化 Spearman ρ 并取绝对值
    spearman_results = []
    for col in rank_cols:
        rho, _ = spearmanr(df["UMAP1"], df[col].values)
        spearman_results.append(abs(rho))
    avg_rho = float(np.mean(spearman_results))
    print(f"Average abs(Spearman rho) across all features: {avg_rho:.3f}")

    # 3) 准备统一色标
    if uniform_cmap:
        all_vals = df[rank_cols].values
        vmin, vmax = all_vals.min(), all_vals.max()
    else:
        vmin = vmax = None

    # 4) 各个 cell-type 子图
    n_plots = len(rank_cols)
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols*3, nrows*2.5),
        sharex=True, sharey=True
    )
    axes = axes.flatten()

    for ax, (col, rho) in zip(axes, zip(rank_cols, spearman_results)):
        sc = ax.scatter(
            df["UMAP1"], df["UMAP2"],
            c=df[col].values,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            s=point_size, alpha=alpha
        )
        ax.set_title(f"{col} (ρ={rho:.2f})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[n_plots:]:
        ax.remove()

    plt.tight_layout()
    plt.show()

    # 5) Spearman 条形图
    rho_df = pd.DataFrame({'cell_type': rank_cols, 'spearman_rho': spearman_results})
    rho_df = rho_df.set_index('cell_type')
    plt.figure(figsize=(ncols*0.8, nrows*0.6))
    rho_df['spearman_rho'].plot(kind='bar')
    plt.ylabel('Spearman ρ with UMAP1')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.title('Continuity of cell-type ranking along UMAP1')
    plt.tight_layout()
    plt.show()