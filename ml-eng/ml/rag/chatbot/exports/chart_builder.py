"""Build static chart images from tabular data."""
from __future__ import annotations

import io
from typing import Any, Literal

import pandas as pd


def _get_plt():
    """Lazy matplotlib import. Keeps the module importable in minimal images
    (e.g. Free/Farmers routes that never build charts) and defers the heavy
    matplotlib import until an export actually runs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    return plt

ChartType = Literal["line", "bar", "stacked_bar", "area", "scatter", "heatmap"]

_AGRI_COLORS = ["#2E7D32", "#1565C0", "#F9A825", "#6A1B9A", "#C62828", "#00838F"]


def _pick_axes(df: pd.DataFrame) -> tuple[str, str | None]:
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError("Need at least two columns for a chart")
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric = [c for c in cols if c not in numeric]
    x_col = non_numeric[0] if non_numeric else cols[0]
    y_col = numeric[0] if numeric else cols[1]
    return x_col, y_col


def build_chart(
    rows: list[dict[str, Any]],
    *,
    chart_type: ChartType = "line",
    title: str = "OpenTrace data",
    x_label: str | None = None,
    y_label: str | None = None,
    filename: str = "chart.png",
) -> tuple[bytes, str]:
    if not rows:
        raise ValueError("No tabular data available for chart export")
    df = pd.DataFrame(rows)
    x_col, y_col = _pick_axes(df)
    if y_col is None:
        raise ValueError("No numeric column found for chart")

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    colors = _AGRI_COLORS

    if chart_type == "bar":
        ax.bar(df[x_col].astype(str), df[y_col], color=colors[0])
    elif chart_type == "stacked_bar":
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) < 2:
            ax.bar(df[x_col].astype(str), df[y_col], color=colors[0])
        else:
            bottom = None
            for i, col in enumerate(numeric_cols):
                ax.bar(
                    df[x_col].astype(str),
                    df[col],
                    bottom=bottom,
                    label=col,
                    color=colors[i % len(colors)],
                )
                bottom = df[col] if bottom is None else bottom + df[col]
            ax.legend(fontsize=8)
    elif chart_type == "area":
        ax.fill_between(range(len(df)), df[y_col], alpha=0.4, color=colors[0])
        ax.plot(range(len(df)), df[y_col], color=colors[0])
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df[x_col].astype(str), rotation=45, ha="right", fontsize=8)
    elif chart_type == "scatter":
        ax.scatter(range(len(df)), df[y_col], color=colors[0], s=40)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df[x_col].astype(str), rotation=45, ha="right", fontsize=8)
    elif chart_type == "heatmap":
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            raise ValueError("No numeric columns for heatmap")
        im = ax.imshow(numeric_df.T, aspect="auto", cmap="YlGn")
        ax.set_yticks(range(len(numeric_df.columns)))
        ax.set_yticklabels(numeric_df.columns, fontsize=8)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df[x_col].astype(str), rotation=45, ha="right", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.plot(df[x_col].astype(str), df[y_col], marker="o", color=colors[0], linewidth=2)
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(x_label or x_col, fontsize=9)
    ax.set_ylabel(y_label or y_col, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    name = filename if filename.endswith(".png") else f"{filename}.png"
    return buf.getvalue(), name


__all__ = ["ChartType", "build_chart"]
