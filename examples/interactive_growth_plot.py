"""I provide an interactive Tkinter dashboard to explore Japan GDP growth predictions and features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


# --- Make `src/` imports work when running with `-m` ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_master_table  # noqa: E402


RESULTS_DIR = PROJECT_ROOT / "results"


class DashboardApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Japan GDP Growth Dashboard")

        self._initializing = True
        self._redraw_job = None

        self.df_master = build_master_table().copy().sort_values("year").reset_index(drop=True)

        self.df_main = self._load_predictions("main")
        self.df_robust = self._load_predictions("robust")

        self.variant_var = tk.StringVar(value="main")
        self.view_var = tk.StringVar(value="time_series")
        self.scatter_x_var = tk.StringVar(value="year")
        self.color_by_var = tk.StringVar(value="set")

        self.year_var = tk.DoubleVar(value=float(self.df_master["year"].min()))

        self._build_ui()

        self._initializing = False
        self._sync_year_slider()
        self._redraw(full=True)

    # -------------------- Data --------------------

    def _load_predictions(self, tag: str) -> pd.DataFrame:
        path = RESULTS_DIR / f"random_forest_predictions_{tag}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}. Run: python3 main.py")

        pred = pd.read_csv(path)
        needed = {"year", "gdp_growth_actual", "gdp_growth_pred_rf", "set"}
        if not needed.issubset(set(pred.columns)):
            raise ValueError(f"{path.name} missing columns. Found: {pred.columns.tolist()}")

        pred["year"] = pd.to_numeric(pred["year"], errors="coerce").astype("Int64")
        pred = pred.dropna(subset=["year"]).copy()
        pred["year"] = pred["year"].astype(int)

        merged = pred.merge(self.df_master, on="year", how="left")
        merged["error"] = merged["gdp_growth_pred_rf"] - merged["gdp_growth_actual"]
        return merged.sort_values("year").reset_index(drop=True)

    def _active_df(self) -> pd.DataFrame:
        return self.df_main if self.variant_var.get() == "main" else self.df_robust

    # -------------------- UI --------------------

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky="nsw")

        right = ttk.Frame(self.root, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=2)
        right.columnconfigure(0, weight=1)

        ttk.Label(left, text="Controls", font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))

        ttk.Label(left, text="Variant").grid(row=1, column=0, sticky="w")
        variant_box = ttk.Combobox(left, textvariable=self.variant_var, values=["main", "robust"], state="readonly", width=16)
        variant_box.grid(row=2, column=0, sticky="w", pady=(0, 10))
        variant_box.bind("<<ComboboxSelected>>", lambda _e: self._on_variant_change())

        ttk.Label(left, text="View").grid(row=3, column=0, sticky="w")
        view_box = ttk.Combobox(left, textvariable=self.view_var, values=["time_series", "scatter"], state="readonly", width=16)
        view_box.grid(row=4, column=0, sticky="w", pady=(0, 10))
        view_box.bind("<<ComboboxSelected>>", lambda _e: self._schedule_redraw(full=True))

        ttk.Label(left, text="Scatter X").grid(row=5, column=0, sticky="w")
        self.scatter_x_box = ttk.Combobox(
            left,
            textvariable=self.scatter_x_var,
            values=[
                "year",
                "n_events",
                "log_total_damage",
                "damage_share_gdp",
                "inflation_cpi",
                "unemployment_rate",
                "exports_pct_gdp",
                "investment_pct_gdp",
                "fx_jpy_per_usd",
                "oil_price_usd",
                "error",
            ],
            state="readonly",
            width=22,
        )
        self.scatter_x_box.grid(row=6, column=0, sticky="w", pady=(0, 10))
        self.scatter_x_box.bind("<<ComboboxSelected>>", lambda _e: self._schedule_redraw(full=True))

        ttk.Label(left, text="Color by").grid(row=7, column=0, sticky="w")
        self.color_by_box = ttk.Combobox(
            left,
            textvariable=self.color_by_var,
            values=[
                "none",
                "set",
                "has_disaster",
                "n_events",
                "log_total_damage",
                "inflation_cpi",
                "unemployment_rate",
                "fx_jpy_per_usd",
                "oil_price_usd",
                "error",
            ],
            state="readonly",
            width=22,
        )
        self.color_by_box.grid(row=8, column=0, sticky="w", pady=(0, 10))
        self.color_by_box.bind("<<ComboboxSelected>>", lambda _e: self._schedule_redraw(full=True))

        # IMPORTANT: create year_label BEFORE setting the scale value (callback safety)
        self.year_label = ttk.Label(left, text="Year: -", font=("TkDefaultFont", 11, "bold"))
        self.year_label.grid(row=9, column=0, sticky="w", pady=(10, 4))

        self.year_scale = ttk.Scale(
            left,
            from_=float(self.df_master["year"].min()),
            to=float(self.df_master["year"].max()),
            orient="horizontal",
            variable=self.year_var,
            command=self._on_year_slide,
        )
        self.year_scale.grid(row=10, column=0, sticky="we")

        ttk.Separator(left, orient="horizontal").grid(row=11, column=0, sticky="we", pady=12)

        ttk.Label(left, text="Run from repo root:", foreground="gray").grid(row=12, column=0, sticky="w")
        ttk.Label(left, text="python3 -m examples.interactive_growth_plot", foreground="gray").grid(row=13, column=0, sticky="w")

        # Plot
        self.fig = Figure(figsize=(8.2, 5.2), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Fixed colorbar axis so plot does not shrink on updates
        self.cbar_ax = self.fig.add_axes([0.92, 0.12, 0.02, 0.76])
        self.cbar_ax.set_visible(False)
        self.cbar = None

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Table
        table_frame = ttk.Frame(right)
        table_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.table = ttk.Treeview(table_frame, columns=("key", "value"), show="headings", height=9)
        self.table.heading("key", text="Field")
        self.table.heading("value", text="Value")
        self.table.column("key", width=180, anchor="w")
        self.table.column("value", width=260, anchor="w")
        self.table.grid(row=0, column=0, sticky="nsew")

        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        scroll.grid(row=0, column=1, sticky="ns")

    # -------------------- Events --------------------

    def _on_variant_change(self) -> None:
        self._sync_year_slider()
        self._schedule_redraw(full=True)

    def _on_year_slide(self, _val: str) -> None:
        # Slider fires during initialization; ensure safe
        if not hasattr(self, "year_label"):
            return
        year = int(round(float(self.year_var.get())))
        self.year_label.config(text=f"Year: {year}")
        self._update_table(year)
        self._update_highlight(year)

    def _schedule_redraw(self, full: bool = True) -> None:
        if self._initializing:
            return
        if self._redraw_job is not None:
            try:
                self.root.after_cancel(self._redraw_job)
            except Exception:
                pass
        self._redraw_job = self.root.after(60, lambda: self._redraw(full=full))

    # -------------------- Plot helpers --------------------

    def _clear_colorbar(self) -> None:
        self.cbar_ax.cla()
        self.cbar_ax.set_visible(False)
        self.cbar = None

    def _set_colorbar(self, mappable, label: str) -> None:
        self.cbar_ax.set_visible(True)
        if self.cbar is None:
            self.cbar = self.fig.colorbar(mappable, cax=self.cbar_ax)
        else:
            self.cbar.update_normal(mappable)
        self.cbar.set_label(label)

    def _sync_year_slider(self) -> None:
        df = self._active_df()
        yr_min = int(df["year"].min())
        yr_max = int(df["year"].max())

        self.year_scale.configure(from_=float(yr_min), to=float(yr_max))

        cur = int(round(float(self.year_var.get())))
        cur = min(max(cur, yr_min), yr_max)

        # IMPORTANT: label already exists, safe to set
        self.year_var.set(float(cur))
        self.year_label.config(text=f"Year: {cur}")

    def _redraw(self, full: bool = True) -> None:
        self._redraw_job = None
        df = self._active_df().copy()

        self.ax.clear()
        self._clear_colorbar()

        view = self.view_var.get()
        color_by = self.color_by_var.get()

        if view == "time_series":
            self._plot_time_series(df, color_by=color_by)
        else:
            xcol = self.scatter_x_var.get()
            self._plot_scatter(df, xcol=xcol, color_by=color_by)

        self.ax.set_title(f"Japan GDP growth: actual vs RF prediction ({self.variant_var.get()})")
        self.ax.grid(True, alpha=0.2)

        # Keep space for fixed colorbar axis
        self.fig.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])
        self.canvas.draw()

        year = int(round(float(self.year_var.get())))
        self._update_table(year)
        self._update_highlight(year)

    def _plot_time_series(self, df: pd.DataFrame, color_by: str) -> None:
        x = df["year"].to_numpy()
        y_actual = df["gdp_growth_actual"].to_numpy()
        y_pred = df["gdp_growth_pred_rf"].to_numpy()

        self.ax.plot(x, y_actual, marker="o", linewidth=1, label="Actual", alpha=0.85)
        self.ax.plot(x, y_pred, marker="o", linewidth=1, label="Predicted (RF)", alpha=0.85)

        if color_by == "none":
            self.ax.legend(loc="best")
            return

        if color_by == "set":
            s = df["set"].astype(str)
            mask_test = (s == "test").to_numpy()
            self.ax.scatter(x[~mask_test], y_actual[~mask_test], label="Train (actual)", alpha=0.5)
            self.ax.scatter(x[mask_test], y_actual[mask_test], label="Test (actual)", alpha=0.9)
            self.ax.legend(loc="best")
            return

        if color_by == "has_disaster" and "has_disaster" in df.columns:
            d = df["has_disaster"].fillna(0).astype(int).to_numpy()
            self.ax.scatter(x[d == 0], y_actual[d == 0], label="No disaster", alpha=0.6)
            self.ax.scatter(x[d == 1], y_actual[d == 1], label="Disaster year", alpha=0.9)
            self.ax.legend(loc="best")
            return

        if color_by in df.columns and pd.api.types.is_numeric_dtype(df[color_by]):
            c = pd.to_numeric(df[color_by], errors="coerce").to_numpy()
            sc = self.ax.scatter(x, y_actual, c=c, s=45, alpha=0.9)
            self._set_colorbar(sc, color_by)
            self.ax.legend(loc="best")
            return

        self.ax.legend(loc="best")

    def _plot_scatter(self, df: pd.DataFrame, xcol: str, color_by: str) -> None:
        if xcol not in df.columns:
            xcol = "year"

        x = pd.to_numeric(df[xcol], errors="coerce")
        y = pd.to_numeric(df["gdp_growth_actual"], errors="coerce")
        yhat = pd.to_numeric(df["gdp_growth_pred_rf"], errors="coerce")

        mask = x.notna() & y.notna() & yhat.notna()
        df2 = df.loc[mask].copy()
        x = x.loc[mask].to_numpy()
        y = y.loc[mask].to_numpy()
        yhat = yhat.loc[mask].to_numpy()

        if color_by == "none":
            self.ax.scatter(x, y, label="Actual", alpha=0.85)
            self.ax.scatter(x, yhat, label="Predicted (RF)", alpha=0.55)
            self.ax.set_xlabel(xcol)
            self.ax.set_ylabel("GDP growth (%)")
            self.ax.legend(loc="best")
            return

        if color_by == "set":
            s = df2["set"].astype(str)
            train = (s == "train").to_numpy()
            test = (s == "test").to_numpy()
            self.ax.scatter(x[train], y[train], label="Train (actual)", alpha=0.6)
            self.ax.scatter(x[test], y[test], label="Test (actual)", alpha=0.9)
            self.ax.scatter(x, yhat, label="Predicted (RF)", alpha=0.35)
            self.ax.set_xlabel(xcol)
            self.ax.set_ylabel("GDP growth (%)")
            self.ax.legend(loc="best")
            return

        if color_by == "has_disaster" and "has_disaster" in df2.columns:
            d = df2["has_disaster"].fillna(0).astype(int).to_numpy()
            self.ax.scatter(x[d == 0], y[d == 0], label="No disaster (actual)", alpha=0.7)
            self.ax.scatter(x[d == 1], y[d == 1], label="Disaster year (actual)", alpha=0.9)
            self.ax.scatter(x, yhat, label="Predicted (RF)", alpha=0.35)
            self.ax.set_xlabel(xcol)
            self.ax.set_ylabel("GDP growth (%)")
            self.ax.legend(loc="best")
            return

        if color_by in df2.columns and pd.api.types.is_numeric_dtype(df2[color_by]):
            c = pd.to_numeric(df2[color_by], errors="coerce").to_numpy()
            sc = self.ax.scatter(x, y, c=c, s=55, alpha=0.9, label="Actual")
            self.ax.scatter(x, yhat, alpha=0.30, label="Predicted (RF)")
            self._set_colorbar(sc, color_by)
            self.ax.set_xlabel(xcol)
            self.ax.set_ylabel("GDP growth (%)")
            self.ax.legend(loc="best")
            return

        self.ax.scatter(x, y, label="Actual", alpha=0.85)
        self.ax.scatter(x, yhat, label="Predicted (RF)", alpha=0.55)
        self.ax.set_xlabel(xcol)
        self.ax.set_ylabel("GDP growth (%)")
        self.ax.legend(loc="best")

    def _update_highlight(self, year: int) -> None:
        # Remove previous highlight markers
        for artist in list(self.ax.collections):
            if getattr(artist, "_is_highlight", False):
                artist.remove()

        df = self._active_df()
        row = df[df["year"] == year]
        if row.empty:
            self.canvas.draw()
            return
        r = row.iloc[0]

        if self.view_var.get() == "time_series":
            x = float(r["year"])
            y = float(r["gdp_growth_actual"])
            hl = self.ax.scatter([x], [y], s=160, alpha=0.95)
            setattr(hl, "_is_highlight", True)
            self.canvas.draw()
            return

        xcol = self.scatter_x_var.get()
        if xcol not in df.columns:
            xcol = "year"
        xv = pd.to_numeric(pd.Series([r.get(xcol)]), errors="coerce").iloc[0]
        if pd.notna(xv):
            hl = self.ax.scatter([float(xv)], [float(r["gdp_growth_actual"])], s=160, alpha=0.95)
            setattr(hl, "_is_highlight", True)
        self.canvas.draw()

    def _update_table(self, year: int) -> None:
        df = self._active_df()
        row = df[df["year"] == year]
        if row.empty:
            return
        r = row.iloc[0].to_dict()

        for item in self.table.get_children():
            self.table.delete(item)

        fields = [
            ("year", "year"),
            ("set", "set"),
            ("actual", "gdp_growth_actual"),
            ("pred_rf", "gdp_growth_pred_rf"),
            ("error", "error"),
            ("n_events", "n_events"),
            ("log_total_damage", "log_total_damage"),
            ("damage_share_gdp", "damage_share_gdp"),
            ("inflation_cpi", "inflation_cpi"),
            ("unemployment_rate", "unemployment_rate"),
            ("exports_pct_gdp", "exports_pct_gdp"),
            ("investment_pct_gdp", "investment_pct_gdp"),
            ("fx_jpy_per_usd", "fx_jpy_per_usd"),
            ("oil_price_usd", "oil_price_usd"),
        ]

        for label, key in fields:
            if key not in r:
                continue
            val = r.get(key)
            if isinstance(val, (float, np.floating)):
                self.table.insert("", "end", values=(label, f"{float(val):.3f}"))
            else:
                self.table.insert("", "end", values=(label, str(val)))


def main() -> None:
    root = tk.Tk()

    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")

    try:
        _ = DashboardApp(root)
    except Exception as e:
        messagebox.showerror("Dashboard error", str(e))
        raise

    root.mainloop()


if __name__ == "__main__":
    main()
