"""I show an interactive plot of Japan's GDP growth where I can move a slider over the years."""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from src.features import build_master_table


def main() -> None:
    df = build_master_table().sort_values("year").reset_index(drop=True)

    years = df["year"].values
    growth = df["gdp_growth"].values
    has_disaster = df["has_disaster"].astype(bool).values

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax.plot(years, growth, "-o", alpha=0.3)
    ax.scatter(years[~has_disaster], growth[~has_disaster], label="No disaster")
    ax.scatter(years[has_disaster], growth[has_disaster], label="At least one disaster")

    highlight = ax.scatter([years[0]], [growth[0]], s=120)

    ax.set_xlabel("Year")
    ax.set_ylabel("GDP growth (%)")
    ax.set_title("Japan GDP growth and natural disasters")
    ax.legend()

    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
    slider = Slider(
        ax_slider,
        "Year",
        float(years.min()),
        float(years.max()),
        valinit=float(years.min()),
        valstep=1.0,
    )

    def update(val: float) -> None:
        year = int(round(val))
        if year in years:
            idx = list(years).index(year)
            highlight.set_offsets([[years[idx], growth[idx]]])
            color = "tab:orange" if has_disaster[idx] else "tab:blue"
            highlight.set_color(color)
            fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()
