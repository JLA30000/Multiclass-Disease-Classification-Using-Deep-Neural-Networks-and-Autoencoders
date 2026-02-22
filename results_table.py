import pandas as pd
import matplotlib.pyplot as plt

# --- data (mean, std) ---
results = {
    100: {
        "Baseline": {
            "Accuracy": (0.8205, 0.0073), "Macro P": (0.8430, 0.0055), "Macro R": (0.8205, 0.0073),
            "Macro F1": (0.8202, 0.0075), "Top-3": (0.9218, 0.0043), "Top-5": (0.9384, 0.0039)
        },
        "Autoencoder": {
            "Accuracy": (0.8420, 0.0042), "Macro P": (0.8603, 0.0036), "Macro R": (0.8420, 0.0042),
            "Macro F1": (0.8401, 0.0042), "Top-3": (0.9480, 0.0020), "Top-5": (0.9728, 0.0021)
        },
    },
    200: {
        "Baseline": {
            "Accuracy": (0.8509, 0.0039), "Macro P": (0.8651, 0.0027), "Macro R": (0.8509, 0.0039),
            "Macro F1": (0.8507, 0.0041), "Top-3": (0.9419, 0.0028), "Top-5": (0.9642, 0.0022)
        },
        "Autoencoder": {
            "Accuracy": (0.8646, 0.0030), "Macro P": (0.8789, 0.0024), "Macro R": (0.8646, 0.0030),
            "Macro F1": (0.8636, 0.0028), "Top-3": (0.9618, 0.0018), "Top-5": (0.9823, 0.0007)
        },
    },
    300: {
        "Baseline": {
            "Accuracy": (0.8583, 0.0027), "Macro P": (0.8709, 0.0026), "Macro R": (0.8583, 0.0027),
            "Macro F1": (0.8585, 0.0028), "Top-3": (0.9484, 0.0019), "Top-5": (0.9697, 0.0018)
        },
        "Autoencoder": {
            "Accuracy": (0.8661, 0.0028), "Macro P": (0.8793, 0.0035), "Macro R": (0.8661, 0.0028),
            "Macro F1": (0.8658, 0.0026), "Top-3": (0.9630, 0.0016), "Top-5": (0.9837, 0.0004)
        },
    },
}

metrics = ["Accuracy", "Macro P", "Macro R", "Macro F1", "Top-3", "Top-5"]


def fmt(mean, std, d=4):
    return f"{mean:.{d}f} ± {std:.{d}f}"


def render_table_as_jpg(df, outfile, title=None, fontsize=12):
    fig_h = 0.55 * (len(df) + 1)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=14, pad=-5)  # <-- pulls title closer

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.15, 1.4)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")

    plt.tight_layout(pad=0.2)   # <-- removes extra white space
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()


# --- generate 3 separate tables (one per threshold) ---
for thr in sorted(results.keys()):
    rows = []
    for m in metrics:
        b_mean, b_std = results[thr]["Baseline"][m]
        a_mean, a_std = results[thr]["Autoencoder"][m]
        rows.append([m, fmt(b_mean, b_std), fmt(a_mean, a_std)])

    df_thr = pd.DataFrame(rows, columns=[
                          "Evaluation Metric", "Baseline Classifier", "Autoencoder Classifier"])

    # print table in console
    print(f"\n=== Threshold {thr} ===")
    print(df_thr)

    # save as separate image
    render_table_as_jpg(
        df_thr,
        outfile=f"threshold_{thr}_comparison.jpg",
        title=f"Threshold {thr}: Baseline vs Autoencoder"
    )
