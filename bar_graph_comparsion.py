import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data (mean, std)
# -----------------------------
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
models = ["Baseline", "Autoencoder"]


def plot_threshold_bars(thr: int, stat: str, save: bool = True):
    """
    stat: "mean" or "std"
    Produces ONE bar chart comparing Baseline vs Autoencoder for all metrics.
    """
    assert stat in {"mean", "std"}
    x = np.arange(len(metrics))
    width = 0.38

    vals_baseline = []
    vals_auto = []

    for m in metrics:
        mean_b, std_b = results[thr]["Baseline"][m]
        mean_a, std_a = results[thr]["Autoencoder"][m]
        vals_baseline.append(mean_b if stat == "mean" else std_b)
        vals_auto.append(mean_a if stat == "mean" else std_a)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, vals_baseline, width, label="Baseline")
    ax.bar(x + width/2, vals_auto, width, label="Autoencoder")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(
        f"Threshold {thr} — {'Mean' if stat == 'mean' else 'Std Dev'} by Metric")
    ax.legend()

    plt.tight_layout()

    if save:
        fname = f"threshold_{thr}_{stat}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")

    plt.show()


# -----------------------------
# Generate 6 bar graphs:
# (mean + std) for each threshold
# -----------------------------
for thr in sorted(results.keys()):
    plot_threshold_bars(thr, "mean", save=True)  # 3 mean plots
    plot_threshold_bars(thr, "std",  save=True)  # 3 std plots
