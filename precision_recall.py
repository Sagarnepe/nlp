import numpy as np

def classification_report_from_confusion(cm, labels):
    """
    cm[i, j] = count where true class = labels[i], predicted class = labels[j]
    """
    cm = np.array(cm, dtype=float)
    n = cm.shape[0]
    assert cm.shape == (n, n) == (len(labels), len(labels))

    # Per-class TP, FP, FN
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    # Safe division
    def safe_div(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=(b != 0))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    # Macro averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()

    # Micro averages (sum over classes first)
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()

    micro_precision = (micro_tp / (micro_tp + micro_fp)) if (micro_tp + micro_fp) else 0.0
    micro_recall = (micro_tp / (micro_tp + micro_fn)) if (micro_tp + micro_fn) else 0.0

    # Print nicely
    print("Confusion Matrix (rows=true, cols=pred):")
    header = " " * 10 + " ".join(f"{lab:>8}" for lab in labels)
    print(header)
    for i, lab in enumerate(labels):
        row = " ".join(f"{int(cm[i,j]):>8}" for j in range(n))
        print(f"{lab:>10} {row}")

    print("\nPer-class metrics:")
    for i, lab in enumerate(labels):
        print(f"- {lab:6s}  precision={precision[i]:.4f}  recall={recall[i]:.4f}")

    print("\nAveraged metrics:")
    print(f"- Macro precision = {macro_precision:.4f}")
    print(f"- Macro recall    = {macro_recall:.4f}")
    print(f"- Micro precision = {micro_precision:.4f}")
    print(f"- Micro recall    = {micro_recall:.4f}")


if __name__ == "__main__":
    labels = ["Cat", "Dog", "Rabbit"]

   #Confusion Matrix
    cm = [
        [5, 10, 5],
        [15, 20, 10],
        [0, 15, 10],
    ]

    classification_report_from_confusion(cm, labels)
