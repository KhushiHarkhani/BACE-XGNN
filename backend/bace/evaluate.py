
"""Merge base-model + GNN out-of-fold predictions, score every setup,
and write artifacts/metrics.json and artifacts/confusion_matrix.png.

Run from backend/:
    python -m bace.evaluate

Requires:
    data/base_preds.npz
    data/gnn_oof.npz
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

from . import data
from .ensemble import meta_fit_predict


ART = data.DATA.parents[1] / "artifacts"

GROUPS = [
    (
        "Random split, 5-fold cross-validation",
        "random",
        "Each molecule is tested once, by a model that did not train on it.",
    ),
    (
        "Scaffold split (randomised), 3 seeds",
        "scaffold",
        "Scaffold families held out. Order shuffled per seed.",
    ),
    (
        "Scaffold split (strict, deterministic)",
        "strict",
        "Rarest scaffolds go to test. One 152-molecule split.",
    ),
]


def pick_threshold(ytr, p_oof):
    """Choose a classification threshold using training OOF predictions."""
    ths = np.linspace(0.3, 0.7, 41)
    scores = [
        accuracy_score(ytr, p_oof > threshold)
        for threshold in ths
    ]
    return float(ths[np.argmax(scores)])


def row(name, ys, ps, ts, inner_accs):
    """Calculate metrics across the evaluation splits."""
    y_true = np.concatenate(ys)
    probabilities = np.concatenate(ps)

    predictions = np.concatenate([
        probs > threshold
        for probs, threshold in zip(ps, ts)
    ]).astype(int)

    accs = [
        accuracy_score(y_part, probs > threshold)
        for y_part, probs, threshold in zip(ys, ps, ts)
    ]

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return dict(
        model=name,
        accuracy=float(np.mean(accs)),
        acc_sd=float(np.std(accs)),
        auc=float(roc_auc_score(y_true, probabilities)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, predictions)),
        sens=float(tp / (tp + fn)) if (tp + fn) else 0.0,
        spec=float(tn / (tn + fp)) if (tn + fp) else 0.0,
        inner_acc=float(np.mean(inner_accs)),
    )


def save_confusion_matrix(y_true, y_pred, output_path):
    """Plot and save a confusion matrix for out-of-fold predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-inhibitor (0)", "Inhibitor (1)"],
    )

    display.plot(
        ax=ax,
        cmap="Purples",
        values_format="d",
        colorbar=False,
    )

    ax.set_title(
        "BACE-XGNN Final Stacking Model\n"
        "Random 5-Fold Out-of-Fold Evaluation"
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nConfusion matrix saved to: {output_path}")
    print("\nConfusion matrix values:")
    print(cm)
    print("Rows = actual [Non-inhibitor, Inhibitor]")
    print("Columns = predicted [Non-inhibitor, Inhibitor]")


def main():
    df = data.load()
    y = df["y"].values

    base_path = data.DATA.parent / "base_preds.npz"
    gnn_path = data.DATA.parent / "gnn_oof.npz"

    B = np.load(base_path)
    G = np.load(gnn_path)

    names = list(B["random0__names"])
    best_single = names.index("cat_dense")

    blocks = []
    pic = df["pIC50"].values

    # -1 / NaN mark molecules without a random-split OOF prediction.
    pred_all = np.full(len(df), -1, dtype=int)
    prob_all = np.full(len(df), np.nan, dtype=float)
    
    for title, prefix, note in GROUPS:
        keys = sorted({
            key.split("__")[0]
            for key in B.files
            if key.startswith(prefix)
        })

        acc = {
            model: ([], [], [], [])
            for model in (
                "D-MPNN (GNN) alone",
                "Best single base model (CatBoost)",
                "Stack of 6 base models",
                "Stack + D-MPNN (final)",
            )
        }

        for key in keys:
            tr = B[f"{key}__tr"]
            te = B[f"{key}__te"]

            assert (tr == G[f"{key}__tr"]).all()
            assert (te == G[f"{key}__te"]).all()

            oof = B[f"{key}__oof"]
            tm = B[f"{key}__te_m"]

            go = G[f"{key}__oof"]
            gt = G[f"{key}__test"]

            candidates = {
                "D-MPNN (GNN) alone": (go, gt),
                "Best single base model (CatBoost)": (
                    oof[:, best_single],
                    tm[:, best_single],
                ),
            }

            # Stack of six classical base models.
            p, t = meta_fit_predict(oof, y[tr], tm)

            m6 = (
                LogisticRegression(max_iter=1000)
                .fit(oof, y[tr])
                .predict_proba(oof)[:, 1]
            )

            candidates["Stack of 6 base models"] = (m6, p)

            # Final stack: six classical models + D-MPNN.
            full_o = np.column_stack([oof, go])
            full_t = np.column_stack([tm, gt])

            pf, tf = meta_fit_predict(full_o, y[tr], full_t)

            mf = (
                LogisticRegression(max_iter=1000)
                .fit(full_o, y[tr])
                .predict_proba(full_o)[:, 1]
            )

            candidates["Stack + D-MPNN (final)"] = (mf, pf)

            for model_name, (p_oof, p_test) in candidates.items():
                threshold = pick_threshold(y[tr], p_oof)

                acc[model_name][0].append(y[te])
                acc[model_name][1].append(p_test)
                acc[model_name][2].append(threshold)
                acc[model_name][3].append(
                    accuracy_score(y[tr], p_oof > threshold)
                )

                # Collect held-out predictions from the random splits
                # for the final stacking model only.
                if (
                    prefix == "random"
                    and model_name == "Stack + D-MPNN (final)"
                ):
                    pred_all[te] = (
                        p_test > threshold
                    ).astype(int)
                    # Keep probabilities for the ROC curve (threshold-free).
                    prob_all[te] = p_test

        blocks.append({
            "title": title,
            "note": note,
            "rows": [
                row(model_name, *values)
                for model_name, values in acc.items()
            ],
        })

        print(f"\n{title}")

        for result in blocks[-1]["rows"]:
            print(
                f"  {result['model']:36s} "
                f"acc={result['accuracy']:.3f}"
                f"+-{result['acc_sd']:.3f} "
                f"auc={result['auc']:.3f} "
                f"f1={result['f1']:.3f} "
                f"mcc={result['mcc']:.3f} "
                f"sens={result['sens']:.3f} "
                f"spec={result['spec']:.3f} "
                f"| inner-CV acc={result['inner_acc']:.3f}"
            )

    # Ensure the output directory exists before saving plots.
    ART.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix for the final stack's random 5-fold OOF results.
    valid = pred_all >= 0

    if valid.any():
        save_confusion_matrix(
            y[valid],
            pred_all[valid],
            ART / "confusion_matrix.png",
        )
    else:
        print(
            "WARNING: No random-split OOF predictions were found; "
            "confusion matrix was not generated."
        )

    # Save ROC curve for the final stack using random 5-fold OOF probabilities.
    roc_valid = np.isfinite(prob_all)

    if roc_valid.any() and len(np.unique(y[roc_valid])) == 2:
        fpr, tpr, _ = roc_curve(y[roc_valid], prob_all[roc_valid])
        auc_value = roc_auc_score(y[roc_valid], prob_all[roc_valid])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            fpr,
            tpr,
            color="#D63384",
            linewidth=2.5,
            label=f"BACE-XGNN Final Stack (AUC = {auc_value:.3f})",
        )
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            linewidth=1.5,
            label="Random classifier (AUC = 0.500)",
        )
        ax.set_title(
            "BACE-XGNN Final Stacking Model\\n"
            "Random 5-Fold Out-of-Fold ROC Curve"
        )
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right")
        fig.tight_layout()

        roc_path = ART / "roc_curve.png"
        fig.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"ROC curve saved to: {roc_path}")
        print(f"ROC-AUC from random 5-fold OOF probabilities: {auc_value:.4f}")
    else:
        print(
            "WARNING: ROC curve was not generated because valid OOF "
            "probabilities for both classes were not found."
        )

    # Accuracy by distance from the pIC50 classification cutoff.
    d = np.abs(pic - 7.0)
    bins = []

    for lo, hi, label in (
        (0, 0.25, "within 0.25"),
        (0.25, 0.5, "0.25 to 0.5"),
        (0.5, 1.0, "0.5 to 1"),
        (1.0, 99, "over 1"),
    ):
        mask = (d >= lo) & (d < hi) & valid

        bins.append({
            "label": label,
            "n": int(mask.sum()),
            "share": float(mask.sum() / valid.sum()) if valid.any() else 0.0,
            "accuracy": float(
                accuracy_score(y[mask], pred_all[mask])
            ) if mask.any() else None,
        })

    near = (d < 0.5) & valid
    wrong = (pred_all != y) & valid

    ceiling = {
        "rule_match": float(
            (((pic >= 7).astype(int)) == y).mean()
        ),
        "bins": bins,
        "near_share_of_molecules": float(near.sum() / valid.sum()) if valid.any() else 0.0,
        "near_share_of_errors": float(
            (wrong & near).sum() / wrong.sum()
        ) if wrong.sum() else 0.0,
    }

    print("\nceiling:", ceiling)

    out = {
        "blocks": blocks,
        "ceiling": ceiling,
        "dataset": {
            "molecules": len(df),
            "inhibitors": int(y.sum()),
            "non_inhibitors": int((1 - y).sum()),
            "source": "MoleculeNet BACE",
        },
        "caveats": [
            "All numbers are out-of-fold. The stacker, threshold "
            "and early stopping use training data only.",
            "pIC50 and canvasUID are excluded from features. Class "
            "matches pIC50 >= 7 for 98.9% of molecules, and canvasUID "
            "is a row ID in a class-sorted file (0.9998 AUC alone).",
            "pIC50 is an extra training target for the GNN, never an input.",
            "The demo models train on all 1,513 molecules. Only the "
            "cross-validated numbers are test results.",
            "Scaffold splits are harder than random splits. Explanations "
            "show model behaviour, not binding mechanisms.",
        ],
    }

    ART.mkdir(exist_ok=True)

    metrics_path = ART / "metrics.json"
    metrics_path.write_text(
        json.dumps(out, indent=2),
        encoding="utf-8",
    )

    print(f"\nWrote {metrics_path}")


if __name__ == "__main__":
    main()
