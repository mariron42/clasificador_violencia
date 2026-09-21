"""Mega-ensemble: combina probabilidades de modelos clásicos + encoders.

Lee los verbose CSVs de todos los experimentos (clásicos y encoders),
busca la combinación óptima por Nelder-Mead sobre macro-F1 en devel,
y genera la predicción final para submission.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from scipy.optimize import minimize

from womenhelp_competition.labels import SUBTASK1_ID_TO_NAME, SUBTASK1_LABELS
from womenhelp_competition.metrics import compute_multiclass_metrics

NUM_CLASSES = len(SUBTASK1_LABELS)
SEED = 3407


def parse_args():
    p = argparse.ArgumentParser(description="Mega-ensemble: combina clásico + encoder")
    p.add_argument("--search-dirs", nargs="+", required=True,
                   help="Directorios donde buscar subdirs con subtask1_verbose.csv")
    p.add_argument("--output-dir",
                   default=str(REPO_ROOT / "outputs" / "mega_ensemble" / datetime.now().strftime("%Y%m%d_%H%M")))
    p.add_argument("--n-restarts", type=int, default=10)
    p.add_argument("--top-k", type=int, default=0,
                   help="Si >0, solo usa los top-k modelos por devel macro F1")
    return p.parse_args()


def ensure_probs(p):
    c = np.clip(np.asarray(p, dtype=np.float64), 1e-9, None)
    return c / c.sum(axis=1, keepdims=True).clip(1e-9)


def load_verbose_csv(path: Path):
    """Load a verbose CSV and return (record_ids, gold_ids, prob_matrix)."""
    ids, golds, probs = [], [], []
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["record_id"])
            golds.append(int(row["gold_class_id"]))
            prob_keys = [f"prob_{label.lower()}" for label in SUBTASK1_LABELS]
            if all(key in row and row[key] not in {None, ""} for key in prob_keys):
                probs.append([float(row[key]) for key in prob_keys])
                continue

            # Algunos artefactos viejos solo guardan la clase predicha. En ese caso
            # degradamos a una distribucion one-hot para no perder diversidad del modelo.
            pred_class_id = int(row["pred_class_id"])
            one_hot = [0.0] * NUM_CLASSES
            one_hot[pred_class_id] = 1.0
            probs.append(one_hot)
    return ids, np.array(golds, dtype=np.int32), ensure_probs(np.array(probs))


def discover_models(search_dirs):
    """Find all subtask1_verbose.csv files across search directories."""
    models = {}
    for sd in search_dirs:
        p = Path(sd)
        if not p.exists():
            continue
        for csv_file in sorted(p.rglob("subtask1_verbose.csv")):
            # Use parent dir name as model name, prefixed with grandparent
            parent = csv_file.parent.name
            grandparent = csv_file.parent.parent.name
            name = f"{grandparent}/{parent}"
            if name not in models:
                models[name] = csv_file
    return models


def calc_macro_f1(gold, probs):
    pred_ids = probs.argmax(axis=1)
    return calc_macro_f1_from_ids(gold, pred_ids)


def calc_macro_f1_from_ids(gold, pred_ids):
    gold_names = [SUBTASK1_LABELS[int(i)] for i in gold]
    pred_names = [SUBTASK1_LABELS[int(i)] for i in pred_ids]
    m = compute_multiclass_metrics(gold_names, pred_names, SUBTASK1_LABELS)
    return m["macro_f1"], m


def ordinal_tail_probs(probs):
    values = np.asarray(probs, dtype=np.float64)
    return np.column_stack([
        values[:, 1:].sum(axis=1),
        values[:, 2:].sum(axis=1),
        values[:, 3],
    ])


def ordinal_threshold_ids(probs, thresholds):
    tails = ordinal_tail_probs(probs)
    pred_ids = np.zeros(tails.shape[0], dtype=np.int32)
    for threshold_index, threshold_value in enumerate(thresholds):
        pred_ids += (tails[:, threshold_index] >= float(threshold_value)).astype(np.int32)
    return pred_ids


def metric_priority(metrics):
    severe = metrics["per_label"].get("Severe", {})
    return (
        float(metrics["macro_f1"]),
        float(severe.get("recall", 0.0)),
        float(metrics["accuracy"]),
        float(metrics["weighted_f1"]),
    )


def search_ordinal_thresholds(gold, probs):
    best_thresholds = [0.5, 0.5, 0.5]
    best_pred_ids = probs.argmax(axis=1)
    _, best_metrics = calc_macro_f1_from_ids(gold, best_pred_ids)
    best_key = metric_priority(best_metrics)

    for t0 in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        for t1 in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
            for t2 in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
                pred_ids = ordinal_threshold_ids(probs, [t0, t1, t2])
                _, metrics = calc_macro_f1_from_ids(gold, pred_ids)
                metrics_key = metric_priority(metrics)
                if metrics_key > best_key:
                    best_key = metrics_key
                    best_thresholds = [t0, t1, t2]
                    best_pred_ids = pred_ids
                    best_metrics = metrics

    return best_thresholds, best_pred_ids, best_metrics


def avg_probs(prob_list, weights=None):
    stacked = np.stack(prob_list, axis=0)
    if weights is None:
        return ensure_probs(stacked.mean(axis=0))
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum().clip(1e-9)
    return ensure_probs(np.tensordot(w, stacked, axes=(0, 0)))


def optimize_weights(prob_list, gold, n_restarts=10):
    n = len(prob_list)
    best_score = -1.0
    best_w = np.ones(n) / n

    def neg_f1(w):
        w_pos = np.abs(w)
        w_norm = w_pos / w_pos.sum().clip(1e-9)
        combined = avg_probs(prob_list, weights=w_norm)
        f1, _ = calc_macro_f1(gold, combined)
        return -f1

    rng = np.random.default_rng(SEED)
    for _ in range(n_restarts):
        w0 = rng.dirichlet(np.ones(n))
        result = minimize(neg_f1, w0, method="Nelder-Mead",
                          options={"maxiter": 1000, "xatol": 1e-5, "fatol": 1e-6})
        if -result.fun > best_score:
            best_score = -result.fun
            w_abs = np.abs(result.x)
            best_w = w_abs / w_abs.sum().clip(1e-9)
    return best_w, best_score


def write_submission(path, pred_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        for pid in pred_ids:
            w.writerow([pid])


def write_verbose_csv(path, record_ids, gold_ids, probs):
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_ids = probs.argmax(axis=1)
    write_verbose_csv_with_ids(path, record_ids, gold_ids, probs, pred_ids)


def write_verbose_csv_with_ids(path, record_ids, gold_ids, probs, pred_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["record_id", "gold_class_id", "pred_class_id"] + [
        f"prob_{label.lower()}" for label in SUBTASK1_LABELS
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record_id, gold_id, pred_id, prob_row in zip(record_ids, gold_ids, pred_ids, probs):
            payload = {
                "record_id": record_id,
                "gold_class_id": int(gold_id),
                "pred_class_id": int(pred_id),
            }
            for label, prob in zip(SUBTASK1_LABELS, prob_row):
                payload[f"prob_{label.lower()}"] = float(prob)
            writer.writerow(payload)


def write_report(path, best_name, best_f1, best_metrics, best_components, top_results):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Mega ensemble de severidad",
        "",
        f"Generado: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Mejor variante: `{best_name}`",
        f"- Macro-F1 devel: `{best_f1:.4f}`",
        f"- Accuracy devel: `{best_metrics['accuracy']:.4f}`",
        f"- Weighted-F1 devel: `{best_metrics['weighted_f1']:.4f}`",
        "",
        "## Componentes del mejor ensemble",
        "",
    ]
    for component in best_components:
        lines.append(f"- `{component}`")
    lines.extend([
        "",
        "## Top resultados",
        "",
        "| variante | macro_f1 | componentes |",
        "|---|---:|---|",
    ])
    for row in top_results:
        components = ", ".join(row["components"])
        calibration = row.get("calibration")
        if calibration is not None:
            components = f"{components} | ordcal={calibration['thresholds']}"
        lines.append(f"| {row['name']} | {row['macro_f1']:.4f} | {components} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    t0 = perf_counter()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Discovering models ...", flush=True)
    model_files = discover_models(args.search_dirs)
    print(f"Found {len(model_files)} models:", flush=True)

    # Load all and verify they have the same record IDs
    ref_ids = None
    ref_gold = None
    model_probs = {}
    model_f1s = {}

    for name, csv_path in sorted(model_files.items()):
        try:
            ids, gold, probs = load_verbose_csv(csv_path)
        except Exception as e:
            print(f"  SKIP {name}: {e}", flush=True)
            continue

        if ref_ids is None:
            ref_ids = ids
            ref_gold = gold
        elif ids != ref_ids:
            print(f"  SKIP {name}: mismatched record IDs ({len(ids)} vs {len(ref_ids)})", flush=True)
            continue

        f1, metrics = calc_macro_f1(gold, probs)
        model_probs[name] = probs
        model_f1s[name] = f1
        print(f"  {name:50s} macro_f1={f1:.4f}", flush=True)

    if len(model_probs) < 2:
        print("Not enough models for ensembling", flush=True)
        return

    # Sort by F1
    sorted_models = sorted(model_f1s.items(), key=lambda x: -x[1])
    all_names = [n for n, _ in sorted_models]
    all_probs = [model_probs[n] for n in all_names]

    print(f"\n=== Ensembling {len(all_names)} models ===", flush=True)

    results = []
    result_pred_ids = {}
    result_calibration = {}

    def add_result(name, probs, metrics, components, *, pred_ids=None, calibration=None):
        if pred_ids is None:
            pred_ids = probs.argmax(axis=1)
        results.append((name, metrics["macro_f1"], probs, metrics, components))
        result_pred_ids[name] = pred_ids
        result_calibration[name] = calibration

        ord_thresholds, ord_pred_ids, ord_metrics = search_ordinal_thresholds(ref_gold, probs)
        ord_name = f"{name}_ordcal"
        results.append((ord_name, ord_metrics["macro_f1"], probs, ord_metrics, components))
        result_pred_ids[ord_name] = ord_pred_ids
        result_calibration[ord_name] = {"type": "ordinal_thresholds", "thresholds": ord_thresholds}

    # 1. Simple mean
    mean_p = avg_probs(all_probs)
    f1_mean, m_mean = calc_macro_f1(ref_gold, mean_p)
    add_result("mean_all", mean_p, m_mean, all_names)
    print(f"  mean_all: {f1_mean:.4f}", flush=True)

    # 2. Top-K subsets
    if args.top_k > 0:
        top_names = all_names[:args.top_k]
        top_probs_list = [model_probs[n] for n in top_names]
    else:
        for k in [3, 5, 8, 10]:
            if k >= len(all_names):
                continue
            tk_names = all_names[:k]
            tk_probs = [model_probs[n] for n in tk_names]
            tk_mean = avg_probs(tk_probs)
            f1_tk, m_tk = calc_macro_f1(ref_gold, tk_mean)
            add_result(f"mean_top{k}", tk_mean, m_tk, tk_names)
            print(f"  mean_top{k}: {f1_tk:.4f}", flush=True)

    # 3. Macro-weighted
    weights = np.array([model_f1s[n] for n in all_names])
    wm_p = avg_probs(all_probs, weights)
    f1_wm, m_wm = calc_macro_f1(ref_gold, wm_p)
    add_result("macro_weighted", wm_p, m_wm, all_names)
    print(f"  macro_weighted: {f1_wm:.4f}", flush=True)

    # 4. Optimized weights on all
    if len(all_probs) <= 20:
        opt_w, opt_score = optimize_weights(all_probs, ref_gold, args.n_restarts)
        opt_p = avg_probs(all_probs, opt_w)
        f1_opt, m_opt = calc_macro_f1(ref_gold, opt_p)
        add_result("optimized_all", opt_p, m_opt, all_names)
        print(f"  optimized_all: {f1_opt:.4f}", flush=True)
        # Log weights
        for n, w in zip(all_names, opt_w):
            if w > 0.01:
                print(f"    {n}: {w:.3f}", flush=True)

    # 5. Optimized on top-K
    for k in [5, 8, 10]:
        if k >= len(all_names):
            continue
        tk_names = all_names[:k]
        tk_probs = [model_probs[n] for n in tk_names]
        opt_w_k, opt_score_k = optimize_weights(tk_probs, ref_gold, args.n_restarts)
        opt_p_k = avg_probs(tk_probs, opt_w_k)
        f1_opt_k, m_opt_k = calc_macro_f1(ref_gold, opt_p_k)
        add_result(f"optimized_top{k}", opt_p_k, m_opt_k, tk_names)
        print(f"  optimized_top{k}: {f1_opt_k:.4f}", flush=True)

    # 6. Anchor pairs with best model
    best_name = all_names[0]
    for partner in all_names[1:min(10, len(all_names))]:
        pair_p = avg_probs([model_probs[best_name], model_probs[partner]])
        f1_pair, m_pair = calc_macro_f1(ref_gold, pair_p)
        add_result(f"pair_{best_name}+{partner}", pair_p, m_pair, [best_name, partner])

    # Sort results
    results.sort(key=lambda x: -x[1])

    print(f"\n=== TOP 10 RESULTS ===", flush=True)
    for name, f1, _, metrics, components in results[:10]:
        print(f"  {name:55s} macro_f1={f1:.4f}", flush=True)

    # Write best
    best_name, best_f1, best_probs, best_metrics, best_components = results[0]
    best_dir = out_dir / best_name.replace("/", "_")
    best_dir.mkdir(parents=True, exist_ok=True)
    pred_ids = result_pred_ids[best_name]
    write_submission(best_dir / "submission" / "subtask1.csv",
                     [str(int(i)) for i in pred_ids])
    write_verbose_csv_with_ids(best_dir / "subtask1_verbose.csv", ref_ids, ref_gold, best_probs, pred_ids)

    # Write summary
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": perf_counter() - t0,
        "num_source_models": len(model_probs),
        "best_name": best_name,
        "best_macro_f1": best_f1,
        "best_metrics": best_metrics,
        "best_components": best_components,
        "best_calibration": result_calibration[best_name],
        "all_results": [
            {"name": n, "macro_f1": f1, "components": c, "calibration": result_calibration[n]}
            for n, f1, _, _, c in results[:20]
        ],
        "individual_scores": dict(sorted_models),
    }
    (out_dir / "mega_ensemble_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))

    metrics_summary = {
        "model_name_or_path": "mega_ensemble",
        "family": "mega_ensemble",
        "description": "combina modelos clasicos y encoders de severidad sobre devel",
        "subtask1_split": "devel",
        "subtask1_records": len(ref_gold),
        "subtask1_metrics": best_metrics,
        "best_name": best_name,
        "best_macro_f1": best_f1,
        "best_components": best_components,
        "best_calibration": payload["best_calibration"],
        "num_source_models": len(model_probs),
        "generated_at": payload["generated_at"],
        "runtime_seconds": payload["runtime_seconds"],
    }
    (best_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2, ensure_ascii=False)
    )
    write_report(
        best_dir / "report.md",
        best_name=best_name,
        best_f1=best_f1,
        best_metrics=best_metrics,
        best_components=best_components,
        top_results=payload["all_results"][:10],
    )

    print(f"\nBest: {best_name} macro_f1={best_f1:.4f}", flush=True)
    print(f"Output: {out_dir}", flush=True)
    print(f"Done in {perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
