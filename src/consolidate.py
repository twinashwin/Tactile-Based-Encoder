"""Compare every tagged run, pick the winner, write ONE canonical result.

The ablation produced a report_data_<tag>.json per configuration. This collapses
them into results/final.json (the configuration that actually wins) and moves the
per-run files into results/runs/ so the results directory has a single obvious
answer in it rather than a dozen near-identical candidates.

Selection is on pooled macro-F1 of the best feature set, with the fold-to-fold
standard deviation reported alongside so a "win" inside the noise band is visible
as such rather than being presented as a result.
"""

import csv
import json
import shutil
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

CONFIGS = {
    "_cv5":    "resnet18, 1 frame/press",
    "_r18p9":  "resnet18, 9 frames pooled",
    "_r18p9t": "resnet18, 9 frames pooled + TTA",
    "_r50p9":  "resnet50, 9 frames pooled",
    "_dinop9": "DINOv2 ViT-S/14, 9 frames pooled",
}


def load(tag):
    """Look in results/ first, then results/runs/ for already-archived runs."""
    for base in (RESULTS, RESULTS / "runs"):
        p = base / f"report_data{tag}.json"
        if p.exists():
            return json.loads(p.read_text())
    return None


def best_run(data):
    """Best (feature set, model) cell in a configuration, by pooled macro-F1."""
    return max(data["runs"], key=lambda r: r["macro_f1"])


def main() -> None:
    loaded = {t: d for t in CONFIGS if (d := load(t)) is not None}
    if not loaded:
        raise SystemExit(f"no report_data_*.json in {RESULTS}")

    print(f"{'configuration':<36}{'dim':>6}{'best cell':<24}"
          f"{'acc':>8}{'macroF1':>9}{'fold sd':>9}")
    print("-" * 92)
    for tag, label in CONFIGS.items():
        if tag not in loaded:
            continue
        r = best_run(loaded[tag])
        cell = f"{r['feature_set']} / {r['model']}"
        print(f"{label:<36}{r['dim']:>6}{cell:<24}"
              f"{r['accuracy']:>8.4f}{r['macro_f1']:>9.4f}"
              f"{r.get('macro_f1_std', 0):>9.3f}")

    winner_tag = max(loaded, key=lambda t: best_run(loaded[t])["macro_f1"])
    win = loaded[winner_tag]
    wr = best_run(win)
    win["config"] = CONFIGS[winner_tag]
    win["config_tag"] = winner_tag
    # Keep the ablation ladder so the report can show what each change bought.
    win["ablation"] = [
        {"config": CONFIGS[t],
         "dim": best_run(loaded[t])["dim"],
         "cell": f"{best_run(loaded[t])['feature_set']} / "
                 f"{best_run(loaded[t])['model']}",
         "accuracy": best_run(loaded[t])["accuracy"],
         "macro_f1": best_run(loaded[t])["macro_f1"],
         "macro_f1_std": best_run(loaded[t]).get("macro_f1_std", 0.0),
         "is_winner": t == winner_tag}
        for t in CONFIGS if t in loaded]

    # Is the winner actually distinguishable from the runner-up?
    ranked = sorted(loaded, key=lambda t: best_run(loaded[t])["macro_f1"],
                    reverse=True)
    print(f"\nWINNER: {CONFIGS[winner_tag]}")
    print(f"  {wr['feature_set']} / {wr['model']}: "
          f"acc {wr['accuracy']:.4f}, macro-F1 {wr['macro_f1']:.4f} "
          f"+/- {wr.get('macro_f1_std', 0):.3f} over {wr['folds']} folds")
    if len(ranked) > 1:
        second = best_run(loaded[ranked[1]])
        margin = wr["macro_f1"] - second["macro_f1"]
        sd = wr.get("macro_f1_std", 0)
        verdict = ("INSIDE the noise band - treat as a tie"
                   if margin < sd else "outside one fold-sd")
        print(f"  margin over '{CONFIGS[ranked[1]]}': {margin:+.4f}  ({verdict})")

    (RESULTS / "final.json").write_text(json.dumps(win))

    # Flat, one row per result. final.json keeps the confusion matrices and
    # per-class arrays the report needs; this is the version meant to be read.
    rows = []
    for tag, label in CONFIGS.items():
        if tag not in loaded:
            continue
        for r in loaded[tag]["runs"]:
            backbone, _, variant = label.partition(", ")
            rows.append({
                "backbone": backbone,
                "variant": variant,
                "feature_set": r["feature_set"],
                "model": r["model"],
                "dim": r["dim"],
                "accuracy": round(r["accuracy"], 4),
                "macro_f1": round(r["macro_f1"], 4),
                "macro_f1_std": round(r.get("macro_f1_std", 0.0), 4),
                "folds": r.get("folds", 1),
                "winner": "yes" if (tag == winner_tag and
                                    r["feature_set"] == wr["feature_set"] and
                                    r["model"] == wr["model"]) else "",
            })
    rows.sort(key=lambda d: d["macro_f1"], reverse=True)
    with (RESULTS / "final.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {RESULTS / 'final.csv'} ({len(rows)} rows, one per result)")
    print(f"\nwrote {RESULTS / 'final.json'}")

    archive = RESULTS / "runs"
    archive.mkdir(exist_ok=True)
    moved = 0
    for p in list(RESULTS.glob("report_data_*.json")) + \
             list(RESULTS.glob("results*.json")) + \
             list(RESULTS.glob("confusion_*.png")):
        if p.name == "final.json":
            continue
        shutil.move(str(p), str(archive / p.name))
        moved += 1
    for stale in (RESULTS / "report_data.json",):
        if stale.exists():
            shutil.move(str(stale), str(archive / stale.name))
            moved += 1
    print(f"archived {moved} per-run files to {archive}")


if __name__ == "__main__":
    main()
