"""Run the two P0 analyses from the frozen AutoDL experiment suites."""

import argparse
import json
from pathlib import Path

import analyze_subgroup_results as subgroup
import benchmark_efficiency as efficiency


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frozen_prediction_paths(main_suite, baseline_suite):
    main_suite = Path(main_suite)
    baseline_suite = Path(baseline_suite)
    full = read_json(main_suite / "three_channel_full" / "completed.json")
    without = read_json(main_suite / "without_mvcm" / "completed.json")
    roberta = read_json(
        baseline_suite / "locked_test" / "roberta" / "completed.json"
    )
    paths = {
        "mvct": Path(full["evaluation_dir"]) / "test_predictions.json",
        "without_mvcm": Path(without["evaluation_dir"]) / "test_predictions.json",
        "roberta": Path(roberta["output_dir"]) / "test_predictions.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen prediction files: " + ", ".join(missing))
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-suite", required=True)
    parser.add_argument("--baseline-suite", required=True)
    parser.add_argument("--data", default="data_all.csv")
    parser.add_argument("--output-root", default="outputs/p0_submission_evidence")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--measured-batches", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--driver-version", default=None)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    predictions = frozen_prediction_paths(args.main_suite, args.baseline_suite)
    subgroup.main([
        "--data", args.data,
        "--prediction", f"mvct={predictions['mvct']}",
        "--prediction", f"without_mvcm={predictions['without_mvcm']}",
        "--prediction", f"roberta={predictions['roberta']}",
        "--output-dir", str(output_root / "subgroups"),
    ])
    efficiency.main([
        "--main-suite", args.main_suite,
        "--baseline-suite", args.baseline_suite,
        "--output-dir", str(output_root / "efficiency"),
        "--batch-sizes", *[str(value) for value in args.batch_sizes],
        "--warmup-batches", str(args.warmup_batches),
        "--measured-batches", str(args.measured_batches),
        "--repeats", str(args.repeats),
        "--driver-version", str(args.driver_version or "unknown"),
    ])


if __name__ == "__main__":
    main()
