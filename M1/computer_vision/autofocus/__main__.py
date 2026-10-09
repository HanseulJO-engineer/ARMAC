"""python -m autofocus: GUI by default, reproducible batch mode when requested."""
import argparse
from pathlib import Path
import sys
from .dataset import discover_sources, open_dataset
from .analysis import AnalysisOptions, analyze_dataset
from .metrics import BY_KEY, DEFAULT_METHODS


def main():
    parser = argparse.ArgumentParser(description="FocusLab desktop / static autofocus")
    parser.add_argument("path", nargs="?", default=str(Path(__file__).resolve().parents[1] / "Documents for autofocusing app"))
    parser.add_argument("--batch", action="store_true", help="Analyze all imagesets without launching the GUI")
    parser.add_argument("--auto-batch", action="store_true", help="Start all-set analysis after opening the desktop app")
    parser.add_argument("--output", default="output", help="Batch report destination")
    parser.add_argument("--methods", nargs="+", choices=list(BY_KEY), default=list(DEFAULT_METHODS))
    parser.add_argument("--all-methods", action="store_true")
    parser.add_argument("--search", choices=["gradient_verified", "multistart", "global"], default="gradient_verified")
    parser.add_argument("--fit-window", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0)
    parser.add_argument("--roi", type=float, nargs=4, default=[0,0,1,1], metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--manual-wd", action="store_true")
    parser.add_argument("--start-mm", type=float, default=10)
    parser.add_argument("--step-um", type=float, default=10)
    parser.add_argument("--metadata-offset", type=int, choices=[0,1], default=0)
    args = parser.parse_args()
    if not args.batch:
        from .gui import run_gui
        return run_gui(args.path, args.auto_batch)
    from .export import export_results
    options = AnalysisOptions(tuple(BY_KEY) if args.all_methods else tuple(args.methods), args.search, .25, args.fit_window, args.sigma, tuple(args.roi))
    sources = discover_sources(args.path)
    if not sources:
        parser.error("No image folders or archives found")
    results, errors = [], []
    for source in sources:
        try:
            dataset = open_dataset(source, args.start_mm, args.step_um, not args.manual_wd, args.metadata_offset)
            result = analyze_dataset(dataset, options)
            results.append(result)
            print(f"{dataset.name}: {len(result.entries)} images, {result.elapsed_seconds:.2f}s", flush=True)
            for warning in result.warnings:
                print(f"  {warning}", flush=True)
        except Exception as e:
            errors.append(f"{source}: {e}")
            print(errors[-1], file=sys.stderr, flush=True)
    if results:
        report = export_results(results, args.output)
        print(f"Report: {report}")
    return 1 if errors or not results else 0


if __name__ == "__main__":
    raise SystemExit(main())
