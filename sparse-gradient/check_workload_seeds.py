"""Small CPU regression checks; optionally validate and plot existing run artifacts."""

import argparse
import itertools
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import sparse_grad_benchmark as benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--save-plots", action="store_true")
    options = parser.parse_args()
    checked = 0
    for mode, layout, order, dist, draw in itertools.product(
        ["truncated", "wrapped"], ["scattered", "adjacent"], ["random", "sorted"],
        ["uniform", "hotspot", "zipf"], range(2),
    ):
        args = argparse.Namespace(zipf_mode=mode, zipf_alpha=1.2, row_layout=layout,
                                  index_order=order, hotspot_rows=16, hotspot_fraction=0.8)
        arrays, index_seeds, source_seeds = [], [], []
        for dim in (64, 512):
            index_seed = benchmark.index_seed_for(1234, 1024, 2048, dist, layout, order, mode, draw)
            source_seed = benchmark.src_seed_for(1234, 1024, 2048, dim, dist, layout, order, mode, draw)
            arrays.append(benchmark.generate_workload_indices(1024, 2048, dist, args, index_seed))
            index_seeds.append(index_seed)
            source_seeds.append(source_seed)
        assert index_seeds[0] == index_seeds[1]
        assert source_seeds[0] != source_seeds[1]
        assert np.array_equal(arrays[0], arrays[1])
        assert benchmark.index_stats(arrays[0]) == benchmark.index_stats(arrays[1])
        assert np.all((arrays[0] >= 0) & (arrays[0] < 1024))
        checked += 1
    print(f"Verified exact arrays/statistics for dim=64/512 across {checked} settings.")
    print(f"Example index seeds: {index_seeds}; source seeds: {source_seeds}")

    # A small table distinguishes proper finite probabilities from a wrapped tail.
    samples = benchmark.sample_truncated_zipf(4, 100000, 1.2, np.random.default_rng(42))
    expected = np.arange(1, 5, dtype=float) ** -1.2
    expected /= expected.sum()
    assert np.allclose(np.bincount(samples, minlength=4) / samples.size, expected, atol=0.005)
    args.zipf_mode = "wrapped"
    actual = benchmark.generate_indices(4, 1000, "zipf", args, np.random.default_rng(42))
    legacy = (np.random.default_rng(42).zipf(args.zipf_alpha, 1000) - 1) % 4
    assert np.array_equal(actual, legacy)

    # An operation winning one of three draws must have win_rate=1/3, not 1.
    fake = pd.DataFrame({"rows": [4]*3, "indices": [10]*3, "dim": [16]*3,
                         "distribution": ["zipf"]*3, "zipf_mode": ["truncated"]*3,
                         "row_layout": ["scattered"]*3, "index_order": ["random"]*3,
                         "cache_mode": ["warm"]*3, "workload_draw": [0, 1, 2],
                         "winner": ["index_add", "sorted_segment", "sorted_segment"],
                         "speedup_vs_index_add": [1., 2., 3.], "winner_ms": [1., .5, .3]})
    with tempfile.TemporaryDirectory(prefix="sparse_grad_check_") as directory:
        grouped = benchmark.write_winner_summary(fake, Path(directory))
        assert grouped.total_draws.eq(3).all()
        assert grouped.loc[grouped.operation.eq("index_add"), "win_rate"].iloc[0] == 1/3
        assert benchmark.write_winner_summary(fake.iloc[:0], Path(directory)).empty
    print("Truncated probabilities, exact wrapped compatibility, and win-rate denominators passed.")

    if options.results:
        path = options.results
        raw = pd.read_csv(path / "raw_results.csv")
        summary = pd.read_csv(path / "summary_results.csv")
        winners = pd.read_csv(path / "winners.csv")
        wins = pd.read_csv(path / "winner_summary.csv")
        metadata = json.loads((path / "metadata.json").read_text())
        assert len(summary) == metadata["completed_combinations"] == metadata["planned_combinations"]
        assert len(raw) == len(summary) * metadata["arguments"]["repeats"]
        assert raw.workload_seed.eq(raw.index_seed).all()
        assert summary.status.eq("ok").all()
        assert summary.loc[summary.has_output, "correctness"].eq("passed").all()
        keys = ["rows", "indices", "distribution", "zipf_mode", "workload_draw",
                "row_layout", "index_order"]
        stats = ["index_seed", "unique_indices", "max_row_count", "top1_share", "top16_share"]
        assert summary.groupby(keys)[stats].nunique().eq(1).all().all()
        assert wins.total_draws.eq(metadata["workload_seeds"]).all()
        assert metadata["methodology_version"] == 3
        if options.save_plots:
            assert benchmark.save_speedup_plots(winners.iloc[:0], path) == []
            metadata["plot_files"] = benchmark.save_plots(summary, path)
            metadata["plot_files"] += benchmark.save_speedup_plots(winners, path)
            benchmark.save_metadata(path / "metadata.json", metadata)
            assert (path / "speedup_vs_top1_share_fair_dense_grad.png").exists()
            assert (path / "speedup_vs_max_row_count_fair_dense_grad.png").exists()
        print(f"Validated {len(raw)} samples, {len(summary)} summaries, {len(winners)} per-draw winners: {path}")


if __name__ == "__main__":
    main()
