"""
This example serves as a test for transfer learning benchmarking setups.
It iterates over all combinations of methods and parameters to ensure they execute correctly for a few trials.
"""
import logging
import itertools
from benchmarking.benchmark_transfer import run
from benchmarking.benchmarks import benchmark_definitions
import sys
import os

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    methods = ["BoundingBox", "QuantileTransfer", "ZeroShot"]
    
    # We choose one representative benchmark
    benchmarks = [b for b in benchmark_definitions.keys() if "autoencodix" in b]
    test_bench = benchmarks[0] if benchmarks else "autoencodix-varix_schc-schc-RNA-METH-CLIN"

    # Identify target architecture
    target_arch = benchmark_definitions[test_bench].blackbox_name.split("_")[1]
    all_architectures = ["varix", "ontix", "disentanglix"]
    extra_archs = [a for a in all_architectures if a != target_arch]

    all_datasets_opts = [False, True]
    cross_arch_only_opts = [False, True]
    extra_archs_opts = [[], extra_archs]

    combinations = list(itertools.product(methods, all_datasets_opts, cross_arch_only_opts, extra_archs_opts))

    print(f"Testing {len(combinations)} combinations on {test_bench}...")

    success_count = 0
    failure_count = 0

    for method, all_datasets, cross_arch_only, e_archs in combinations:
        if cross_arch_only and not e_archs:
            # Skip invalid combination: cross_arch_only requires extra architectures
            continue
            
        print("\n" + "="*80)
        print(f"TESTING: method={method}, all_datasets={all_datasets}, cross_arch_only={cross_arch_only}, extra_archs={e_archs}")
        print("="*80)

        try:
            run(
                method_names=[method],
                benchmark_names=[test_bench],
                seeds=[42],
                max_num_evaluations=2,
                n_workers=1,
                all_datasets=all_datasets,
                cross_arch_only=cross_arch_only,
                extra_architectures=e_archs,
            )
            success_count += 1
            print(">>> SUCCESS\n")
        except Exception as e:
            failure_count += 1
            print(f">>> FAILED: {e}\n")
            logging.exception("Detailed error:")

    print(f"Done testing. {success_count} successful, {failure_count} failed.")
    if failure_count > 0:
        sys.exit(1)
