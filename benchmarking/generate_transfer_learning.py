"""
This example serves as a test for transfer learning benchmarking setups.
It iterates over all combinations of methods and parameters and generates
a text file containing all the commands to run each experiment as a separate job.
"""
import logging
from benchmarking.benchmarks import benchmark_definitions
import sys

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    methods = [
        "BoundingBox",
        "QuantileTransfer",
        "ZeroShot"
    ]
    
    # Run all benchmarks for all test tasks
    benchmarks = [b for b in benchmark_definitions.keys() if "bbomix" in b]

    all_architectures = ["ontix", "disentanglix", "vanillix" ,"varix"]

    print(f"Generating commands for methods {methods} on {len(benchmarks)} benchmarks for 30 seeds...")

    commands = []

    for method in methods:
        for bench in benchmarks:
            # benchmark format: bbomix-vanillix_schc-schc-RNA-CLIN
            parts = bench.split("-")
            target_arch = parts[1].split("_")[0]

            for seed in range(30):
                base_cmd = (
                    f"python benchmarking/benchmark_transfer.py "
                    f"--method {method} --benchmark {bench} "
                    f"--seed {seed} --cross_arch_only"
                )

                # Get all other architectures (exclude target)
                all_others = [arch for arch in all_architectures if arch != target_arch]

                # 1. Original behavior: cross architecture runs of all other architectures
                all_others_str = " ".join(all_others)
                commands.append(f"{base_cmd} --extra_architectures {all_others_str}")

                # 2. Additional behavior: cross architecture runs of ONLY ONE other architecture
                for arch in all_others:
                    commands.append(f"{base_cmd} --extra_architectures {arch}")

    output_file = "launch_transfer_learning_commands.txt"
    with open(output_file, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")

    print(f"Generated {len(commands)} commands and saved to '{output_file}'.")
    print("Done generating commands.")
