import subprocess
import os
import time

# CONFIGURATION
COMMANDS_FILE = "launch_transfer_only.txt"
LOG_DIR = "logs"

def generate_commands():
    architectures = ["vanillix", "varix", "ontix", "disentanglix"]
    tasks = [
        "schc_RNA_METH_CLIN",
        "schc_METH_CLIN",
        "schc_RNA_CLIN",
        "tcga_RNA_CLIN",
        "tcga_METH_CLIN",
        "tcga_DNA_CLIN",
        "tcga_RNA_DNA_METH_CLIN",
    ]
    # Adjust tasks to include ontologies for ontix
    task_variants = {}
    for arch in architectures:
        if arch == "ontix":
            task_variants[arch] = [f"{t}_reactome" for t in tasks] + [f"{t}_chromosome" for t in tasks]
        else:
            task_variants[arch] = tasks
            
    methods = [
        "launch_bounding_box.py",
        "launch_quantile_transfer.py",
        "launch_zero_shot.py"
    ]
    
    seeds = [42] # Modify if more seeds are needed
    
    commands = []
    
    for method in methods:
        for arch in architectures:
            for task in task_variants[arch]:
                for seed in seeds:
                    base_cmd = f"python {method} --architecture {arch} --test-task {task} --seed {seed}"
                    
                    # Setup 1: Baseline Transfer (Same Architecture, Same Dataset)
                    # This relies on the default --same-dataset-only (no --all-datasets flag) and no extra architectures.
                    commands.append(f"{base_cmd}")
                    
                    # Setup 2: Cross-Dataset (Same Architecture, Across Datasets)
                    # Adds --all-datasets
                    commands.append(f"{base_cmd} --all-datasets")
                    
                    # Setup 3: Cross-Architecture (Different architectures as sources, plus same-arch sources)
                    # Provide all other architectures as extra architectures
                    extra_archs = " ".join([a for a in architectures if a != arch])
                    if method == "launch_zero_shot.py":
                        # Require surrogate for cross-architecture if config spaces differ, but here let's assume it's best to use it when grids might differ.
                        # Wait, launch_zero_shot.py actually mentions that surrogates are required if HP grids differ across tasks. Let's add --use-surrogates just in case.
                        commands.append(f"{base_cmd} --extra-architectures {extra_archs} --use-surrogates")
                    else:
                        commands.append(f"{base_cmd} --extra-architectures {extra_archs}")
                        
                    # Setup 4: Cross-Architecture-Only (Isolating architectural transfer, no same-arch sources)
                    if method == "launch_zero_shot.py":
                        commands.append(f"{base_cmd} --extra-architectures {extra_archs} --cross-arch-only --use-surrogates")
                    else:
                        commands.append(f"{base_cmd} --extra-architectures {extra_archs} --cross-arch-only")
                        
    # Note: Cross-Modality is naturally covered because the datasets in `tasks` consist of different modalities
    # (e.g., RNA_METH vs RNA vs METH), and the setup transfers from all other tasks within the dataset or across datasets.

    with open(COMMANDS_FILE, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")
            
    print(f"Generated {len(commands)} commands in {COMMANDS_FILE}")


def main():
    # Generate commands first
    generate_commands()
    
    # 1. Ensure log directory exists
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # 2. Read commands from the file
    if not os.path.exists(COMMANDS_FILE):
        print(f"Error: {COMMANDS_FILE} not found.")
        return

    with open(COMMANDS_FILE, "r") as f:
        commands = [line.strip() for line in f if line.strip()]

    total_commands = len(commands)
    print(f"Starting sequential execution of {total_commands} commands.")
    print("-" * 60)

    for index, cmd in enumerate(commands):
        # Create a clean log filename
        # e.g., 001_launch_zero_shot_varix_tcga_RNA_CLIN.log
        clean_cmd_name = cmd.replace("python ", "").replace("--", "").replace(" ", "_").replace("/", "_")[:80]
        log_path = os.path.join(LOG_DIR, f"{index + 1:04d}_{clean_cmd_name}.log")

        print(f"[{index + 1}/{total_commands}] Running: {cmd}")
        start_time = time.time()

        try:
            # shell=True allows the execution of the full command string
            # we use 'with' to ensure the file is closed immediately after the command finishes
            with open(log_path, "w") as log_file:
                subprocess.run(cmd, shell=True, check=True, stdout=log_file, stderr=log_file)

            elapsed = time.time() - start_time
            print(f"    Success (took {elapsed:.2f}s). Log: {log_path}")

        except subprocess.CalledProcessError:
            print(f"    ERROR: Command failed. Check {log_path} for details.")
            # Optional: break if you want to stop the whole queue on the first error
            # break

    print("-" * 60)
    print("All tasks completed.")


if __name__ == "__main__":
    main()
