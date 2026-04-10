import subprocess
import os
import time

# CONFIGURATION
COMMANDS_FILE = "launch_transfer_only.txt"
LOG_DIR = "logs"


def main():
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