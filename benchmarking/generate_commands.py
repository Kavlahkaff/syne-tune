# generate_commands.py
scripts = [#"launch_zero_shot.py",
           #"launch_bounding_box.py",
           "launch_quantile_transfer.py"]
architectures = ["vanillix", "varix", "ontix", "disentanglix"]


selected_tasks = [
    "schc_RNA_METH_CLIN",
    "schc_METH_CLIN",
    "schc_RNA_CLIN",
    "tcga_RNA_CLIN",
    "tcga_METH_CLIN",
    "tcga_DNA_CLIN",
    "tcga_RNA_DNA_METH_CLIN",
]

ontix_selected = ["schc_RNA_METH_CLIN_reactome",
    "schc_METH_CLIN_reactome",
    "schc_RNA_CLIN_reactome",
    "tcga_RNA_CLIN_reactome",
    "tcga_METH_CLIN_reactome",
    "tcga_DNA_CLIN_reactome",
    "tcga_RNA_DNA_METH_CLIN_reactome",
    "schc_RNA_METH_CLIN_chromosome",
    "schc_METH_CLIN_chromosome",
    "schc_RNA_CLIN_chromosome",
    "tcga_RNA_CLIN_chromosome",
    "tcga_METH_CLIN_chromosome",
    "tcga_DNA_CLIN_chromosome",
    "tcga_RNA_DNA_METH_CLIN_chromosome",]

seeds = range(0, 30)
commands = []

for script in scripts:
    for arch in architectures:
        tasks = ontix_selected if arch == "ontix" else selected_tasks
        extra_archs = " ".join([a for a in architectures if a != arch])

        for task in tasks:
            # Special handling for Bounding Box searchers
            if "bounding_box" in script:
                searchers = ["random_search", "cqr"]
            else:
                searchers = [None]  # Use default for other scripts

            for searcher in searchers:
                search_arg = f"--searcher {searcher}" if searcher else ""

                for seed in seeds:
                    # Construct the base call
                    cmd_base = f"python {script} --architecture {arch} --test-task {task} {search_arg}".strip()

                    # CASE 1: Domain Transfer (Same Architecture, All Datasets)
                    commands.append(f"{cmd_base} --all-datasets --seed {seed}")

                    # CASE 2: Full Hybrid Transfer (Same + Cross Architecture, All Datasets)
                    commands.append(f"{cmd_base} --all-datasets --extra-architectures {extra_archs} --seed {seed}")

# Write out
output_file = "launch_transfer_only.txt"
with open(output_file, "w") as f:
    f.write("\n".join(commands) + "\n")

print(f"Generated {len(commands)} transfer-learning focused commands.")