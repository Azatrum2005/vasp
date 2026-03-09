import os

# Source and destination directories
src_dir = "LCMGM_novel_materials/CIFs"
dst_dir = "LCMGM_novel_materials/named_cif"

# Create destination directory if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

# Loop over all CIF files
for filename in os.listdir(src_dir):
    if filename.lower().endswith(".cif"):
        src_path = os.path.join(src_dir, filename)

        with open(src_path, "r") as f:
            lines = f.readlines()

        if not lines:
            continue  # skip empty files

        # Read first line and extract name
        first_line = lines[0].strip()
        print(first_line)
        # Expected format: data_CsMnO3
        if first_line.startswith("data_"):
            material_name = first_line.replace("data_", "", 1)
        else:
            material_name = first_line

        # New file path
        new_filename = material_name + ".cif"
        dst_path = os.path.join(dst_dir, new_filename)

        # Write full content to new file
        with open(dst_path, "w") as f:
            f.writelines(lines)
