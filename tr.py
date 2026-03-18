# import os
# import shutil

# # Source and destination folders
# source_folder = r"data\test"
# destination_folder = r"data\train"

# # Create destination folder if it does not exist
# os.makedirs(destination_folder, exist_ok=True)

# # Loop through numbers 0000 → 0230
# for i in range(1000):
    
#     # Original index (0000 → 0230)
#     old_index = f"{i:04d}"
    
#     # New index (1000 → 1230)
#     new_index = f"{1000 + i:04d}"

#     # Search for files ending with this index
#     for file in os.listdir(source_folder):
#         if file.endswith(f"_{old_index}.csv"):
            
#             old_path = os.path.join(source_folder, file)

#             # Replace index in filename
#             new_filename = file.replace(f"_{old_index}", f"_{new_index}")
#             new_path = os.path.join(destination_folder, new_filename)

#             # Copy and rename
#             shutil.copy(old_path, new_path)

#             print(f"Copied: {file} → {new_filename}")
import intel_extension_for_pytorch
print("IPEX loaded")