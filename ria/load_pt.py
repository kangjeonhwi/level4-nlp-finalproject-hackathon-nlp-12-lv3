import os
import torch

def check_pt_files(directory):
    """
    Check if all .pt files in the given directory can be loaded successfully.

    Args:
        directory (str): Path to the directory containing .pt files.

    Returns:
        None
    """
    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    pt_files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    if not pt_files:
        print(f"No .pt files found in the directory '{directory}'.")
        return

    print(f"Checking .pt files in directory: {directory}\n")

    for file_name in pt_files:
        file_path = os.path.join(directory, file_name)
        print(f"Loading file: {file_name}")
        try:
            data = torch.load(file_path)
            print(f"  -> Successfully loaded {file_name}.")
            print(f"  -> Data type: {type(data)}, Shape: {getattr(data, 'shape', 'N/A')}\n")
        except Exception as e:
            print(f"  -> Failed to load {file_name}. Error: {e}\n")

if __name__ == "__main__":
    # Specify the directory containing the .pt files
    pt_files_directory = "output_embeddings"  # Adjust this path as needed

    # Run the check
    check_pt_files(pt_files_directory)