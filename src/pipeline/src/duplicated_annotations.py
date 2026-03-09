import os
import hashlib
from tqdm import tqdm
from pathlib import Path
from typing import List

def file_hash(filepath, hash_algo=hashlib.sha256, chunk_size=4096):
    hash_func = hash_algo()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def get_deduped_filepaths(directory: str, state=None, skip_annotations: List[str] = None) -> List[str]:
    """
    Finds unique files in the given directory based on their content hash.
    Records statistics into the state object if provided.
    Returns a list of deduplicated file paths (absolute).
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
        
    if skip_annotations is None:
        skip_annotations = []
        
    # Process only geojson files
    all_files = [f for f in directory_path.iterdir() if f.is_file() and f.suffix == '.geojson']
    
    if state:
        state.total_annotations_found = len(all_files)
        hashes_dict = {}
        for file_path in all_files:
            h = file_hash(file_path)
            hashes_dict.setdefault(h, []).append(file_path.name)
        
        duplicates = {h: files for h, files in hashes_dict.items() if len(files) > 1}
        state.duplicates = duplicates
    
    seen_hashes = set()
    unique_files = []
    
    print("\n--- Stage 1: Deduplication ---")
    for file_path in tqdm(all_files, desc="Deduplicating Annotations"):
        h = file_hash(file_path)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_files.append(str(file_path.resolve()))
            
    if state:
        state.unique_annotations = len(unique_files)
        
    print(f"Found {len(unique_files)} unique files out of {len(all_files)} total files.")

    filtered_files = [fname for fname in unique_files if fname not in skip_annotations]
    print(f"Skipped {len(unique_files) - len(filtered_files)} annotations.")

    return filtered_files
