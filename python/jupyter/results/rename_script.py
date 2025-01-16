import json
from pathlib import Path
import re

def rename_benchmark_files(directory):
    """Rename benchmark files according to metadata
    
    Args:
        directory: Path to directory containing benchmark files
    """
    # Mode shortening mapping
    mode_shorts = {
        'random_vectors': 'rv',
        'random_embeddings': 're',
        'query_file': 'q'
    }
    
    # Find all benchmark files
    pattern = re.compile(r'benchmark_results_\d+\.json')
    benchmark_files = [f for f in Path(directory).glob('*.json') 
                      if pattern.match(f.name)]
    
    print(f"Found {len(benchmark_files)} benchmark files")
    
    for file_path in benchmark_files:
        try:
            # Read the metadata
            with open(file_path, 'r') as f:
                data = json.load(f)
                metadata = data['metadata']
            
            # Extract relevant information
            dim = metadata['vector_dim']
            k = metadata['k']
            mode = mode_shorts[metadata['mode']]
            
            # Create new filename
            new_name = f"benchmark_dim{dim}_k{k}_{mode}.json"
            new_path = file_path.parent / new_name
            
            # Check if target file already exists
            if new_path.exists():
                print(f"Warning: {new_name} already exists, skipping {file_path.name}")
                continue
            
            # Rename file
            file_path.rename(new_path)
            print(f"Renamed {file_path.name} to {new_name}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print("\nRenaming complete!")

# Example usage
if __name__ == "__main__":
    # Use current directory
    rename_benchmark_files('.')