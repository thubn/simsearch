import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional, Iterator, Tuple, Dict
import numpy as np
from tqdm.auto import tqdm
import gc
import os
import torch
import json
from datetime import datetime
import random


class ParquetEmbeddingGenerator:
    def __init__(
        self,
        model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
        batch_size: int = 32,
        device: Optional[str] = None,
        max_length: int = 4096
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            batch_size: Number of texts to process at once
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detection)
            max_length: Maximum length of formatted text in symbols
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def truncate_text(self, text: str, max_length: int) -> str:
        """
        Truncate text to specified length while trying to preserve word boundaries.
        
        Args:
            text: Text to truncate
            max_length: Maximum length in characters
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
            
        # Try to find a space to break at
        space_index = text.rfind(' ', 0, max_length)
        if space_index != -1:
            return text[:space_index] + "..."
        
        # If no space found, do hard truncation
        return text[:max_length-3] + "..."

    
    def format_text(self, title: str, text: str) -> str:
        """
        Format title and text according to the specified template,
        ensuring the total length doesn't exceed max_length.
        
        Args:
            title: The title of the document
            text: The main text content
            
        Returns:
            Formatted string combining title and text, limited to max_length
        """
        # Calculate lengths and template overhead
        template_overhead = len("title:  text: ")  # Space for format strings
        available_length = self.max_length - template_overhead
        
        if available_length <= 0:
            raise ValueError(f"Max length {self.max_length} is too small for the template overhead")
            
        # Calculate target lengths for title and text
        # Allocate 20% to title and 80% to text, but ensure title has at least 100 chars if available
        min_title_length = min(100, available_length // 4)
        target_title_length = max(min_title_length, int(available_length * 0.2))
        target_text_length = available_length - target_title_length
        
        # Truncate title and text if needed
        truncated_title = self.truncate_text(title, target_title_length)
        truncated_text = self.truncate_text(text, target_text_length)
        
        # Combine and format
        formatted = f"title: {truncated_title} text: {truncated_text}"
        
        # Final safety check
        if len(formatted) > self.max_length:
            formatted = formatted[:self.max_length-3] + "..."
            
        return formatted

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        """
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
    def generate_queries(self, texts: List[str]) -> np.ndarray:
        """
        Generate queries for a list of texts.
        """
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            prompt_name="query"
        )

    def iterate_parquet_file(
        self,
        file_path: str,
        chunk_size: int,
        columns: List[str],
        start_row: int = 0,
        max_rows: Optional[int] = None,
        dataset_config: Optional[str] = None,
        dataset_split: str = "train",
        random_rows: Optional[int] = None,
        random_seed: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        if random_seed is not None:
            random.seed(random_seed)
        
        if file_path.endswith('.parquet'):
            # Get total number of rows
            parquet_file = pq.ParquetFile(file_path)
            total_rows = parquet_file.metadata.num_rows
            
            if random_rows:
                # Generate random row indices
                selected_rows = random.sample(range(total_rows), min(random_rows, total_rows))
                selected_rows.sort()  # Sort for sequential reading
                
                # Read chunks of random rows
                current_index = 0
                while current_index < len(selected_rows):
                    # Get indices for current chunk
                    chunk_indices = selected_rows[current_index:current_index + chunk_size]
                    
                    # Read selected rows
                    df_chunk = pd.concat([
                        pd.read_parquet(file_path, columns=columns, rows=slice(idx, idx + 1))
                        for idx in chunk_indices
                    ])
                    
                    yield df_chunk
                    current_index += chunk_size
            else:
                # Original sequential reading logic for parquet files
                if max_rows is not None:
                    total_rows = min(total_rows - start_row, max_rows)
                else:
                    total_rows = total_rows - start_row
                
                for offset in range(0, total_rows, chunk_size):
                    current_chunk_size = min(chunk_size, total_rows - offset)
                    df_chunk = pd.read_parquet(
                        file_path,
                        columns=columns,
                        rows=slice(start_row + offset, start_row + offset + current_chunk_size)
                    )
                    yield df_chunk
        else:
            try:
                # Load Hugging Face dataset with configuration
                if dataset_config:
                    print(f"Loading dataset '{file_path}' with configuration '{dataset_config}'")
                    dataset = load_dataset(
                        file_path,
                        dataset_config,
                        split=dataset_split
                    )
                else:
                    print(f"Loading dataset '{file_path}'")
                    dataset = load_dataset(
                        file_path,
                        split=dataset_split
                    )

                total_rows = len(dataset)
                
                if random_rows:
                    # Generate random indices
                    selected_rows = random.sample(range(total_rows), min(random_rows, total_rows))
                    selected_rows.sort()  # Sort for sequential reading
                    
                    # Process in chunks
                    buffer = []
                    for idx in selected_rows:
                        data_row = dataset[idx]
                        
                        # Check if required columns exist
                        row_data = {}
                        missing_columns = False
                        for col in columns:
                            if col not in data_row:
                                print(f"Warning: Column '{col}' not found in dataset. Available columns: {list(data_row.keys())}")
                                missing_columns = True
                                break
                            row_data[col] = data_row[col]
                        
                        if missing_columns:
                            continue
                        
                        buffer.append(row_data)
                        
                        if len(buffer) >= chunk_size:
                            yield pd.DataFrame(buffer)
                            buffer = []
                    
                    if buffer:
                        yield pd.DataFrame(buffer)
                        
                else:
                    # Original sequential processing
                    buffer = []
                    processed_rows = 0
                    
                    for idx in range(total_rows):
                        if max_rows and processed_rows >= max_rows:
                            break
                            
                        if processed_rows < start_row:
                            processed_rows += 1
                            continue
                        
                        data_row = dataset[idx]
                        
                        # Check if required columns exist
                        row_data = {}
                        missing_columns = False
                        for col in columns:
                            if col not in data_row:
                                print(f"Warning: Column '{col}' not found in dataset. Available columns: {list(data_row.keys())}")
                                missing_columns = True
                                break
                            row_data[col] = data_row[col]
                        
                        if missing_columns:
                            continue
                        
                        buffer.append(row_data)
                        processed_rows += 1
                        
                        if len(buffer) >= chunk_size:
                            yield pd.DataFrame(buffer)
                            buffer = []
                    
                    if buffer:
                        yield pd.DataFrame(buffer)
                        
            except Exception as e:
                raise ValueError(f"Error loading dataset: {str(e)}")


    def process_chunks_to_parquet(
        self,
        file_path: str,
        output_path: str,
        title_column: str = "title",
        text_column: str = "text",
        chunk_size: int = 1000,
        max_rows: Optional[int] = None,
        start_row: int = 0,
        dataset_config: Optional[str] = None,
        dataset_split: str = "train",
        random_rows: int = None,
        random_seed: int = None
    ) -> None:
        """
        Process the dataset in chunks and save formatted text with embeddings to parquet files.
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        writer = None
        total_processed = 0
        
        try:
            # Iterate through chunks
            for chunk in tqdm(
                self.iterate_parquet_file(
                    file_path=file_path,
                    chunk_size=chunk_size,
                    columns=[title_column, text_column],
                    start_row=start_row,
                    max_rows=max_rows,
                    dataset_config=dataset_config,
                    dataset_split=dataset_split,
                    random_rows=random_rows,
                    random_seed=random_seed,
                ),
                desc="Processing chunks"
            ):
                # Format texts with length limiting
                formatted_texts = [
                    self.format_text(title, text)
                    for title, text in zip(chunk[title_column], chunk[text_column])
                ]
                
                # Generate embeddings
                chunk_embeddings = self.generate_embeddings(formatted_texts)
                
                # Create DataFrame with formatted text and embeddings
                result_data = {
                    'formatted_text': formatted_texts
                }
                
                # Add embedding columns
                for i in range(chunk_embeddings.shape[1]):
                    result_data[f'embedding_{i}'] = chunk_embeddings[:, i]
                
                result_chunk = pd.DataFrame(result_data)
                
                # Convert to PyArrow Table
                table = pa.Table.from_pandas(result_chunk)
                
                # Create writer if it doesn't exist
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                
                # Write chunk
                writer.write_table(table)
                
                # Update progress
                total_processed += len(chunk)
                print(f"\nProcessed {total_processed} rows...")
                
                # Clear memory
                del formatted_texts, chunk_embeddings, result_chunk
                gc.collect()
                
        except Exception as e:
            raise Exception(f"Error processing chunks: {str(e)}")


    def process_parquet_file(
        self,
        file_path: str,
        output_path: str,
        title_column: str = "title",
        text_column: str = "text",
        chunk_size: int = 1000,
        max_rows: Optional[int] = None,
        start_row: int = 0,
        dataset_config: Optional[str] = None,
        dataset_split: str = "train",
        random_rows: int = None,
        random_seed: int = None
    ) -> None:
        """
        Process a large parquet file or Hugging Face dataset memory-efficiently.
        
        Args:
            file_path: Path to the parquet file or Hugging Face dataset
            output_path: Path where to save the processed chunks
            title_column: Name of the title column
            text_column: Name of the text column
            chunk_size: Number of rows to process at once
            max_rows: Maximum number of rows to process
            start_row: Starting row index
            dataset_config: Configuration/subset name for Hugging Face dataset
            dataset_split: Dataset split to use (train, test, validation)
        """
        print(f"Starting processing with chunk size: {chunk_size}")
        print(f"Output will be saved to: {output_path}")
        
        self.process_chunks_to_parquet(
            file_path=file_path,
            output_path=output_path,
            title_column=title_column,
            text_column=text_column,
            chunk_size=chunk_size,
            max_rows=max_rows,
            start_row=start_row,
            dataset_config=dataset_config,
            dataset_split=dataset_split,
            random_rows=random_rows,
            random_seed=random_seed
        )
        
        print("Processing completed!")
        
    def interactive_query_embeddings(
        self,
        output_file: Optional[str] = None,
        prefix_text: str = ""
    ) -> None:
        """
        Interactive command-line interface for generating embeddings from user queries.
        Saves queries and their embeddings to a JSONL file.
        
        Args:
            output_file: Path to save the JSONL file. If None, generates a timestamp-based filename
            prefix_text: Optional text to prefix each query (e.g., "title: " or "query: ")
        """
        # Generate default output filename if none provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"query_embeddings/query_embeddings_{timestamp}.jsonl"
        
        print("\nInteractive Query Mode")
        print("Enter your queries one at a time. Press Enter after each query.")
        print("To finish, type 'quit', 'exit', or press Ctrl+C\n")
        
        try:
            queries = []
            while True:
                # Get user input
                query = input("Enter query (or 'quit' to finish): ").strip()
                
                # Check for exit commands
                if query.lower() in ['quit', 'exit', '']:
                    if not queries:  # If no queries were entered
                        print("No queries entered. Exiting without saving.")
                        return
                    break
                
                # Add prefix if specified
                formatted_query = f"{prefix_text}{query}" if prefix_text else query
                queries.append(formatted_query)
                
                # Generate embedding for the current query
                embedding = self.generate_embeddings([formatted_query])[0]
                
                # Create dictionary for JSONL entry
                entry = {
                    'query': query,
                    'formatted_query': formatted_query,
                    'embedding': embedding.tolist()
                }
                
                # Append to JSONL file
                with open(output_file, 'a') as f:
                    json.dump(entry, f)
                    f.write('\n')
                
                print(f"Embedding generated and saved. Vector dimension: {len(embedding)}")
                
        except KeyboardInterrupt:
            print("\nProcess interrupted by user.")
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
        finally:
            if queries:
                print(f"\nAll embeddings have been saved to: {output_file}")
                print(f"Total queries processed: {len(queries)}")
    
    def load_query_embeddings(self, jsonl_file: str) -> List[Dict]:
        """
        Load previously saved query embeddings from a JSONL file.
        
        Args:
            jsonl_file: Path to the JSONL file containing saved embeddings
            
        Returns:
            List of dictionaries containing queries and their embeddings
        """
        queries = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                queries.append(json.loads(line.strip()))
        return queries
    
    def process_json_queries(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        prefix_text: str = ""
    ) -> None:
        """
        Process queries from a JSON file and generate embeddings.
        Saves queries and their embeddings to a JSONL file.
        
        Args:
            input_file: Path to JSON file containing an array of queries under 'queries' key
            output_file: Path to save the JSONL file. If None, generates a timestamp-based filename
            prefix_text: Optional text to prefix each query (e.g., "title: " or "query: ")
        """
        # Generate default output filename if none provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"query_embeddings/batch_embeddings_{timestamp}.jsonl"
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            # Load queries from JSON file
            with open(input_file, 'r') as f:
                data = json.load(f)
                
            if not isinstance(data, dict) or 'queries' not in data:
                raise ValueError("Input JSON must contain a 'queries' array")
                
            queries = data['queries']
            if not isinstance(queries, list):
                raise ValueError("The 'queries' field must be an array")
                
            print(f"\nProcessing {len(queries)} queries from {input_file}")
            
            # Process queries in batches
            formatted_queries = [
                f"{prefix_text}{query}" if prefix_text else query
                for query in queries
            ]
            
            # Generate embeddings for all queries
            embeddings = self.generate_queries(formatted_queries)
            
            # Save results to JSONL file
            with open(output_file, 'w') as f:
                for query, formatted_query, embedding in zip(queries, formatted_queries, embeddings):
                    entry = {
                        'query': query,
                        'formatted_query': formatted_query,
                        'embedding': embedding.tolist()
                    }
                    json.dump(entry, f)
                    f.write('\n')
            
            print(f"\nProcessing completed!")
            print(f"Processed {len(queries)} queries")
            print(f"Results saved to: {output_file}")
            print(f"Embedding dimension: {len(embeddings[0])}")
            
        except FileNotFoundError:
            print(f"Error: Input file {input_file} not found")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {input_file}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

def main():
    """
    Command-line interface for generating query embeddings in both interactive and batch modes.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate embeddings for queries')
    parser.add_argument('--model', default='mixedbread-ai/mxbai-embed-large-v1',
                       help='Name of the sentence-transformers model to use')
    parser.add_argument('--output', default=None,
                       help='Output JSONL file path')
    parser.add_argument('--prefix', default="Represent this sentence for searching relevant passages: ",
                       help='Text to prefix each query')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for embedding generation')
    parser.add_argument('--mode', choices=['interactive', 'batch'], default='interactive',
                       help='Operation mode: interactive for command-line input, batch for processing JSON file')
    parser.add_argument('--input', default=None,
                       help='Input JSON file path (required for batch mode)')
    
    args = parser.parse_args()
    
    # Initialize the generator
    generator = ParquetEmbeddingGenerator(
        model_name=args.model,
        batch_size=args.batch_size
    )
    
    # Check mode and required arguments
    if args.mode == 'batch':
        if not args.input:
            parser.error("Batch mode requires --input argument")
        generator.process_json_queries(
            input_file=args.input,
            output_file=args.output,
            prefix_text=args.prefix
        )
    else:  # interactive mode
        generator.interactive_query_embeddings(
            output_file=args.output,
            prefix_text=args.prefix
        )

if __name__ == "__main__":
    main()