import argparse
import numpy as np
import json
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple
from math import log2, exp
from embedding_search_benchmark import EmbeddingSearch

# Import power measurement class from power.py
from power import N6705C

def calculate_ndcg(ground_truth: List[Tuple[float, int, str]], 
                  prediction: List[Tuple[float, int, str]]) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain between ground truth and prediction.
    Implements the same logic as the C++ version.
    """
    if not ground_truth or not prediction:
        return 0.0

    k = min(len(ground_truth), len(prediction))
    
    # Create position lookup for ground truth
    truth_positions = {idx: pos for pos, (_, idx, _) in enumerate(ground_truth[:k])}
    
    # Calculate DCG
    dcg = 0.0
    for i in range(k):
        pred_idx = prediction[i][1]
        if pred_idx in truth_positions:
            # Calculate relevance score based on position difference
            position_diff = abs(float(truth_positions[pred_idx] - i))
            relevance = exp(-position_diff / k)  # Exponential decay
            
            # DCG formula: rel_i / log2(i + 2)
            dcg += relevance / log2(i + 2)
    
    # Calculate IDCG (ideal DCG - when order is perfect)
    idcg = sum(1.0 / log2(i + 2) for i in range(k))
    
    return dcg / idcg if idcg > 0 else 0.0

class PowerMeasurement:
    def __init__(self, output_dir, measure_interval=0.01, measure_duration=None):
        self.power_meter = N6705C()
        self.output_dir = output_dir
        self.measure_interval = measure_interval
        self.measure_duration = measure_duration  # If None, will measure until stop_measurement is called
        self.measurements = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def start_measurement(self, method_name):
        """Start power measurement for a specific method"""
        print(f"Starting power measurement for method: {method_name}")
        
        # Turn on channel if not already on
        self.power_meter.ch0_on()
        
        # Start the measurement
        if self.measure_duration:
            power, current, voltage, interval = self.power_meter.ch0_measure(
                interval=self.measure_interval, 
                mtime=self.measure_duration
            )
            
            # Store the measurement data
            self.measurements[method_name] = {
                "power": power,
                "current": current,
                "voltage": voltage,
                "interval": interval,
                "timestamp": time.time()
            }
            
            # Save the measurement data
            self._save_measurement_data(method_name)
            
            return power, current, voltage, interval
        else:
            # In this case, we'll need to implement a separate way to capture data
            # This would require threading or a non-blocking approach
            pass
    
    def stop_measurement(self, method_name):
        """Stop the current measurement if using non-blocking approach"""
        # Would be implemented if using a non-blocking approach
        pass
    
    def _save_measurement_data(self, method_name):
        """Save the measurement data to disk"""
        if method_name not in self.measurements:
            print(f"No measurement data for method: {method_name}")
            return
        
        measurement = self.measurements[method_name]
        
        # Create a dataframe
        data = self.power_meter.to_dataframe(
            measurement["power"],
            measurement["current"],
            measurement["voltage"],
            measurement["interval"]
        )
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, f"{method_name}_power_data.csv")
        data.to_csv(csv_path, index=False)
        print(f"Saved power measurement data to {csv_path}")
        
        # Generate and save plot
        self._save_plot(method_name)
    
    def _save_plot(self, method_name):
        """Generate and save a plot of the measurement data"""
        if method_name not in self.measurements:
            return
        
        measurement = self.measurements[method_name]
        
        fig, axs = plt.subplots(3, figsize=(12, 9))
        fig.suptitle(f"Power Measurement for {method_name}")
        
        # Plot power
        power = measurement["power"]
        interval = measurement["interval"]
        times = np.multiply(range(0, len(power)), interval)
        
        total_energy = sum(power) * interval
        avg_power = sum(power) / len(power)
        
        axs[0].plot(times, power)
        axs[0].set_title(f"Power Consumption (Avg: {avg_power:.4f}W, Total: {total_energy:.4f}Wh [J])")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Power (W)")
        
        # Plot voltage
        voltage = measurement["voltage"]
        avg_voltage = sum(voltage) / len(voltage)
        
        axs[1].plot(times, voltage)
        axs[1].set_title(f"Voltage (Avg: {avg_voltage:.4f}V)")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Voltage (V)")
        
        # Plot current
        current = measurement["current"]
        avg_current = sum(current) / len(current)
        
        axs[2].plot(times, current)
        axs[2].set_title(f"Current (Avg: {avg_current:.4f}A)")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Current (A)")
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"{method_name}_power_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved power measurement plot to {plot_path}")
    
    def generate_summary(self):
        """Generate a summary of all measurements"""
        summary = {}
        
        for method_name, measurement in self.measurements.items():
            power = measurement["power"]
            interval = measurement["interval"]
            
            total_energy = sum(power) * interval
            avg_power = sum(power) / len(power)
            duration = len(power) * interval
            
            summary[method_name] = {
                "avg_power_watts": avg_power,
                "total_energy_joules": total_energy,
                "duration_seconds": duration,
                "energy_per_query": total_energy / duration if duration > 0 else 0
            }
        
        # Save summary to file
        summary_path = os.path.join(self.output_dir, "power_measurement_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved power measurement summary to {summary_path}")
        
        return summary
        
class VectorSearchBenchmarkWithPower:
    def __init__(self, embedding_file: str, k: int = 25, runs: int = 100, 
                 rescoring_factors: List[int] = None, embedding_dim = 1024,
                 power_output_dir = "power_results",
                 measure_duration = None):
        self.searcher = EmbeddingSearch()
        self.searcher.load(filename=embedding_file, embedding_dim=embedding_dim, 
                          init_pca=False, init_avx2=True, init_binary=True, 
                          init_int8=True, init_float16=False, init_mf=False)
        self.k = k
        self.runs = runs
        self.rescoring_factors = rescoring_factors or []
        self.num_vectors, self.vector_dim = self.searcher.get_dimensions()
        print(f"Loaded {self.num_vectors} vectors of dimension {self.vector_dim}")
        if rescoring_factors:
            print(f"Will run two-step search with rescoring factors: {rescoring_factors}")
        
        # Initialize power measurement
        self.power_measurement = PowerMeasurement(
            output_dir=power_output_dir,
            measure_duration=measure_duration
        )
        
        # Define search methods
        self.search_methods = [
            ("float", self.searcher.search_float),
            ("avx2", self.searcher.search_avx2),
            ("binary", self.searcher.search_binary),
            ("int8", self.searcher.search_int8),
            # ("float16", self.searcher.search_float16),
            # ("mf", self.searcher.search_mf),
            # ("pca2", self.searcher.search_pca2),
            # ("pca4", self.searcher.search_pca4),
            # ("pca8", self.searcher.search_pca8),
            # ("pca16", self.searcher.search_pca16),
            # ("pca32", self.searcher.search_pca32)
        ]
        
        # Add two-step searches for each rescoring factor
        if rescoring_factors:
            for factor in self.rescoring_factors:
                self.search_methods.append((
                    f"twostep_rf{factor}",
                    lambda q, k, rf=factor: self.searcher.search_twostep(q, k, rf)
                ))
            # for factor in self.rescoring_factors:
            #     self.search_methods.append((
            #         f"ts_mf_rf{factor}",
            #         lambda q, k, rf=factor: self.searcher.search_twostep_mf(q, k, rf)
            #     ))

    def _generate_queries(self, mode: str, query_file: str = None) -> List[Dict[str, Any]]:
        """Generate query vectors based on the selected mode"""
        queries = []
        
        if mode == "random":
            # Mode 1: Random indices from existing embeddings
            random_indices = np.random.randint(0, self.num_vectors, size=self.runs)
            for run in range(self.runs):
                query_vector = self.searcher.get_float_embedding(random_indices[run])
                queries.append({
                    "run": run,
                    "query_index": int(random_indices[run]),
                    "vector": query_vector
                })
                
        elif mode == "query":
            # Mode 2: Queries from JSONL file
            loaded_queries = self._load_queries(query_file)
            for i, query_data in enumerate(loaded_queries):
                query_vector = np.array(query_data["embedding"], dtype=np.float32)
                queries.append({
                    "run": i,
                    "query_text": query_data["query"],
                    "formatted_query": query_data["formatted_query"],
                    "vector": query_vector
                })
                
        elif mode == "random-vec":
            # Mode 3: Randomly generated normalized vectors
            for run in range(self.runs):
                # Generate random vector and normalize it
                query_vector = np.random.randn(self.vector_dim).astype(np.float32)
                query_vector = query_vector / np.linalg.norm(query_vector)
                queries.append({
                    "run": run,
                    "vector": query_vector
                })
                
        return queries

    def benchmark_sequential_with_power(self, mode: str, query_file: str = None) -> Dict[str, Any]:
        """Run benchmarks sequentially by method with power measurement"""
        # Generate all query vectors first
        print(f"Generating {self.runs} queries...")
        queries = self._generate_queries(mode, query_file)
        print(f"Generated {len(queries)} queries")
        
        results = []
        method_power_data = {}
        
        # Create a dummy run to warm up the system
        print("\nPerforming warm-up run...")
        warmup_vector = queries[0]["vector"]
        for method_name, search_func in self.search_methods:
            try:
                search_func(warmup_vector, self.k)
            except Exception as e:
                print(f"  Warning: Error in {method_name} warm-up: {str(e)}")
        
        # First run the float search (reference for comparison) on all queries
        print("\nRunning reference float search on all queries with power measurement...")
        float_results = []
        
        # Start power measurement for float search
        self.power_measurement.start_measurement("float")
        
        for i, query in enumerate(queries):
            query_vector = query["vector"]
            results_tup, time_us = self.searcher.search_float(query_vector, self.k)
            
            # Store query metadata + float results
            query_result = {
                "run": query["run"],
                "float_results": results_tup,
                "float_time_us": time_us,
                "searches": []
            }
            
            # Add query-specific metadata
            if "query_index" in query:
                query_result["query_index"] = query["query_index"]
            if "query_text" in query:
                query_result["query_text"] = query["query_text"]
                query_result["formatted_query"] = query["formatted_query"]
                
            float_results.append(query_result)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{len(queries)} float searches")
        
        # Add some separation between methods for power analysis
        print("Completed all float searches. Pausing for 5 seconds...")
        time.sleep(5)
                
        # Run each method separately on all queries
        for method_name, search_func in self.search_methods:
            # Skip float search as we already did it
            if method_name == "float":
                continue
                
            print(f"\nRunning {method_name} search on all queries with power measurement...")
            
            # Start power measurement for this method
            self.power_measurement.start_measurement(method_name)
            
            for i, query_result in enumerate(float_results):
                query_vector = queries[i]["vector"]
                float_indices = set(idx for _, idx, _ in query_result["float_results"])
                
                try:
                    # Run the current search method
                    results_tup, time_us = search_func(query_vector, self.k)
                    result_indices = set(idx for _, idx, _ in results_tup)
                    
                    # Calculate metrics compared to float search
                    metrics = {
                        "time_us": time_us,
                        "results": [(score, int(idx), text[:100]) for score, idx, text in results_tup[:5]],  # Store first 5 results
                        "overlap_with_float": len(float_indices & result_indices),
                        "jaccard_index": len(float_indices & result_indices) / len(float_indices | result_indices),
                        "ndcg": calculate_ndcg(query_result["float_results"], results_tup),
                        "ndcg_10": calculate_ndcg(query_result["float_results"][:10], results_tup[:10])
                    }
                    
                    query_result["searches"].append({
                        "method": method_name,
                        "metrics": metrics
                    })
                    
                except Exception as e:
                    print(f"  Error in {method_name} search (run {i}): {str(e)}")
                    continue
                
                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{len(queries)} {method_name} searches")
            
            # Add some separation between methods for power analysis
            print(f"Completed all {method_name} searches. Pausing for 5 seconds...")
            time.sleep(5)
        
        # Re-format results to match the original structure
        # Move "float_results" and "float_time_us" into the "searches" array
        for query_result in float_results:
            float_metrics = {
                "time_us": query_result.pop("float_time_us"),
                "results": [(score, int(idx), text[:100]) for score, idx, text in query_result.pop("float_results")[:5]]
            }
            
            query_result["searches"].insert(0, {
                "method": "float",
                "metrics": float_metrics
            })
        
        # Generate power measurement summary
        power_summary = self.power_measurement.generate_summary()
        
        return {
            "mode": mode,
            "results": float_results,
            "power_summary": power_summary
        }
        
    def _load_queries(self, query_file: str) -> List[Dict[str, Any]]:
        """Load queries from JSONL file"""
        queries = []
        with open(query_file, 'r') as f:
            for line in f:
                queries.append(json.loads(line))
        return queries

    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save benchmark results in a format optimized for Jupyter analysis"""
        timestamp = int(time.time())
        output_path = Path(output_file)
        final_output = output_path.with_stem(f"{output_path.stem}_{timestamp}")
        
        # Calculate aggregate statistics across all runs
        method_stats = {}
        for run in results['results']:
            for search in run['searches']:
                method_name = search['method']
                if method_name not in method_stats:
                    method_stats[method_name] = {
                        'times_us': [],
                        'jaccard_indices': [],
                        'ndcg_scores': [],
                        'overlap_counts': []
                    }
                
                metrics = search['metrics']
                method_stats[method_name]['times_us'].append(metrics['time_us'])
                
                if method_name != 'float':
                    method_stats[method_name]['jaccard_indices'].append(metrics['jaccard_index'])
                    method_stats[method_name]['ndcg_scores'].append(metrics['ndcg'])
                    method_stats[method_name]['overlap_counts'].append(metrics['overlap_with_float'])
        
        # Compute summary statistics with conversion to native Python types
        summary_stats = {}
        for method, stats in method_stats.items():
            summary_stats[method] = {
                'time_us': {
                    'mean': float(np.mean(stats['times_us'])),
                    'std': float(np.std(stats['times_us'])),
                    'min': float(np.min(stats['times_us'])),
                    'max': float(np.max(stats['times_us'])),
                    'median': float(np.median(stats['times_us']))
                }
            }
            
            if method != 'float':
                summary_stats[method].update({
                    'jaccard_index': {
                        'mean': float(np.mean(stats['jaccard_indices'])),
                        'std': float(np.std(stats['jaccard_indices']))
                    },
                    'ndcg': {
                        'mean': float(np.mean(stats['ndcg_scores'])),
                        'std': float(np.std(stats['ndcg_scores']))
                    },
                    'overlap': {
                        'mean': float(np.mean(stats['overlap_counts'])),
                        'std': float(np.std(stats['overlap_counts']))
                    }
                })

        # Ensure all numpy values in results are converted to native Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        # Add power summary to the results
        if 'power_summary' in results:
            power_summary = results['power_summary']
        else:
            power_summary = {}

        # Reorganize results for easier analysis
        analysis_ready = {
            'metadata': {
                'num_vectors': self.num_vectors,
                'vector_dim': self.vector_dim,
                'k': self.k,
                'runs': self.runs,
                'timestamp': timestamp,
                'timestamp_human': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)),
                'mode': results['mode'],
                'rescoring_factors': self.rescoring_factors if hasattr(self, 'rescoring_factors') else None
            },
            'summary': summary_stats,
            'method_stats': method_stats,
            'power_summary': power_summary,
            'runs': convert_numpy(results['results'])  # Convert any numpy types in the results
        }
        
        # Save main results
        with open(final_output, 'w') as f:
            json.dump(analysis_ready, f, indent=2)
        print(f"Results saved to {final_output}")
        
        # Save summary as separate file for quick reference
        summary_output = final_output.with_stem(f"{final_output.stem}_summary")
        with open(summary_output, 'w') as f:
            json.dump({
                'metadata': analysis_ready['metadata'],
                'summary': analysis_ready['summary'],
                'power_summary': power_summary
            }, f, indent=2)
        print(f"Summary saved to {summary_output}")

def main():
    parser = argparse.ArgumentParser(description="Vector Similarity Search Benchmark with Power Measurement")
    parser.add_argument("--embedding-file", "-f", required=True, help="Path to embedding file")
    parser.add_argument("--mode", "-m", choices=["random", "query", "random-vec"], required=True,
                      help="Benchmark mode: random indices, query file, or random vectors")
    parser.add_argument("--query-file", "-q", help="Path to query file (required for query mode)")
    parser.add_argument("--k", "-k", type=int, default=25, help="Number of results to retrieve")
    parser.add_argument("--runs", "-r", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--output", "-o", default="benchmark_results.json",
                      help="Output file path for results")
    parser.add_argument("--rescoring-factor", type=str, help="Comma-separated list of rescoring factors for two-step search")
    parser.add_argument("--embedding-dim", "-d", type=int, default=1024, help="Dimensions of embedding file")
    parser.add_argument("--method-pause", type=float, default=5.0, 
                      help="Pause between methods in seconds (for power analysis)")
    parser.add_argument("--power-output-dir", default="power_results",
                      help="Directory to save power measurement results")
    parser.add_argument("--measure-duration", type=float,
                      help="Duration of each power measurement in seconds (if not specified, measures entire method execution)")
    
    args = parser.parse_args()
    
    if args.mode == "query" and not args.query_file:
        parser.error("Query file is required for query mode")

    # Parse rescoring factors if provided
    rescoring_factors = None
    if args.rescoring_factor:
        try:
            rescoring_factors = [int(x) for x in args.rescoring_factor.split(',')]
        except ValueError:
            parser.error("Rescoring factors must be comma-separated integers")
    
    benchmark = VectorSearchBenchmarkWithPower(
        args.embedding_file, 
        args.k, 
        args.runs, 
        rescoring_factors, 
        embedding_dim=args.embedding_dim,
        power_output_dir=args.power_output_dir,
        measure_duration=args.measure_duration
    )
    
    # Run sequential benchmark with power measurement
    results = benchmark.benchmark_sequential_with_power(args.mode, args.query_file)
    
    # Save results
    benchmark.save_results(results, args.output)

if __name__ == "__main__":
    main()