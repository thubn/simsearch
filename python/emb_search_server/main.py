#!/usr/bin/env python3
from math import exp, log2
import signal
import sys
import socket
import subprocess
import os
from contextlib import asynccontextmanager
from typing import List, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from embedding_search_benchmark import EmbeddingSearch
from sentence_transformers import SentenceTransformer
import uvicorn
from pathlib import Path
import torch
import time

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

def find_and_kill_process_by_port(port):
    """Find and kill process using specified port"""
    try:
        # Try using lsof
        cmd = f"lsof -ti tcp:{port}"
        try:
            pid = subprocess.check_output(cmd, shell=True).decode().strip()
            if pid:
                print(f"Found process {pid} using port {port}")
                os.kill(int(pid), signal.SIGKILL)
                time.sleep(1)  # Give the system time to free the port
                return True
        except subprocess.CalledProcessError:
            pass

        # Try using netstat as fallback
        cmd = f"netstat -tunlp 2>/dev/null | grep :{port}"
        try:
            output = subprocess.check_output(cmd, shell=True).decode()
            if output:
                pid = output.split()[-1].split('/')[0]
                if pid:
                    print(f"Found process {pid} using port {port}")
                    os.kill(int(pid), signal.SIGKILL)
                    time.sleep(1)  # Give the system time to free the port
                    return True
        except (subprocess.CalledProcessError, IndexError):
            pass

        return False
    except Exception as e:
        print(f"Error killing process on port {port}: {e}")
        return False

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            s.close()
            return False
        except OSError:
            return True

# Global variables for model and searcher
model = None
searcher = None

def initialize_services():
    global model, searcher
    try:
        print("Initializing model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1', device=device)
        
        print("Initializing searcher...")
        searcher = EmbeddingSearch()
        
        print("Loading embeddings...")
        success = searcher.load(
            "../out/1_2M_random_out.parquet",
            embedding_dim=1024,
            init_pca=False,
            init_avx2=True,
            init_binary=True,
            init_int8=True,
            init_float16=False,
            init_mf=True
        )
        if not success:
            raise Exception("Failed to load embeddings")
        #searcher.unset_base()
        print("Embeddings loaded successfully")
        return True
    except Exception as e:
        print(f"Error during initialization: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize services
    if not initialize_services():
        raise Exception("Failed to initialize services")
    yield
    # Shutdown: Clean up resources
    global model, searcher
    model = None
    searcher = None

# Initialize FastAPI app with lifespan
app = FastAPI(title="Embedding Search Server", lifespan=lifespan)

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    rescoring_factor: int = 25

@app.post("/api/search")
async def search(request: SearchRequest):
    if model is None or searcher is None:
        raise HTTPException(status_code=503, detail="Server is still initializing")
    
    try:
        query_embedding = model.encode([request.query])[0]
        
        # Perform searches with different methods
        avx2_results, avx2_time = searcher.search_avx2(query_embedding, request.k)
        binary_results, binary_time = searcher.search_binary(query_embedding, request.k)
        int8_results, int8_time = searcher.search_int8(query_embedding, request.k)
        twostep_results, twostep_time = searcher.search_twostep(query_embedding, request.k, request.rescoring_factor)
        twostep_mf_results, twostep_mf_time = searcher.search_twostep_mf(query_embedding, request.k, request.rescoring_factor)
        
        # Get reference results for calculating metrics
        # float_results, _ = searcher.search_float(query_embedding, request.k)
        float_indices = set(idx for _, idx, _ in avx2_results)
        
        def calculate_metrics(results):
            result_indices = set(idx for _, idx, _ in results)
            return {
                'jaccard_index': len(float_indices & result_indices) / len(float_indices | result_indices),
                'ndcg' : calculate_ndcg(avx2_results, results),
                'overlap': len(float_indices & result_indices)
            }
        
        # Format results for each method
        def format_results(method_name, results, time_us, metrics=None):
            data = {
                'method': method_name,
                'time_us': time_us,
                'results': [{
                    'score': float(score),
                    'index': int(idx),
                    'text': searcher.get_sentence(int(idx)),
                } for score, idx, text in results]
            }
            if metrics:
                data.update(metrics)
            return data
        
        # Calculate metrics for each method
        binary_metrics = calculate_metrics(binary_results)
        int8_metrics = calculate_metrics(int8_results)
        twostep_metrics = calculate_metrics(twostep_results)
        twostep_mf_metrics = calculate_metrics(twostep_mf_results)
        
        return [
            format_results('AVX2', avx2_results, avx2_time),
            format_results('Binary AVX2', binary_results, binary_time, binary_metrics),
            format_results('Int8', int8_results, int8_time, int8_metrics),
            format_results('Two Step', twostep_results, twostep_time, twostep_metrics),
            format_results('Two Step MF', twostep_mf_results, twostep_mf_time, twostep_mf_metrics)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)

app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

def run_server(host: str = "0.0.0.0", port: int = 8100, max_retries: int = 3):
    retries = 0
    while retries < max_retries:
        if is_port_in_use(port):
            print(f"Port {port} is in use. Attempting to clean up (attempt {retries + 1}/{max_retries})...")
            if find_and_kill_process_by_port(port):
                time.sleep(1)
            else:
                print(f"Could not free port {port}")
                retries += 1
                continue
        
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            reload=False,
            workers=1,
            loop="asyncio",
            log_level="info",
            limit_max_requests=1000,
            timeout_keep_alive=30
        )
        
        server = uvicorn.Server(config)
        
        def handle_exit(signo, frame):
            print("\nShutting down gracefully...")
            find_and_kill_process_by_port(port)
            sys.exit(0)
        
        signal.signal(signal.SIGINT, handle_exit)
        signal.signal(signal.SIGTERM, handle_exit)
        
        try:
            server.run()
            break
        except Exception as e:
            print(f"Server error: {e}")
            find_and_kill_process_by_port(port)
            retries += 1
    
    if retries >= max_retries:
        print(f"Failed to start server after {max_retries} attempts")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Start the Embedding Search Server')
    parser.add_argument('--port', type=int, default=8080, help='Port number to use')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host address to bind to')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of retry attempts')
    
    args = parser.parse_args()
    
    def signal_handler(sig, frame):
        print('\nCleaning up and shutting down...')
        find_and_kill_process_by_port(args.port)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        run_server(args.host, args.port, args.max_retries)
    except KeyboardInterrupt:
        print('\nShutdown requested...')
        find_and_kill_process_by_port(args.port)
        sys.exit(0)
    except Exception as e:
        print(f'Error: {e}')
        find_and_kill_process_by_port(args.port)
        sys.exit(1)