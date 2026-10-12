"""
Benchmark script for evaluating:
1. AutoEncoder (AE)
2. AnomalyTransformer
3. CATCH
4. STAND
5. PCA
6. ExtraTrees

Metrics measured on PSM dataset:
- GPU VRAM: Peak training VRAM, Peak inference VRAM (MB)
- Host RAM: Peak training RAM, Peak inference RAM (MB)
- Training Time: Total time (s), Time per epoch (s/epoch)
- Inference Latency: Total inference time (s), Latency per sample (ms)
- Throughput: Training throughput (samples/s), Inference throughput (samples/s)
"""

import os
import sys
import time
import gc
import argparse
import pathlib
import psutil
import pandas as pd
import numpy as np
import torch

# Ensure project root in sys.path
proj_pth = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(proj_pth))

# Mock modules if not present in environment
from unittest.mock import MagicMock
if 'torchinfo' not in sys.modules:
    try:
        import torchinfo
    except ImportError:
        sys.modules['torchinfo'] = MagicMock()

# Imports for data and models
from src.data_utils.SimAD_data_loader2 import get_loader_segment, SupervisedDataset
from src.models.AE import AutoEncoder
from src.models.AnomalyTransformer import AnomalyTransformer
from src.models.supervised.STAND import STAND
from src.models.CATCH import CATCH
from src.models.PCA import PCA
from src.models.supervised.core_ml_wrap import ExtraTrees


def get_cpu_ram_mb():
    """Returns current process RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def get_gpu_peak_mb():
    """Returns peak GPU allocated memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def reset_memory():
    """Cleans up garbage and resets GPU peak memory statistics."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_model_instance(name, dim, win_size, epochs, batch_size, device='cuda'):
    """Instantiate model with suitable configuration."""
    if name == 'AE':
        # AutoEncoder
        return AutoEncoder(
            slidingWindow=win_size,
            hidden_neurons=[64, 32],
            batch_size=batch_size,
            epochs=epochs
        ), False
    elif name == 'AnomalyTransformer':
        return AnomalyTransformer(
            win_size=win_size,
            input_c=dim,
            batch_size=batch_size,
            num_epochs=epochs,
            lr=1e-4
        ), False
    elif name == 'CATCH':
        return CATCH(
            seq_len=win_size,
            batch_size=batch_size,
            num_epochs=epochs,
            lr=1e-4
        ), False
    elif name == 'STAND':
        return STAND(
            win_size=win_size,
            d_model=32,
            num_layers=1,
            bidirectional=False,
            epochs=epochs,
            batch_size=batch_size,
            lr=1e-3,
            device=device
        ), True
    elif name == 'PCA':
        return PCA(
            slidingWindow=30,
            n_components=None
        ), False
    elif name == 'ExtraTrees':
        return ExtraTrees(
            win_size=3,
            n_estimators=100
        ), True
    else:
        raise ValueError(f"Unknown model name: {name}")


def run_benchmark_for_model(name, train_y, train_labels, test_y, epochs, batch_size, win_size, device):
    print(f"\n==================================================")
    print(f"  Benchmarking: {name}")
    print(f"==================================================")
    
    reset_memory()
    ram_before_train = get_cpu_ram_mb()
    
    dim = train_y.shape[1]
    model, is_supervised = get_model_instance(name, dim, win_size, epochs, batch_size, device)
    
    # ---------------- 1. Training Phase ----------------
    print(f"[{name}] Starting training...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    t_train_start = time.time()
    
    if is_supervised:
        model.fit(train_y, train_labels)
    else:
        model.fit(train_y)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_train_total = time.time() - t_train_start
    
    train_peak_vram = get_gpu_peak_mb()
    train_peak_ram = get_cpu_ram_mb()
    train_ram_delta = max(0.0, train_peak_ram - ram_before_train)
    
    # Epoch time calculation
    is_dl = name in ['AE', 'AnomalyTransformer', 'CATCH', 'STAND']
    if is_dl and epochs > 0:
        time_per_epoch = t_train_total / epochs
        train_throughput = (len(train_y) * epochs) / t_train_total
    else:
        time_per_epoch = None
        train_throughput = len(train_y) / t_train_total

    print(f"[{name}] Training finished in {t_train_total:.2f}s "
          f"({f'{time_per_epoch:.2f}s/epoch' if time_per_epoch else 'N/A'}), "
          f"Train Peak VRAM: {train_peak_vram:.1f} MB, RAM Delta: {train_ram_delta:.1f} MB")

    # ---------------- 2. Inference Phase ----------------
    print(f"[{name}] Starting inference on {len(test_y)} samples...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    ram_before_infer = get_cpu_ram_mb()
    
    t_infer_start = time.time()
    scores = model.decision_function(test_y)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_infer_total = time.time() - t_infer_start
    
    infer_peak_vram = get_gpu_peak_mb()
    infer_peak_ram = get_cpu_ram_mb()
    infer_ram_delta = max(0.0, infer_peak_ram - ram_before_infer)
    
    n_samples = len(test_y)
    latency_per_sample_ms = (t_infer_total / n_samples) * 1000.0
    infer_throughput = n_samples / t_infer_total
    
    print(f"[{name}] Inference finished in {t_infer_total:.2f}s, "
          f"Latency: {latency_per_sample_ms:.4f} ms/sample, "
          f"Throughput: {infer_throughput:.1f} samples/s, "
          f"Infer Peak VRAM: {infer_peak_vram:.1f} MB")

    # Clean up model
    del model
    del scores
    reset_memory()

    return {
        "Model": name,
        "Type": "Deep Learning (GPU)" if is_dl else "Machine Learning (CPU)",
        "Train Peak VRAM (MB)": round(train_peak_vram, 2) if is_dl else 0.0,
        "Infer Peak VRAM (MB)": round(infer_peak_vram, 2) if is_dl else 0.0,
        "Train RAM Delta (MB)": round(train_ram_delta, 2),
        "Infer RAM Delta (MB)": round(infer_ram_delta, 2),
        "Train Time / Epoch (s)": round(time_per_epoch, 3) if time_per_epoch is not None else "N/A",
        "Total Train Time (s)": round(t_train_total, 2),
        "Train Throughput (samples/s)": round(train_throughput, 1),
        "Infer Total Time (s)": round(t_infer_total, 2),
        "Infer Latency (ms/sample)": round(latency_per_sample_ms, 4),
        "Infer Throughput (samples/s)": round(infer_throughput, 1)
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark AE, AnomalyTransformer, CATCH, STAND, PCA, ExtraTrees on PSM")
    parser.add_argument("--dataset_path", type=str, default="/home/pcl/zzj/projects/FTSAD/datasets", help="Path to dataset root")
    parser.add_argument("--dataset_name", type=str, default="PSM", help="Dataset name")
    parser.add_argument("--win_size", type=int, default=100, help="Sliding window size")
    parser.add_argument("--step_size", type=int, default=1, help="Step size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs for DL models")
    parser.add_argument("--train_test_split", type=float, default=0.5, help="Train-test split ratio")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--models", nargs="+", default=["STAND", "AE", "AnomalyTransformer", "CATCH", "PCA", "ExtraTrees"],
                        help="List of models to benchmark")
    parser.add_argument("--save_csv", type=str, default="tests/benchmark_results_psm.csv", help="CSV save path")
    parser.add_argument("--quick_test", action="store_true", help="Run on small slice for verification")
    args = parser.parse_args()

    print("==================================================")
    print("           PSM Benchmark Configuration            ")
    print("==================================================")
    print(f"Dataset Path     : {args.dataset_path}/{args.dataset_name}")
    print(f"Models to test   : {args.models}")
    print(f"Epochs (DL)      : {args.epochs}")
    print(f"Batch Size       : {args.batch_size}")
    print(f"Window Size      : {args.win_size}")
    print(f"Device           : {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Quick Test Mode  : {args.quick_test}")
    print("==================================================")

    # 1. Load Dataset
    print("\nLoading PSM dataset...")
    dataset_ = get_loader_segment(
        0, 
        f"{args.dataset_path}/{args.dataset_name}", 
        args.batch_size, 
        args.win_size, 
        args.step_size, 
        'train', 
        args.dataset_name, 
        0, 
        True
    )

    train_dataset = SupervisedDataset(
        dataset_.test, 
        dataset_.test_labels, 
        args.win_size, 
        args.step_size, 
        'train', 
        args.train_test_split
    )

    train_y = train_dataset.train_y
    train_labels = train_dataset.train_labels
    test_y = train_dataset.all_test_y
    test_labels = train_dataset.all_test_labels

    if args.quick_test:
        train_y = train_y[:2000]
        train_labels = train_labels[:2000]
        test_y = test_y[:1000]
        test_labels = test_labels[:1000]
        print(f"[Quick Test] Trimmed train_y: {train_y.shape}, test_y: {test_y.shape}")
    else:
        print(f"Full PSM Dataset loaded:")
        print(f"  - Train data   : {train_y.shape} (labels: {train_labels.shape})")
        print(f"  - Test data    : {test_y.shape} (labels: {test_labels.shape})")

    # 2. Benchmark each model
    results = []
    for model_name in args.models:
        try:
            res = run_benchmark_for_model(
                name=model_name,
                train_y=train_y,
                train_labels=train_labels,
                test_y=test_y,
                epochs=args.epochs,
                batch_size=args.batch_size,
                win_size=args.win_size,
                device=args.device
            )
            results.append(res)
        except Exception as e:
            print(f"[ERROR] Failed benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # 3. Print Summary Table
    df = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print("                             BENCHMARK RESULTS SUMMARY")
    print("=" * 90)
    print(df.to_markdown(index=False))
    print("=" * 90)

    # Save to CSV
    save_path = pathlib.Path(args.save_csv).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
