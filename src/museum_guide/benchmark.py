"""Speed and size measures for the kiosk question: how long does one photo take on a small CPU?"""
import contextlib
import io
import pickle
import time

import numpy as np
import torch


@contextlib.contextmanager
def cpu_threads(n):
    """Run PyTorch with n CPU threads (a kiosk has few cores), then restore the previous setting."""
    old = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def latency_ms(predict_one, inputs, warmup=5, runs=50):
    """Time predict_one(x) for one input at a time, as on a kiosk: median and 95th percentile in ms."""
    for x in inputs[:warmup]:
        predict_one(x)
    times = []
    for k in range(runs):
        x = inputs[k % len(inputs)]
        t0 = time.perf_counter()
        predict_one(x)
        times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def size_mb(model):
    """Size of the saved model: state_dict for PyTorch, pickle for scikit-learn."""
    buf = io.BytesIO()
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), buf)
    else:
        pickle.dump(model, buf)
    return buf.tell() / 1e6
