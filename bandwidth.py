import numpy as np
import time

# Allocate a 256 MB array, larger than the CPU cache
a = np.ones(256 * 1024 * 1024 // 8)

best = 0.0

for _ in range(10):
    start = time.perf_counter()
    s = a.sum()
    elapsed = time.perf_counter() - start

    best = max(best, a.nbytes / elapsed / 1e9)

print(f"Approx. read bandwidth: {best:.1f} GB/s (single thread, NumPy)")
