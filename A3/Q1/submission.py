import numpy as np
import faiss

def solve(base_vectors, query_vectors, k, K, time_budget):
    
    # Convert to float32 (Faiss expects float32)
    base = base_vectors.astype('float32')
    queries = query_vectors.astype('float32')
    N, d = base.shape
    Q = queries.shape[0]
    is_gpu = False  # Set to True if GPU is available and desired
    
    # Use all CPU cores and GPU if available
    try:
        n_threads = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(n_threads)
        ngpus = faiss.get_num_gpus()
        print(f"Using {n_threads} CPU threads. GPUs available: {ngpus}")
        if ngpus > 0: is_gpu = True
    except AttributeError:
        n_threads = 1  # fallback if not available
    
    # Choose index based on size: use HNSW if memory permits (fast, high recall)
    # otherwise IVF-PQ to save memory. For simplicity, we use HNSW for N < 2B, which should be fine for 16GB RAM with float32 vectors
    if is_gpu:
        index = faiss.IndexFlatL2(d)
        index = faiss.index_cpu_to_all_gpus(index)
    else:
        if N <= 2_000_000_000:
            # HNSW configuration
            M = 32 # neighbors per node
            index = faiss.IndexHNSWFlat(d, M)
            efSearch = 256  # search effort
            index.hnsw.efSearch = efSearch
            # No training needed
        else:
            # For very large N
            nlist = int(16 * np.sqrt(N))
            quant = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFPQ(quant, d, nlist, 64, 8)  # 64-byte PQ, 8-bit subquantizer
            index.nprobe = 32  # number of cells to search
            # Train IVF/PQ on a sample of base vectors
            np.random.seed(123)
            sample_size = min(N, nlist * 30)  # 30*Nlist sample heuristically
            train_idx = np.random.choice(N, sample_size, replace=False)
            index.train(base[train_idx])
    
    
    index.add(base)
    # build_time = time.time() - start_time
    
    # Process all queries in one batch
    D, I = index.search(queries, k)  # I: (Q,k) array of neighbor indices
    
    # Aggregate frequencies of base indices
    flat_neighbors = I.reshape(-1)
    freq = np.bincount(flat_neighbors, minlength=N)
    
    # Select top-K indices by frequency (ties by smaller index automatically)
    if K < N:
        # argpartition for efficiency
        topk_idx = np.argpartition(-freq, K)[:K]
        # Sort these top-K by (freq desc, idx asc)
        topk_idx = topk_idx[np.argsort((-freq[topk_idx], topk_idx), kind='stable')]
    else:
        # If K >= N, just sort all
        topk_idx = np.argsort((-freq, np.arange(N)), kind='stable')
    
    result = topk_idx[:K]
    return result

