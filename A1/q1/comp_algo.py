import matplotlib.pyplot as plt
import time
import os
import argparse
import subprocess
import shutil

# Two lists to store execution times
apriori_times = []
fpgrowth_times = []

# Thresholds as specified to test
support_thresholds = [5, 10, 25, 50, 90]

def run_experiments(apr_path, fp_path, data_path, out_dir, write_output):
    print("Running experiments for given support")
    for support in support_thresholds:
        apr_time = measure_runtime(apr_path, data_path, support, out_dir, "ap", write_output)
        fpg_time = measure_runtime(fp_path, data_path, support, out_dir, "fp", write_output)
        
        apriori_times.append(apr_time)
        fpgrowth_times.append(fpg_time)
        print(f"Support: {support} | Apriori Time: {apr_time:.4f}s | FP-Growth Time: {fpg_time:.4f}s")


# Given a binary measure its runtime
def measure_runtime(bin_path, data_path, support, out_dir, algo_type, write_output):
    # Create output file path
    out_path = os.path.join(out_dir, f"{algo_type}{support}")
    
    # Create the output file if it doesn't exist
    open(out_path, 'a').close()
    
    start_time = time.time()
    # Timeout throws runtime error if process doesn't finish within limits
    try:
        params = [bin_path, f"-s{support}", data_path]
        # Write output only if specified
        if write_output:
            params.append(out_path)
        # Timeout set to 1 hour
        subprocess.run(params, timeout=3600)
    except Exception as e:
        print(f"Error running {bin_path} with support {support}: {e}")
    end_time = time.time()
    # Return the time taken by the binary
    return end_time - start_time

def plot_results(out_dir):
    print("Plotting results")
    plt.figure(figsize=(10, 6))
    plt.plot(support_thresholds, apriori_times, marker='o', label='Apriori', color='blue')
    plt.plot(support_thresholds, fpgrowth_times, marker='o', label='FP-Growth', color='orange')
    plt.xlabel('Support Threshold (%)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Support Threshold (in %) vs Run Time (in s)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'plot.png'))
    plt.close()

def main():
    arg = argparse.ArgumentParser("Comparing Frequent Itemset Mining Algorithms")
    arg.add_argument("--ap", type=str, required=True, help="Path to the apriori binary")
    arg.add_argument("--fp", type=str, required=True, help="Path to the fpgrowth binary")
    arg.add_argument("--data", type=str, required=True, help="Path to the dataset file")
    arg.add_argument("--out", type=str, required=True, help="Path to output the comparison results")
    arg.add_argument("--write-output", type=str, help="Whether to write output files from algorithms")
    args = arg.parse_args()
    # Clear output directory if it exists
    shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(args.out, exist_ok=True)
    
    run_experiments(args.ap, args.fp, args.data, args.out, args.write_output=="true")
    plot_results(args.out)
    
if __name__ == "__main__":
    main()