import matplotlib.pyplot as plt
import time
import os
import argparse
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import math
import traceback
from convert_dataset import convert_to_gaston

def run_experiments(gsp_path, fsg_path, gaston_path, data_path, out_dir):
    data_dir = Path(data_path).parent

    total_graph = convert_to_gaston(data_path, data_dir)
    with ThreadPoolExecutor() as executor:
        for support in support_thresholds:
            futures = {
                'GSP': executor.submit(run_gsp, gsp_path, data_dir / "gsp.txt", support, total_graph, out_dir),
                'FSG': executor.submit(run_fsg, fsg_path, data_dir / "fsg.txt", support, out_dir),
                'GASTON': executor.submit(run_gaston, gaston_path, data_dir / "gaston.txt", support, total_graph, out_dir),
            }

            gsp_time = fsg_time = gaston_time = math.nan

            for name, fut in futures.items():
                try:
                    t = fut.result()
                except Exception as e:
                    print(f"[Error] {name} failed for support={support}: {e}")
                    print(traceback.format_exc())
                    t = math.nan
                if name == 'GSP':
                    gsp_time = t
                    gsp_times.append(gsp_time)
                elif name == 'FSG':
                    fsg_time = t
                    fsg_times.append(fsg_time)
                else:
                    gaston_time = t
                    gaston_times.append(gaston_time)
            print(f"Support: {support} | GSP Time: {gsp_time:.4f}s | FSG Time: {fsg_time:.4f}s | Gaston Time: {gaston_time:.4f}s")

def run_gsp(bin_path, data_path, support, total_graph, out_dir):
    out_path = os.path.join(out_dir, f"gspan{support}")
    open(out_path, 'a').close()
    
    start_time = time.time()
    try: 
        result = subprocess.run(
            [
                bin_path,
                "-s", str(support / 100),
                "-f", data_path,
                "-o",
                "-i"
            ],
            timeout=3600,
            check=True
        )
        end_time = time.time()
        total_time = end_time - start_time
    except Exception as e:
        print(f"[Error]: {e}")
        total_time = 3600
    
    
    input_path = Path(data_path)
    
    base = input_path.stem

    generated_file = Path(f"{input_path}.fp")

    # Safety check
    if generated_file.exists():
        # copy contents to another file
        output_file = Path(out_path)

        with generated_file.open("rb") as src, output_file.open("wb") as dst:
            dst.write(src.read())
    else:
        print(f"File doesn't exist: {generated_file}")
    
    return total_time


def run_fsg(bin_path, data_path, support, out_dir):
    out_path = os.path.join(out_dir, f"fsg{support}")
    open(out_path, 'a').close()
    
    start_time = time.time()
    try: 
        result = subprocess.run([bin_path, "-s", str(support), data_path], timeout=3600, check=True)
        end_time = time.time()
        total_time = end_time - start_time
    except Exception as e:
        print(f"[Error]: {e}")
        total_time = 3600
    
    input_path = Path(data_path)

    new_path = input_path.with_suffix(".fp")

    # copy contents to another file
    output_file = Path(out_path)

    with new_path.open("rb") as src, output_file.open("wb") as dst:
        dst.write(src.read())
    
    return total_time

def run_gaston(bin_path, data_path, support, total_graph, out_dir):
    out_path = os.path.join(out_dir, f"gaston{support}")
    open(out_path, 'a').close()
    start_time = time.time()
    # print(f"Support for gaston: {int(total_graph * support / 100)}")
    try:
        result = subprocess.run([bin_path, f"{int(total_graph * support / 100)}", data_path, out_path], timeout=3600, check=True)
        end_time = time.time()
        total_time = end_time - start_time
    except Exception as e:
        print(f"[Error]: {e}")
        total_time = 3600
    
    return total_time

def plot_results(out_dir, support_thresholds, gsp_times, fsg_times, gaston_times):
        print(gsp_times)
        print(fsg_times)
        print(gaston_times)
        plt.figure(figsize=(10, 6))
        plt.plot(support_thresholds, gsp_times, marker='o', label='GSP', color='blue')
        plt.plot(support_thresholds, fsg_times, marker='o', label='FSG', color='orange')
        plt.plot(support_thresholds, gaston_times, marker='o', label='Gaston', color='green')
        plt.xlabel('Support Threshold (%)')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Support Threshold (in %) vs Run Time (in s)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, 'plot.png'))
        plt.close()

if __name__ == "__main__":
    arg = argparse.ArgumentParser("Comparing Frequent Subgraph Mining Algorithms")
    arg.add_argument("--gsp", type=str, required=True, help="Path to the gsp binary")
    arg.add_argument("--fsg", type=str, required=True, help="Path to the fsg binary")
    arg.add_argument("--gaston", type=str, required=True, help="Path to the gaston binary")
    arg.add_argument("--data", type=str, required=True, help="Path to the dataset file")
    arg.add_argument("--out", type=str, required=True, help="Path to output the comparison results")
    args = arg.parse_args()
    
    # Two lists to store execution times
    gsp_times = []
    fsg_times = []
    gaston_times = []
    # support_thresholds = [95, 50, 25, 10, 5]
    support_thresholds = [95, 90]
        
    shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(args.out, exist_ok=True)
    run_experiments(args.gsp, args.fsg, args.gaston, args.data, args.out)
    plot_results(args.out, support_thresholds[::-1], gsp_times[::-1], fsg_times[::-1], gaston_times[::-1])