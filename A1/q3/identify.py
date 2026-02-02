import argparse
import os
from utils import parse_graph

def remove_duplicate(graphs):
    unique_graphs = []
    seen = set()
    for graph in graphs:
        graph_tuple = tuple(sorted(graph))
        if graph_tuple not in seen:
            seen.add(graph_tuple)
            unique_graphs.append(graph)
    return unique_graphs

def gaston_format(graphs, output_file):
    with open(output_file, 'w') as f:
        for i, graph in enumerate(graphs):
            f.write(f'# 1\nt {i+1}\n')
            for item in graph:
                if len(item) == 2:
                    f.write(f'v {item[0]} {item[1]}\n')
                elif len(item) == 3:
                    f.write(f'e {item[0]} {item[1]} {item[2]}\n')

def run_gaston(gaston_input_file, gaston_output_file, total_graphs, support_threshold=5):
    import subprocess
    freq = max(1, int((support_threshold / 100) * total_graphs))
    gaston_path = os.path.join(os.path.dirname(__file__), 'gaston-1.1/gaston')

    if not os.path.exists(gaston_path):
        print(f"Warning: Gaston executable not found at {gaston_path}")
        print("Please ensure Gaston is compiled and available")
        return
    
    cmd = [gaston_path, str(freq), gaston_input_file, gaston_output_file]
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Gaston Failed: {result.stderr}")

def extract_topk(gaston_output_file, top_k, output_file):
    subgraphs = []
    with open(gaston_output_file, 'r') as f:
        current_subgraph = []
        current_support = 0
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                if current_subgraph:
                    subgraphs.append((current_support, current_subgraph))
                    current_subgraph = []
                    _, current_support = line.split()
            elif line.startswith('t'):
                continue
            else:
                current_subgraph.append(line)

        if current_subgraph:
            subgraphs.append((current_support, current_subgraph))

    print(f'Total subgraphs : {len(subgraphs)}')

    subgraphs.sort(key=lambda x: int(x[0]), reverse=True)
    with open(output_file, 'w') as f:
        id = 1
        for support, subgraph in subgraphs[:top_k]:
            f.write(f'# Support: {support}\n')
            f.write(f't {id}\n')
            id += 1
            for line in subgraph:
                f.write(f'{line}\n')
            f.write('\n')
        
def identify_discriminative_subgraphs(input_file, output_file):
    graphs = parse_graph(input_file)
    unique_graphs = remove_duplicate(graphs)
    unique_graph_path = input_file + ".unique"
    gaston_format(unique_graphs, unique_graph_path)
    gaston_output_path = unique_graph_path + ".gaston"
    run_gaston(unique_graph_path, gaston_output_path, len(unique_graphs), support_threshold=55)
    extract_topk(gaston_output_path, 50, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify and extract discriminative subgraphs.")
    parser.add_argument("input_file", help="Input file containing graphs and labels")
    parser.add_argument("output_file", help="Output file to write discriminative subgraphs")
    args = parser.parse_args()

    identify_discriminative_subgraphs(args.input_file, args.output_file)
