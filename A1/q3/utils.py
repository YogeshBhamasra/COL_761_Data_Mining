def parse_graph(input_file):
    graphs = []
    with open(input_file, 'r') as f:
        current_graph = []
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                if current_graph:
                    graphs.append(current_graph)
                    current_graph = []
            elif line.startswith('v'):
                parts = line.split()
                node_id = int(parts[1])
                node_label = int(parts[2])
                current_graph.append((node_id, node_label))
            elif line.startswith('e'):
                parts = line.split()
                # print(parts)
                src = int(parts[1])
                dst = int(parts[2])
                edge_label = int(parts[3])
                current_graph.append((src, dst, edge_label))
        if current_graph:
            graphs.append(current_graph)
    return graphs