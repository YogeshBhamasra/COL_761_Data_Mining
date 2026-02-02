def convert_to_gaston(input_file, output_dir):
    labels = {
        'Br': 0,
        'C': 1,
        'Cl': 2,
        'F': 3,
        'H': 4,
        'I': 5,
        'N': 6,
        'O': 7,
        'P': 8,
        'S': 9,
        'Si': 10
    }
    count = 11
    total_graph = 0
    with open(input_file, "r") as fin, open(output_dir / "gsp.txt", "w") as gsp_out, open(output_dir / "fsg.txt", "w") as fsg_out, open(output_dir / "gaston.txt", "w") as gaston_out:
        lines = [line.strip() for line in fin if line.strip()]
        i = 0

        gsp_out.write("# Converted to Gaston format\n")
        fsg_out.write("# Converted to Gaston format\n")
        gaston_out.write("# Converted to Gaston format\n")
        while i < len(lines):
            # graph id
            graph_id = lines[i].lstrip("#")
            i += 1
            total_graph += 1

            gsp_out.write(f"t # {graph_id}\n")
            fsg_out.write(f"t # {graph_id}\n")
            gaston_out.write(f"t # {graph_id}\n")

            # number of nodes
            num_nodes = int(lines[i])
            i += 1

            # node labels
            for node_id in range(num_nodes):
                label = lines[i]
                if label not in labels:
                    labels[label] = count
                    count += 1
                i += 1
                gsp_out.write(f"v {node_id} {labels[label]}\n")
                fsg_out.write(f"v {node_id} {labels[label]}\n")
                gaston_out.write(f"v {node_id} {labels[label]}\n")

            # number of edges
            num_edges = int(lines[i])
            i += 1
            # edges
            for _ in range(num_edges):
                src, dst, edge_label = lines[i].split()
                i += 1
                gsp_out.write(f"e {src.strip()} {dst.strip()} {edge_label.strip()}\n")
                fsg_out.write(f"e {src.strip()} {dst.strip()} {edge_label.strip()}\n")
                gaston_out.write(f"e {src.strip()} {dst.strip()} {edge_label.strip()}\n")
    return total_graph
if __name__ == "__main__":
    convert_to_gaston("Yeast/temp.txt_graph")