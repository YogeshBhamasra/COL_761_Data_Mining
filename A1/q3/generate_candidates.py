import argparse
import numpy as np

def generate_candidates(database_features, query_features):
    candidates = []

    for query_feature in query_features:
        query_candidate = []

        for j, db_feature in enumerate(database_features):
            if np.all(query_feature <= db_feature):
                query_candidate.append(j)

        candidates.append(query_candidate)

    return candidates

def candidates_output_file(candidates, output_file):
    with open(output_file, 'w') as f:
        for q_idx, candidate_list in enumerate(candidates):
            f.write(f"q # {q_idx}\n")
            
            if candidate_list:
                candidate_str = ' '.join(map(str, candidate_list))
                f.write(f"c # {candidate_str}\n")
            else:
                f.write("c #\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_database_graph_features", type=str, help="Path to the database graph features file")
    parser.add_argument("path_query_graph_features", type=str, help="Path to the query graph features file")
    parser.add_argument("path_out_file", type=str, help="Path to the output file for candidates")
    args = parser.parse_args()

    path_database_graph_features = args.path_database_graph_features
    path_query_graph_features = args.path_query_graph_features
    path_out_file = args.path_out_file

    database_features = np.load(path_database_graph_features)
    query_features = np.load(path_query_graph_features)
    
    candidates = generate_candidates(database_features, query_features)
    candidates_output_file(candidates, path_out_file)

    
