import urllib . request
import json
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Let {xi} i=1 to n ⊂ Rd with d > 3 denote a given dataset. Consider the k-means objective
# min {Cj } j=1 to K
# j=1 to k xi ∈Cj
# ∥xi− µj ∥2
# ,
# where µj is the empirical mean of cluster Cj . Assume Euclidean distance.

def run_kmeans(data):
    silhouette_scores = []
    objective_values = []
    for k in range(1, 16):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        objective_values.append(kmeans.inertia_)
        if k>1:
            score = silhouette_score(data, labels)
            silhouette_scores.append(score)
            # print(f"K: {k}, Silhouette Score: {score}")
            
    return objective_values, silhouette_scores


def load_data(dataset_num):
    url = f"http://hulk.cse.iitd.ac.in:3000/dataset?student_id=mcs242456&dataset_num={dataset_num}"
    with urllib.request.urlopen(url) as response:
        raw_data = response.read().decode('utf-8')
        data = json.loads(raw_data)
    data = np.array(data["X"])
    # print(f"Data shape: {data.shape}")
    return data

def plot_data_points_multidimensional(data):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=50, alpha=0.7)
    plt.title('Data Points (PCA Projection)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    plt.savefig("data_points.png", dpi=300)
    # plt.show()
    
def plot_objective_dual(objective_values1, objective_values2):
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(2,1)
    ax[0].plot(range(1, 16), objective_values1, marker='o', color='blue')
    ax[0].set_title('K-means Objective Values vs K (Dataset 1)')
    ax[0].set_xlabel('Number of Clusters (K)')
    ax[0].set_yscale('log')
    ax[0].set_ylabel('Objective Value (Inertia)')
    ax[0].set_xticks(range(1, 16))
    ax[0].grid()
    
    ax[1].plot(range(1, 16), objective_values2, marker='o', color='orange')
    ax[1].set_title('K-means Objective Values vs K (Dataset 2)')
    ax[1].set_xlabel('Number of Clusters (K)')
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Objective Value (Inertia)')
    ax[1].set_xticks(range(1, 16))
    ax[1].grid()
    
    fig.suptitle('K-means Objective Values vs K for Two Datasets', fontsize=16)
    
    plt.tight_layout()
    plt.savefig("plot.png", dpi=300)
    

def plot_objective_values(objective_values):
    # apply log scale to objective values for better visualization
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 16), objective_values, marker='o')
    plt.title('K-means Objective Values vs K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Objective Value (Inertia)')
    plt.xticks(range(1, 16))
    plt.grid()
    plt.savefig("plot.png", dpi=300)
    # plt.show()
    
def find_optimal_k(silhouette_scores):
    optimal_k = np.argmax(silhouette_scores) + 2  # +2 because silhouette_scores starts from K=2
    print(f"Optimal K based on Silhouette Score: {optimal_k}")
    return optimal_k

def get_optimal_k_elbow(objective_values):
    # Simple elbow method: find the point where the decrease in objective value slows down
    deltas = np.diff(objective_values)
    second_deltas = np.diff(deltas)
    optimal_k = np.argmax(second_deltas[1:]) + 3  # +2 because objective_values starts from K=1
    # print(f"Optimal K based on Elbow Method: {optimal_k}")
    return optimal_k

def main():
    data = None
    if len(sys.argv) != 2:
        print("Usage: python Q1.py <dataset_num> or <path_to_npy_file>")
        sys.exit(1)
    arg = sys.argv[1]
    if ".npy" in arg:
        data = np.load(arg)
        objective_values, silhouette_scores = run_kmeans(data)
        plot_objective_values(objective_values)
        k_elbow = get_optimal_k_elbow(objective_values)
        print(f"Optimal K based on Elbow Method for dataset 1: {k_elbow}")
    elif arg.isdigit():
        data = load_data(1)
        print(f"Data shape: {data.shape}")
        obj1, _ = run_kmeans(data)
        k_elbow = get_optimal_k_elbow(obj1)
        print(f"Optimal K based on Elbow Method for dataset 1: {k_elbow}")
        data = load_data(2)
        print(f"Data shape: {data.shape}")
        obj2, _ = run_kmeans(data)
        plot_objective_dual(obj1, obj2)
        k_elbow = get_optimal_k_elbow(obj2)
        print(f"Optimal K based on Elbow Method for dataset 2: {k_elbow}")

        
    else:
        print("Invalid input: Either provide a .npy file or <dataset_num > as an integer.")
        sys.exit(1)
    
    # print("Objective Values:", objective_values)
    # print("Silhouette Scores:", silhouette_scores)
    
    # k = find_optimal_k(silhouette_scores)
    # print(f"Optimal K based on Silhouette Score: {k}")

if __name__ == "__main__":    main()