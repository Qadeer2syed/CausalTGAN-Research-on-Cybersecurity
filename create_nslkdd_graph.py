"""
Create Causal Graph for NSL-KDD using the 10 Causal Features
Identified by Algorithm 1 (CRFS) from Zeng et al. 2021

Causal Features (ranked by effect):
1. service (1.00)
2. dst_bytes (0.81)
3. src_bytes (0.75)
4. diff_srv_rate (0.73)
5. dst_host_srv_count (0.69)
6. dst_host_same_srv_rate (0.67)
7. dst_host_diff_srv_rate (0.65)
8. dst_host_serror_rate (0.62)
9. count (0.56)
10. flag (0.20)
+ label (target)

Causal Structure based on Cybersecurity Domain Knowledge:
- Service type determines connection characteristics
- Connection counts determine rate statistics
- Local patterns influence destination host patterns
- All contribute to attack detection
"""

import pickle
import networkx as nx
import os


def create_nslkdd_causal_graph():
    """
    Create causal graph for NSL-KDD using the 10 causal features

    Causal Logic:
    1. service → determines what kind of traffic (HTTP, FTP, etc.)
    2. service + flag → determines data transfer patterns (src_bytes, dst_bytes)
    3. count → connection frequency affects rates
    4. Local patterns → affect destination host statistics
    """

    causal_graph = [
        # Layer 1: Service (Root - most influential, effect=1.0)
        ['service', []],

        # Layer 2: Connection State (depends on service)
        ['flag', ['service']],

        # Layer 3: Data Transfer (depends on service and state)
        ['src_bytes', ['service', 'flag']],
        ['dst_bytes', ['service', 'flag', 'src_bytes']],

        # Layer 4: Connection Counting
        ['count', ['service']],

        # Layer 5: Service Rate (depends on count and service)
        ['diff_srv_rate', ['service', 'count']],

        # Layer 6: Destination Host Statistics
        # These are aggregated statistics at the destination level
        ['dst_host_srv_count', ['service', 'count']],
        ['dst_host_same_srv_rate', ['service', 'dst_host_srv_count']],
        ['dst_host_diff_srv_rate', ['service', 'diff_srv_rate', 'dst_host_srv_count']],
        ['dst_host_serror_rate', ['flag', 'service']],

        # Layer 7: Attack Label (depends on key indicators)
        ['label', [
            'service',
            'flag',
            'src_bytes',
            'dst_bytes',
            'count',
            'diff_srv_rate',
            'dst_host_srv_count',
            'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate',
            'dst_host_serror_rate'
        ]]
    ]

    return causal_graph


def visualize_graph(graph):
    """Visualize and validate the causal graph"""
    G = nx.DiGraph()

    # Add edges
    for node, parents in graph:
        for parent in parents:
            G.add_edge(parent, node)

    print("\n" + "="*70)
    print("CAUSAL GRAPH STATISTICS")
    print("="*70)
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    # Validate DAG
    if nx.is_directed_acyclic_graph(G):
        print("[OK] Valid DAG (no cycles)")
    else:
        print("[ERROR] Graph contains cycles!")
        cycles = list(nx.simple_cycles(G))
        print(f"Cycles: {cycles}")
        return None

    # Topological order
    topo_order = list(nx.topological_sort(G))
    print(f"\nTopological Order (generation sequence):")
    for i, node in enumerate(topo_order, 1):
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        print(f"  {i:2d}. {node:30s} (parents: {in_deg}, children: {out_deg})")

    # Root and leaf nodes
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]

    print(f"\nRoot nodes (no parents): {roots}")
    print(f"Leaf nodes (no children): {leaves}")

    return G


def save_graph(graph, output_dir, filename="graph.txt"):
    """Save causal graph in pickle format"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(graph, f)

    print(f"\n[OK] Saved causal graph to: {filepath}")
    return filepath


def main():
    OUTPUT_DIR = r"C:\Users\qadee\Desktop\CausalTGAN\data\real_world\nsl_kdd"

    print("="*70)
    print("CREATING CAUSAL GRAPH FOR NSL-KDD")
    print("="*70)
    print("Based on 10 causal features identified by Algorithm 1 (CRFS)")
    print("Paper: Zeng et al. 2021\n")

    # Expected features (from causal feature identification)
    causal_features = [
        'service',
        'dst_bytes',
        'src_bytes',
        'diff_srv_rate',
        'dst_host_srv_count',
        'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate',
        'dst_host_serror_rate',
        'count',
        'flag',
        'label'
    ]

    print("Causal Features:")
    for i, feat in enumerate(causal_features[:-1], 1):
        print(f"  {i:2d}. {feat}")
    print(f"  11. label (target)\n")

    # Create graph
    causal_graph = create_nslkdd_causal_graph()

    # Validate
    graph_features = [node for node, _ in causal_graph]
    if set(graph_features) == set(causal_features):
        print("[OK] Graph features match causal features exactly")
    else:
        missing = set(causal_features) - set(graph_features)
        extra = set(graph_features) - set(causal_features)
        if missing:
            print(f"[ERROR] Missing features: {missing}")
        if extra:
            print(f"[ERROR] Extra features: {extra}")
        return

    # Visualize
    G = visualize_graph(causal_graph)
    if G is None:
        print("[ERROR] Graph validation failed!")
        return

    # Save
    save_graph(causal_graph, OUTPUT_DIR)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"[OK] Created causal graph with {len(causal_graph)} nodes")
    print(f"[OK] {G.number_of_edges()} causal relationships defined")
    print(f"[OK] Matches Zeng et al. finding: 7-10 causal features")
    print(f"[OK] Root node: service (most influential)")
    print(f"[OK] Leaf node: label (target)")

    print(f"\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Update helper/constant.py with NSL_KDD_CATEGORY")
    print("2. Update helper/utils.py to recognize 'nsl_kdd' dataset")
    print("3. Train Causal TGAN:")
    print("   python train.py --data_name nsl_kdd --epochs 400 --batch_size 500")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
