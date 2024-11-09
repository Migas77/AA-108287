import networkx as nx
from itertools import combinations
from random import seed, randint, shuffle
from matplotlib import pyplot as plt

vertices_lower_bound = 1
vertices_upper_bound = 1000
weight_lower_bound = 1
weight_upper_bound = 100


def generate_random_graph(n_nodes, edge_density, generator_seed=108287):
    """
    Function that allows to generate a random graph according to the following conditions:
    - graph vertices are 2D points on the XOY plane, with integer valued coordinates between
    1 and 1000.
    - graph vertices should neither be coincident nor too close.
    - the number of edges sharing a vertex is randomly determined.
    Generate successively larger random graphs, with 4, 5, 6, … vertices, using your student number as seed.
    For each fixed number of vertices, generate graph instances with 12.5%, 25%, 50% and 75% of the maximum number of
    possible edges for that number of vertices.
    :param n_nodes:
    :param edge_density:
    :param generator_seed:
    :return: graph
    """
    # using your student number as seed.
    print()
    seed(generator_seed)

    G = nx.Graph()

    # 2D points on the XOY plane, with integer valued coordinates between 1 and 1000.
    positions = set()
    while(len(positions) < n_nodes):
        x = randint(vertices_lower_bound, vertices_upper_bound)
        y = randint(vertices_lower_bound, vertices_upper_bound)
        new_pos = (x, y)
        if new_pos in positions:
            continue
        G.add_node(len(positions), pos=new_pos)
        positions.add(new_pos)

    # For each fixed number of vertices, generate graph instances with 12.5%, 25%, 50% and 75%
    # of the maximum number of possible edges for that number of vertices.
    max_n_edges = n_nodes * (n_nodes - 1) // 2
    n_edges = int(edge_density * max_n_edges)

    # all nodes should have at least one edge
    required_edges = []
    all_nodes = list(G.nodes)
    shuffle(all_nodes)
    for i in range(n_nodes - 1):
        u, v = all_nodes[i], all_nodes[i + 1]
        weight = randint(weight_lower_bound, weight_upper_bound)
        required_edges.append((u, v, weight))
    G.add_weighted_edges_from(required_edges[:min(n_edges, len(required_edges))])

    if len(G.edges) == n_edges:
        print("early", len(G.edges), n_edges, len(G.edges) == n_edges)
        return G

    # the number of edges sharing a vertex is randomly determined.
    # (random subset of all possible edges with a random weight)
    existing_edges = set(G.edges)
    remaining_possible_edges = []
    for node_combination in combinations(G.nodes, 2):
        u, v = node_combination
        edge = (u, v, randint(weight_lower_bound, weight_upper_bound))
        if (u,v) in existing_edges or (v,u) in existing_edges:
            continue
        remaining_possible_edges.append(edge)
    shuffle(remaining_possible_edges)
    G.add_weighted_edges_from(remaining_possible_edges[:n_edges - len(required_edges)])
    print("not early", len(G.edges), n_edges, len(G.edges) == n_edges)
    return G

