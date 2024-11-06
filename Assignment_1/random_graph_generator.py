import networkx as nx
from itertools import combinations
from random import seed, randint, shuffle

from matplotlib import pyplot as plt

vertices_lower_bound = 1
vertices_upper_bound = 1000
weight_lower_bound = 1
weight_upper_bound = 1000


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
    seed(generator_seed)

    G = nx.Graph()

    # 2D points on the XOY plane, with integer valued coordinates between 1 and 1000.
    for i in range(n_nodes):
        x = randint(vertices_lower_bound, vertices_upper_bound)
        y = randint(vertices_lower_bound, vertices_upper_bound)
        G.add_node(i, pos=(x, y))

    # For each fixed number of vertices, generate graph instances with 12.5%, 25%, 50% and 75%
    # of the maximum number of possible edges for that number of vertices.
    max_n_edges = n_nodes * (n_nodes - 1) // 2
    n_edges = int(edge_density * max_n_edges)

    # the number of edges sharing a vertex is randomly determined.
    # (random subset of all possible edges with a random weight)
    all_possible_edges = [
        # (x, y, weight)
        (*node_combination, randint(weight_lower_bound, weight_upper_bound))
        for node_combination in combinations(G.nodes, 2)
    ]
    shuffle(all_possible_edges)
    edges = all_possible_edges[:n_edges]
    G.add_weighted_edges_from(edges)
    return G

