import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import combinations


def exhaustive_search_max_weight_matching_animated(G):
    """
    Find the maximum weighted matching using exhaustive search.

    Parameters:
    - G: A networkx.Graph instance with weighted edges.
    """
    max_weight = 0
    max_weight_matching = set()

    pos = nx.spring_layout(G, seed=7)
    labels = nx.get_edge_attributes(G, 'weight')
    fig, ax = plt.subplots(figsize=(9, 9))
    nx.draw(
        G, pos, with_labels=True,
        ax=ax, width=3
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    info_text = ax.text(
        -0.15, 1.14, "", transform=ax.transAxes,
        horizontalalignment='left', verticalalignment='top', fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
    )

    # Get all edges with weights
    edges = G.edges(data="weight")
    edge_combinations = [subset for r in range(1, len(edges) + 1) for subset in combinations(edges, r)]

    def exhaustive_search_and_update_frame(frame):
        """
        Animated function to find the maximum weighted matching.

        Parameters:
        - frame: The current frame index.
        """
        nonlocal max_weight, max_weight_matching
        # Get edge combination for the current frame
        subset = edge_combinations[frame]

        # Extract edge pairs
        # no need to extract the weights as I'm using nx.is_matching
        edge_set = {(u, v) for u, v, _ in subset}

        # clear last frame
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color="black", width=3, ax=ax)

        # Check if the subset is a valid matching
        if nx.is_matching(G, edge_set):
            nx.draw_networkx_edges(G, pos, edgelist=edge_set, edge_color="green", width=3, ax=ax)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

            # Calculate the total weight of the matching
            # and update if found maximum weight
            weight = sum(weight for _, _, weight in subset)
            if weight > max_weight:
                max_weight = weight
                max_weight_matching = edge_set
            info_text.set_text(
                f"MaxWeightMatching: {max_weight_matching}\n"
                f"MaxWeight: {max_weight}\n"
                f"Matching: False\n"
                f"Weight: {weight}"
            )
        else:
            nx.draw_networkx_edges(G, pos, edgelist=edge_set, edge_color="red", ax=ax)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
            info_text.set_text(
                f"MaxWeightMatching: {max_weight_matching}\n"
                f"MaxWeight: {max_weight}\n"
                f"Matching: False\n"
                f"Weight: N/A"
            )


    anim = FuncAnimation(
        fig, exhaustive_search_and_update_frame,
        frames=len(edge_combinations), interval=1000, repeat=False
    )
    plt.show()


if __name__ == '__main__':
    Graph = nx.Graph()
    Graph.add_weighted_edges_from(
        [(1, 2, 2), (1, 3, 1), (2, 3, 4), (2, 4, 3)]
    )
    print(
        nx.max_weight_matching(Graph),
        sum(Graph.get_edge_data(*e)["weight"] for e in nx.max_weight_matching(Graph))
    )
    exhaustive_search_max_weight_matching_animated(Graph)

# Edmond Blossom Algorithm
# networkX max_weight_matching
