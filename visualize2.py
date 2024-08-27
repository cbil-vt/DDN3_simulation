"""Visualization functions of DDN, improved v2 version

Deprecated. Use the iDDN visualize_multi functions instead.

This one is only used in the DDN3.0 paper revision.

These functions are still quite basic.
For advanced plotting of networks, consider using specialized tools.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from iddn import visualize_basic as vizcomm


def draw_network_for_ddn(
    edges_df,
    nodes_to_draw,
    mode="common",
    nodes_type=None,
    cen_lst=None,
    rad_lst=None,
    labels=None,
    fig_size=(16, 9),
    font_size_scale=1,
    node_size_scale=1,
    min_alpha=0.4,
    dashed=True,
    pdf_name="",
):
    """Draw networks for DDN

    Support drawing any number of ellipses according to `nodes_type`.
    The positions and shapes are given in `cen_lst` and `rad_lst`.
    This makes the layout of the graphs more flexible.

    The direction of the labels now points to the center of each ellipse.
    This allows showing larger fonts.

    The node size, font size, edge weight are now automatically adjusted according to figure size and node number.

    For common network, if two nodes in an edge have same type, draw grey line.
    If two nodes in an edge have different type, draw green line.

    For differential network, if an edge comes from condition 1, draw blue line.
    If an edge comes from condition 2, draw red line.

    Parameters
    ----------
    edges_df : pandas.DataFrame
        Edge information.
        First two columns for the two feature names.
        Third column for edge type (common=0, diff1=1, diff2=2)
        Fourth column for weight.
    nodes_to_draw : list of str
        Name of nodes to draw
    fig_size : tuple, optional
        Size of figure.
    font_size_scale : int, optional
        Scale of fonts, by default 1
        To make the font larger, set larger value.
        The default value is scaled according to the node number.
    node_size_scale : int, optional
        Scale of node sizes, by default 2
    nodes_type : None or dict, optional
        Node type (e.g., Gene=0, TF=1), by default None
    cen_lst : None or ndarray, optional
        The center of ellipse for each type of node. For node type i, use cen_lst[i]
        Shape (k, 2), k is the number of types. 2 for (x, y)
        Assume the y-axis of the figure is in range [-1,1], so do not set too large values.
    rad_lst : None or ndarray, optional
        The radius of ellipse for each type of node. For node type i, use rad_lst[i]
        Shape (k, 2), k is the number of types. 2 for shorter and longer axis length.
        Assume the y-axis of the figure is in range [-1,1], so do not set too large values.
    labels : dict
        Alternative (e.g., simplified) names for nodes
    mode : str, optional
        Draw common graph or differential graph, by default "common"
    pdf_name : str, optional
        Name of the PDF file to export, by default "". If set as "", no pdf will be output.

    Returns
    -------
    nx.Graph
        Graph object
    """
    # by default, assume there is one node type, and we draw circle
    if nodes_type is None:
        nodes_type = dict()
        for node in nodes_to_draw:
            nodes_type[node] = 0

    if cen_lst is None:
        cen_lst = np.array([[0.0, 0.0]])

    if rad_lst is None:
        rad_lst = np.array([[1.0, 1.0]])

    # create networkx graph
    G = create_nx_graph(
        edges_df,
        nodes_to_draw,
        min_alpha=min_alpha,
        mode=mode,
        nodes_type=nodes_type,
    )

    if labels is None:
        labels = dict((n, n) for n in G.nodes())

    # nodes positions
    pos, d_min = vizcomm.get_pos_multi_parts(
        nodes_to_draw, nodes_type, cen_lst=cen_lst, rad_lst=rad_lst
    )

    # plot the network
    fig, ax = plot_network(
        G,
        pos,
        d_min=d_min,
        labels=labels,
        node_type=nodes_type,
        cen_lst=cen_lst,
        rad_lst=rad_lst,
        fig_size=fig_size,
        font_size_scale=font_size_scale,
        node_size_scale=node_size_scale,
        use_dashed=dashed,
    )

    # export figure
    if len(pdf_name) > 0:
        plt.savefig(f"{pdf_name}_{mode}.pdf", format="pdf", bbox_inches="tight")

    return G, fig, ax


def create_nx_graph(
    edges_df,
    nodes_show,
    min_alpha=0.4,
    max_alpha=1.0,
    mode="common",
    nodes_type=None,
):
    """Create NetworkX graph based on edge iddn_data frame.

    Add nodes and edges. Provide visualization related properties to the nodes.

    For common network, if two nodes in an edge have same type, draw grey line.
    If  two nodes in an edge have different type, draw green line.

    For differential network, if an edge comes from condition 1, draw blue line.
    If an edge comes from condition 2, draw red line.

    Parameters
    ----------
    edges_df : pandas.DataFrame
        Edge information.
        First two columns for the two feature names.
        Third column for edge type (common=0, diff1=1, diff2=2)
        Fourth column for weight.
    nodes_show : list of str
        Name of nodes to draw
    min_alpha : float, optional
        Minimum alpha value of edges, by default 0.2
        This is for the most light edges.
    max_alpha : float, optional
        Maximum alpha value of edges, by default 1.0
    mode : str, optional
        Draw common graph or differential graph, by default "common"
    nodes_type : None or dict, optional
        Node type (e.g., Gene=0, TF=1), by default None

    Returns
    -------
    nx.Graph
        Generated graph
    """
    # create the overall graph
    color_condition = {
        0: [0.7, 0.7, 0.7],
        1: [0.21484375, 0.4921875, 0.71875],
        2: [0.890625, 0.1015625, 0.109375],
        3: [0, 0.6, 0.3],
    }
    beta_max = np.max(edges_df["weight"])

    if nodes_type is None:
        nodes_type = dict()
        for node in nodes_show:
            nodes_type[node] = 0

    G = nx.Graph()
    for node in nodes_show:
        G.add_node(node)

    for i in range(len(edges_df)):
        gene1, gene2, condition, beta = edges_df.iloc[i]
        if condition in color_condition:
            alpha = np.abs(beta) / beta_max * (max_alpha - min_alpha) + min_alpha
            weight = np.abs(beta) / beta_max * 3.0 + 0.5
            if mode != "common":
                color = list(1 - (1 - np.array(color_condition[condition])) * alpha)
            else:
                if nodes_type[gene1] == nodes_type[gene2]:
                    color = list(1 - (1 - np.array(color_condition[0])) * alpha)
                else:
                    color = list(1 - (1 - np.array(color_condition[3])) * alpha)
            G.add_edge(gene1, gene2, color=color, weight=weight)

    return G


def plot_network(
    G,
    pos,
    d_min,
    labels,
    node_type,
    cen_lst,
    rad_lst,
    fig_size,
    font_size_scale=1,
    node_size_scale=2,
    font_alpha_min=0.4,
    use_dashed=True,
):
    """Draw the network

    Parameters
    ----------
    G : nx.Graph
        Graph to draw
    pos : dict
        Position of each node
    d_min : float
        Minimum distance between nodes.
        We use this to adjust node size, font size, etc.
    labels : dict
        Alternative names for nodes
    node_type : None or dict, optional
        Node type (e.g., Gene=0, TF=1), by default None
    cen_lst : None or ndarray, optional
        The center of ellipse for each type of node. For node type i, use cen_lst[i]
        Shape (k, 2), k is the number of types. 2 for (x, y)
        Assume the y-axis of the figure is in range [-1,1], so do not set too large values.
    fig_size : tuple
        Size of figure. The unit is inch.
    font_size_scale : int, optional
        Scale of fonts, by default 1
        To make the font larger, set larger value.
        The default value is scaled according to the node number.
    node_size_scale : int, optional
        Scale of node sizes, by default 1
    font_alpha_min : float, optional
        The smallest alpha value for fonts in labels, between 0 and 1
    """
    # The positions are given in a [-a,a]x[-1,1] region
    # Re-scale it to figure size, but leave some margin for text (here 0.8)
    fig_half_size = fig_size[1] * 0.9 / 2
    for x in pos:
        pos[x] = pos[x] * fig_half_size
    cen_lst = cen_lst * fig_half_size
    d_min = d_min * fig_half_size

    # node size
    # Node size in unit points^2. 1 inch = 72 points.
    # in case all nodes have degree zero
    # too large nodes are ugly
    s_min = (d_min * 72) ** 2
    s_min = min(s_min, 36 * 36)
    node_size = np.array([d for n, d in G.degree()]) + 0.1
    node_size = node_size / (np.max(node_size) + 1)
    node_size = node_size * s_min * node_size_scale

    # font size
    # In points. 1 inch = 72 points. Font size about the height of a character.
    # too large font may go outside the figure
    font_size = d_min * 72 * 0.8 * font_size_scale
    font_size = min(font_size, fig_half_size * 0.1 * 20)
    font_size_lst = font_size + node_size * 0
    # font_size_lst = font_size * (
    #     np.abs(node_size) / np.max(node_size) * (1.0 - 0.5) + 0.5
    # )
    font_alpha_lst = (
        np.abs(node_size) / np.max(node_size) * (1.0 - font_alpha_min) + font_alpha_min
    )

    # draw
    fig, ax = plt.subplots(figsize=fig_size)

    # color_lst = []
    # for n in node_type:
    #     if n[-2:] == '_0':
    #         color_lst.append("blue")
    #     else:
    #         color_lst.append("lightblue")

    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        # node_color=color_lst,
        node_color="lightblue",
        node_size=node_size,
        alpha=0.5,
    )

    # edges properties
    # in case there are no edges
    # too thick edges are ugly
    edges = G.edges()
    if len(edges) > 0:
        edge_color = np.array([G[u][v]["color"] for u, v in edges])
        edge_weight = np.array([G[u][v]["weight"] for u, v in edges])

        e_mean = np.mean(edge_weight)
        e_std = np.std(edge_weight)
        std_scl = 1
        edge_weight[edge_weight < e_mean - std_scl * e_std] = e_mean - std_scl * e_std
        edge_weight[edge_weight > e_mean + std_scl * e_std] = e_mean + std_scl * e_std
        print(np.min(edge_weight), np.max(edge_weight), len(edge_weight))

        # when there are too many edges, make the edge thin
        # edge weight also in points, 1 inch = 72 points
        d_min1 = min(d_min, 0.1)
        edge_weight = edge_weight / np.max(edge_weight) * d_min1 * 72 / 6
        if len(edge_weight) > 200:
            edge_weight = edge_weight / len(edge_weight) * 200

        if use_dashed:
            edge_grp = []
            edge0 = []
            edge1 = []
            for u, v in edges:
                if u[-1:] == v[-1:]:
                    edge_grp.append(0)
                    edge0.append([u, v])
                else:
                    edge_grp.append(1)
                    edge1.append([u, v])
            edge_grp = np.array(edge_grp)

            nx.draw_networkx_edges(
                G,
                pos=pos,
                ax=ax,
                edgelist=edge0,
                edge_color=edge_color[edge_grp == 0],
                width=edge_weight[edge_grp == 0],
                style="dashed",
            )

            nx.draw_networkx_edges(
                G,
                pos=pos,
                ax=ax,
                edgelist=edge1,
                edge_color=edge_color[edge_grp == 1],
                width=edge_weight[edge_grp == 1],
                style="solid",
            )
        else:
            nx.draw_networkx_edges(
                G,
                pos=pos,
                ax=ax,
                edgelist=edges,
                edge_color=edge_color,
                width=edge_weight,
                style="solid",
            )

    font_color_lst = ["black" for _ in labels]

    vizcomm.draw_network_labels(
        ax,
        pos,
        d_min,
        node_type,
        cen_lst,
        rad_lst,
        labels,
        font_size_lst,
        font_alpha_lst,
        font_color_lst,
    )

    ax.set_xlim((-fig_size[0] / 2, fig_size[0] / 2))
    ax.set_ylim((-fig_size[1] / 2, fig_size[1] / 2))
    return fig, ax
