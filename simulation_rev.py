# Simulations for the hyper-parameter tuning sensitivity

import numpy as np
import networkx as nx


def barabasi_albert_digraph(n):
    G = nx.DiGraph()
    G.add_nodes_from([0])
    repeated_nodes = [0]*2

    # Start adding the other nodes.
    source = len(G)
    while source < n:
        targets = np.random.choice(repeated_nodes, 1, replace=False)

        # Add edges to m nodes from the source.
        G.add_edges_from(zip(targets, [source]))

        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)

        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source])

        source += 1
    return G


def prep_sim_from_graph(G):
    regu_edges = {}
    for key, val in dict(G.in_degree).items():
        if val > 0:
            edges_in = list(G.in_edges(key))
            nodes_in = np.sort(np.array([x[0] for x in edges_in]))
            if len(nodes_in) == 2:
                if np.random.rand() > 0.5:
                    signs_in = np.array([1, 0])
                else:
                    signs_in = np.array([0, 1])
            else:
                signs_in = np.random.randint(0, 2, len(nodes_in))
            regu_edges[key] = (nodes_in, signs_in)
        else:
            regu_edges[key] = ((), ())

    return regu_edges


def graph_to_matrix(G: nx.DiGraph):
    node_names = list(G.nodes())
    node_names.sort()

    node_idx_dict = {name:idx for idx, name in enumerate(node_names)}

    # weight matrix
    n_node = len(G)
    mat_wt = np.zeros((n_node, n_node))
    for edge in G.edges():
        wt = G.edges[edge]['weight']
        n0 = node_idx_dict[edge[0]]
        n1 = node_idx_dict[edge[1]]
        mat_wt[n0, n1] = wt

    # mat_wt_sym = mat_wt + mat_wt.T
    gt_net = 1*(mat_wt!=0)

    # ground truth for gene-gene interaction
    gt_net_sym = gt_net + gt_net.T

    return mat_wt, gt_net_sym, node_idx_dict


def sim_steady_state_linear_gaussian(
    mat_wt,
    n_sample,
):
    # for each node, find inputs and weights
    n_node = len(mat_wt)
    node_in_lst = []
    noise_scl_in_lst = []
    dat_scl_in_lst = []
    for i in range(n_node):
        idx = np.where(mat_wt[:, i] != 0)[0]
        wt = mat_wt[:, i][idx]
        sign = np.ones_like(wt)
        sign[wt < 0] = -1
        scl = np.sqrt((1 / wt) ** 2 - 1)
        node_in_lst.append(idx)
        noise_scl_in_lst.append(scl)
        dat_scl_in_lst.append(sign)

    # run simulation
    dat = np.zeros((n_node, n_sample))
    stable_mask = np.zeros(n_node)
    n_max_steps = n_node

    for i in range(n_max_steps):
        node_lst = np.random.permutation(n_node)
        if int(np.sum(stable_mask)) == n_node:
            break

        for node_cur in node_lst:
            if stable_mask[node_cur] == 1:
                continue

            if len(node_in_lst[node_cur]) == 0:
                # if this node has no input, use N(0,1)
                # then set it as stable and no longer update
                x = np.random.randn(n_sample)
                stable_mask[node_cur] = 1
            else:
                # if all inputs are already stable, set current node as stable
                if np.sum(stable_mask[node_in_lst[node_cur]] == 0) == 0:
                    stable_mask[node_cur] = 1
                x = np.zeros(n_sample)

                # include the contribution of each input
                for i, node in enumerate(node_in_lst[node_cur]):
                    wt_dat = dat_scl_in_lst[node_cur][i]
                    wt_noise = noise_scl_in_lst[node_cur][i]
                    x += dat[node] * wt_dat + np.random.randn(n_sample) * wt_noise

                # scale to N(0,1)
                if np.sum(noise_scl_in_lst[node_cur]) > 0:
                    x = x - np.mean(x)
                    x = x / np.std(x)
            dat[node_cur] = x
    # print(np.sum(stable_mask))
    dat = dat.T

    return dat


def get_noise_sample(n_sample, beta_scl=1, gaussian_scl=0.1):
    xx = np.random.beta(0.5, 0.5, n_sample)*beta_scl + np.random.randn(n_sample)*gaussian_scl
    xx = xx - np.mean(xx)
    return xx/np.std(xx)


def sim_steady_state_linear_nongaussian(
    mat_wt,
    n_sample,
):
    # for each node, find inputs and weights
    n_node = len(mat_wt)
    node_in_lst = []
    noise_scl_in_lst = []
    dat_scl_in_lst = []
    for i in range(n_node):
        idx = np.where(mat_wt[:, i] != 0)[0]
        wt = mat_wt[:, i][idx]
        sign = np.ones_like(wt)
        sign[wt < 0] = -1
        scl = np.sqrt((1 / wt) ** 2 - 1)
        node_in_lst.append(idx)
        noise_scl_in_lst.append(scl)
        dat_scl_in_lst.append(sign)

    # run simulation
    dat = np.zeros((n_node, n_sample))
    stable_mask = np.zeros(n_node)
    n_max_steps = n_node

    for i in range(n_max_steps):
        node_lst = np.random.permutation(n_node)
        if int(np.sum(stable_mask)) == n_node:
            break

        for node_cur in node_lst:
            if stable_mask[node_cur] == 1:
                continue

            if len(node_in_lst[node_cur]) == 0:
                # if this node has no input, use N(0,1)
                # then set it as stable and no longer update
                # x = np.random.randn(n_sample)
                x = get_noise_sample(n_sample)
                stable_mask[node_cur] = 1
            else:
                # if all inputs are already stable, set current node as stable
                if np.sum(stable_mask[node_in_lst[node_cur]] == 0) == 0:
                    stable_mask[node_cur] = 1
                x = np.zeros(n_sample)

                # include the contribution of each input
                for i, node in enumerate(node_in_lst[node_cur]):
                    wt_dat = dat_scl_in_lst[node_cur][i]
                    wt_noise = noise_scl_in_lst[node_cur][i]
                    # noise = np.random.randn(n_sample)
                    noise = get_noise_sample(n_sample)
                    x += dat[node] * wt_dat + noise * wt_noise

                # scale to N(0,1)
                if np.sum(noise_scl_in_lst[node_cur]) > 0:
                    x = x - np.mean(x)
                    x = x / np.std(x)
            dat[node_cur] = x
    # print(np.sum(stable_mask))
    dat = dat.T

    return dat

