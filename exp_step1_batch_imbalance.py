import numpy as np
import scan_r
from joblib import Parallel, delayed


def batch_run(l1_lst, l2_lst, n1, n2, n_node=100, ratio_diff=0.25, graph_type="random", n_group=0, n_rep=20, jgl_weight='equal', ddn_weight='rebalance'):
    # repeat, method number, l2 number, l1 number, comm/diff, metric number
    # res = np.zeros((n_rep, 2, len(l2_lst), len(l1_lst), 2, 5))
    f_out = f"./res_ddn_jgl_{graph_type}_n_{n_rep}_n-node_{n_node}_{n1}p{n2}_group_{n_group}_jgl-weights_{jgl_weight}_ddn-weight_{ddn_weight}.npz"
    print("Running ......", f_out, "\n")

    res = Parallel(n_jobs=5)(
        delayed(scan_r.scan_wrap_ddn_jgl)(
            l1_lst,
            l2_lst,
            n1,
            n2,
            n_node=n_node,
            ratio_diff=ratio_diff,
            graph_type=graph_type,
            n_group=n_group,
            weights_jgl=jgl_weight,
        )
        for _ in range(n_rep)
    )

    # save results
    res = np.array(res)
    np.savez(
        f_out,
        res=res,
        n_node=n_node,
        n1=n1,
        n2=n2,
        ratio_diff=ratio_diff,
        graph_type=graph_type,
        l1_lst=l1_lst,
        l2_lst=l2_lst,
    )


if __name__ == "__main__":
    l1_lst = np.arange(0.02, 1.0, 0.02)
    l2_lst = np.arange(0, 0.16, 0.025)

    # graph_type = "random"
    graph_type = "scale-free-multi"
    # graph_type = "scale-free"
    # graph_type = "hub"
    # graph_type = "cluster"

    # batch_run(l1_lst, l2_lst, n1=50, n2=500, n_node=100, ratio_diff=0.25, graph_type=graph_type, n_group=2, n_rep=20, jgl_weight='equal')
    batch_run(l1_lst, l2_lst, n1=50, n2=500, n_node=100, ratio_diff=0.25, graph_type=graph_type, n_group=2, n_rep=20, jgl_weight='sample.size', ddn_weight='mean')
    # batch_run(l1_lst, l2_lst, n1=275, n2=275, n_node=100, ratio_diff=0.25, graph_type=graph_type, n_group=2, n_rep=20, jgl_weight='equal')
