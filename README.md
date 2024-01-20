# Simulation studies in DDN3.0 paper

## Installation
Required Python packages: `numpy`, `matplotlib`, `joblib`, `rpy2`, `ddn3`, and their dependencies.

Required R packages: `JGL`, `iDINGO`, `glasso`, `huge`, and their dependencies.
We call `huge` to generate synthetic data. `JGL` and `iDINGO` contain peer methods.

Users need to Install R locally, and edit the path to R in `tools_r.py`.

```python
os.environ["R_HOME"] = "C:\\Program Files\\R\\R-4.3.2"
```

## Experiments
Simulations for DDN 3.0 and JGL with different graphs can be performed in `exp_step1_batch.py`. 
The script below runs simulation on a graph with two scale free subgraphs.
It also run methods on a grid of lambda1 and lambda2 values.

```python
    l1_lst = np.arange(0.02, 1.0, 0.02)
    l2_lst = np.arange(0, 0.16, 0.025)
    graph_type = "scale-free-multi"
    batch_run(l1_lst, l2_lst, n1=200, n2=200, n_node=200, ratio_diff=0.25, graph_type=graph_type, n_group=2, n_rep=20)
```

The simualtion with imbalanced samples are in `exp_step1_batch_imbalance.py`. 
This script is similar to `exp_step1_batch.py`.
These results can be plot by `exp_step2_analysis.ipynb` and the summary plot used in the paper is generated by `exp_step2_analysis_summary.ipynb`.
The experiments related to DINGO is in `exp_step0_ddn_jgl_dingo.ipynb`.

The speed comparisons can be found in `exp_speed_comparison.ipynb`.
We found that there is negligible cost for calling R from Python.
In addition, the first time we call Numba accelerated code, like DDN 3.0, it will compile the code and it will take some time.
This only need to be done once. Therefore, we record the time for later calls of DDN 3.0.