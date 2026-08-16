# MILP Visualizer

Turn a MILP/LP model into a 2D map of its structure using graph embedding. Variables placed closer together are more "similar" in some sense. In a CVRP model, for example, each vehicle forms its own cluster, and each cluster breaks down further into sub-clusters per customer node. Since these patterns emerge from problem structure and not variable names, similar models produce similar looking maps. This introduces a way to structurally compare different models, without the need of initial knowledge.

## Install

```bash
pip install milp_visualizer
```

## Quick start

```python
import milp_visualizer

milp_visualizer.visualize("model.mps")                 # or "model.lp"
```

Creates `variable_graph.html`, which contains the final result.

Works directly on live solver models too, no file needed:

```python
import gurobipy as gp

model = gp.Model()
# ... build model ...
milp_visualizer.visualize(model)
```

Similarly, `HiGHS` and `ortools` models can be visualized by supplying the model as first parameter.

## Usage

```python
milp_visualizer.visualize(
    source,                     # path (str/Path) or a Gurobi/HiGHS/OR-Tools model
    output=None,                # output path e.g. "graph.html" or "graph.png"; default: variable_graph.html / constraint_graph.html
    mode="variables",           # "variables" or "constraints"
    exclude=None,               # drop nodes before graph construction list of prefixes and globs
    groups=None,                # merge related nodes into one before embedding
    max_neighbors=None,         # cap edges drawn per node
    label_nodes=None,           # annotate node names (default: auto for <=50 nodes)
    node_categories=None,       # {node_name: hex_color} manual color override
)
```

### `exclude` / `groups` syntax

Both take a string (or list of strings). Plain strings are a prefix match;
strings containing `*` or `?` are a glob:

- `*` captures an integer - used by `groups` to partition nodes by that value
- `?` matches any integer without capturing it

```python
milp_visualizer.visualize("model.mps", exclude="slack_")   # drop all slack_* variables
milp_visualizer.visualize("model.mps", groups=["x[?,*]"])  # group x[i,j] by j
```

## Output files

`visualize()` writes one file, based on `output`'s extension:

- `.html` (default, recommended) - interactive Plotly scatter
- `.png` - static matplotlib scatter

## License

[GPL](LICENSE)