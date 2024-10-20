from pyomo.environ import ConcreteModel, Set, Param, Var, Objective, Constraint, NonNegativeReals, Binary
from pyomo.contrib.appsi.solvers import Highs
import itertools

model = ConcreteModel()

# Edges
# node, node, value
# edge[0] < edge[1]
edges = {
    (1, 2): 1,
    (1, 3): 1,
    (2, 3): 10,
    # (3, 4, 1)
}

# Nodes
def get_nodes(edges: dict[tuple[int, int], int | float]) -> set[int]:
    nodes = set()
    for edge in edges.keys():
        nodes.add(edge[0])
        nodes.add(edge[1])
    
    return nodes


# Distance for all possible edges, undirected graph
def add_non_existent_edges(
    nodes: set,
    edges: dict[tuple[int, int], int | float],
    edge_value: int | float
    ) -> dict[tuple[int, int], int | float]:
    '''Add value for all non-existent edges.'''


    for edge in itertools.product(nodes, repeat=2):
        is_ordered_edge = edge[0] < edge[1]
        if is_ordered_edge and edge not in edges:
            edges[edge] = edge_value

    return edges

nodes = get_nodes(edges)
nonex_edge_value = 2 * sum(edges.values())
M = 2 * nonex_edge_value
edges = add_non_existent_edges(nodes, edges, nonex_edge_value)
edge_keys = list(edges.keys())
edge_values = list(edges.values())
n_dims = len(nodes) + 1

# Indices
model.nodes = Set(initialize=list(nodes))
model.edges = Set(initialize=range(len(edges)))
model.dims = Set(initialize=range(n_dims))

# Parameters
model.v = Param(model.edges, initialize=edge_values)

# Variables
model.x = Var(model.nodes, model.dims, domain=NonNegativeReals)
model.d = Var(model.edges, model.dims, domain=NonNegativeReals)
model.b = Var(model.edges, model.dims, domain=Binary)
model.l = Var(model.edges, domain=NonNegativeReals)


# Objective function
model.obj = Objective(expr=sum(model.l[i] for i in model.edges), sense=1)

# Constraints
# dist_p_constr and dist_n_constr constraints are mutually feasible
def dist_p_constr(model, edge, dim):
    i, j = edge_keys[edge]
    return model.x[i, dim] - model.x[j, dim] <= model.d[edge, dim]

def dist_n_constr(model, edge, dim):
    i, j = edge_keys[edge]
    return -(model.x[i, dim] - model.x[j, dim]) <= model.d[edge, dim]

# dist_p_eq_constr and dist_n_eq_constr force equailty |x_i - x_j| = d,
# but would be mutually exclusive without big-M
def dist_p_eq_constr(model, edge, dim):
    i, j = edge_keys[edge]
    return model.x[i, dim] - model.x[j, dim] >= model.d[edge, dim] - M * (1 - model.b[edge, dim])

def dist_n_eq_constr(model, edge, dim):
    i, j = edge_keys[edge]
    return -(model.x[i, dim] - model.x[j, dim]) >= model.d[edge, dim] - M * model.b[edge, dim]

def loss_p_constr(model, edge):
    l1_dist = sum(model.d[edge, dim] for dim in model.dims)
    return model.v[edge] - l1_dist <= model.l[edge]

def loss_n_constr(model, edge):
    l1_dist = sum(model.d[edge, dim] for dim in model.dims)
    return -(model.v[edge] - l1_dist) <= model.l[edge]

model.dist_p = Constraint(model.edges, model.dims, rule=dist_p_constr)
model.dist_n = Constraint(model.edges, model.dims, rule=dist_n_constr)
model.dist_p_eq = Constraint(model.edges, model.dims, rule=dist_p_eq_constr)
model.dist_n_eq = Constraint(model.edges, model.dims, rule=dist_n_eq_constr)
model.loss_p = Constraint(model.edges, rule=loss_p_constr)
model.loss_n = Constraint(model.edges, rule=loss_n_constr)

# Solve the model using HiGHS
solver = solver = Highs()
highs_results = solver.solve(model)

print(highs_results.termination_condition)
