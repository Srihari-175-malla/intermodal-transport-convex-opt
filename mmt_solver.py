"""
Multi-Modal Transportation & Freight Routing Optimizer
Uses SciPy Linprog / CVXPY for multi-commodity flow optimization across Rail, Road, Air, and Sea transit modes.

Formulation:
  Minimizes Total Logistics Cost = Transit Cost + Transfer/Handling Cost + Carbon Tax / Emission Penalty
  Subject to:
    1. Flow conservation at all intermediate nodes.
    2. Modality capacity bounds on edges.
    3. Delivery deadline / maximum transit time constraints.
"""

import numpy as np
from scipy.optimize import linprog

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

class MultiModalTransportOptimizer:
    def __init__(self, nodes, edges):
        """
        Parameters:
        - nodes: list of node identifiers [e.g. 'Origin', 'Hub_A', 'Hub_B', 'Destination']
        - edges: list of tuples/dicts representing available multi-modal routes
                 (source, target, mode, cost_per_ton, time_hours, capacity_tons, co2_per_ton)
        """
        self.nodes = list(nodes)
        self.node_map = {node: i for i, node in enumerate(nodes)}
        self.N = len(nodes)
        self.edges = edges
        self.E = len(edges)

    def optimize_flow(self, origin, destination, demand_tons, max_transit_time=None, co2_penalty=0.0):
        orig_idx = self.node_map[origin]
        dest_idx = self.node_map[destination]

        # Objective vector c: cost per ton
        c = np.array([e['cost_per_ton'] + co2_penalty * e.get('co2_per_ton', 0.0) for e in self.edges], dtype=float)

        # Bounds: 0 <= x_e <= capacity_tons
        bounds = [(0, e['capacity_tons']) for e in self.edges]

        # Flow conservation matrix A_eq @ x == b_eq
        # Net flow at node i = Outflow - Inflow
        A_eq = np.zeros((self.N, self.E))
        for idx, e in enumerate(self.edges):
            u_idx = self.node_map[e['source']]
            v_idx = self.node_map[e['target']]
            A_eq[u_idx, idx] += 1.0
            A_eq[v_idx, idx] -= 1.0

        b_eq = np.zeros(self.N)
        b_eq[orig_idx] = demand_tons
        b_eq[dest_idx] = -demand_tons

        # Inequality constraints: A_ub @ x <= b_ub
        A_ub = []
        b_ub = []

        if max_transit_time is not None:
            times = np.array([e['time_hours'] for e in self.edges])
            A_ub.append(times)
            b_ub.append(max_transit_time * demand_tons)

        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if not res.success:
            return {"success": False, "message": res.message}

        flow_values = res.x
        route_summary = []
        for idx, e in enumerate(self.edges):
            if flow_values[idx] > 1e-4:
                route_summary.append({
                    "source": e['source'],
                    "target": e['target'],
                    "mode": e['mode'],
                    "flow_tons": float(flow_values[idx]),
                    "unit_cost": e['cost_per_ton'],
                    "total_edge_cost": float(flow_values[idx] * e['cost_per_ton'])
                })

        return {
            "success": True,
            "total_logistics_cost": float(res.fun),
            "demand_tons": demand_tons,
            "active_routes": route_summary
        }

if __name__ == "__main__":
    nodes = ['Factory', 'RailHub', 'Port', 'DistributionCenter']
    edges = [
        {'source': 'Factory', 'target': 'RailHub', 'mode': 'Road', 'cost_per_ton': 50, 'time_hours': 4, 'capacity_tons': 500, 'co2_per_ton': 0.1},
        {'source': 'Factory', 'target': 'Port', 'mode': 'Road', 'cost_per_ton': 80, 'time_hours': 6, 'capacity_tons': 300, 'co2_per_ton': 0.15},
        {'source': 'RailHub', 'target': 'DistributionCenter', 'mode': 'Rail', 'cost_per_ton': 30, 'time_hours': 12, 'capacity_tons': 1000, 'co2_per_ton': 0.04},
        {'source': 'Port', 'target': 'DistributionCenter', 'mode': 'Maritime', 'cost_per_ton': 20, 'time_hours': 24, 'capacity_tons': 1500, 'co2_per_ton': 0.02}
    ]

    opt = MultiModalTransportOptimizer(nodes, edges)
    res = opt.optimize_flow('Factory', 'DistributionCenter', demand_tons=400)
    print("=== Multi-Modal Transportation Optimization ===")
    print("Success:", res["success"])
    print(f"Total Logistics Cost: ${res['total_logistics_cost']:,.2f}")
    for r in res["active_routes"]:
        print(f"Route {r['source']} -> {r['target']} via {r['mode']}: {r['flow_tons']:.1f} tons (Cost: ${r['total_edge_cost']:,.2f})")
