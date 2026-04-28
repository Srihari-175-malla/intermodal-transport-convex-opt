import unittest
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mmt_solver import MultiModalTransportOptimizer

class TestMMTOptimizer(unittest.TestCase):
    def setUp(self):
        self.nodes = ['A', 'B', 'C', 'D']
        self.edges = [
            {'source': 'A', 'target': 'B', 'mode': 'Road', 'cost_per_ton': 50, 'time_hours': 4, 'capacity_tons': 500},
            {'source': 'A', 'target': 'C', 'mode': 'Road', 'cost_per_ton': 80, 'time_hours': 6, 'capacity_tons': 300},
            {'source': 'B', 'target': 'D', 'mode': 'Rail', 'cost_per_ton': 30, 'time_hours': 12, 'capacity_tons': 1000},
            {'source': 'C', 'target': 'D', 'mode': 'Sea', 'cost_per_ton': 20, 'time_hours': 24, 'capacity_tons': 1500}
        ]
        self.opt = MultiModalTransportOptimizer(self.nodes, self.edges)

    def test_basic_flow(self):
        res = self.opt.optimize_flow('A', 'D', demand_tons=400)
        self.assertTrue(res['success'])
        self.assertGreater(res['total_logistics_cost'], 0)
        # Flow via A->B->D costs (50+30)*400 = 32000
        self.assertAlmostEqual(res['total_logistics_cost'], 32000.0, places=1)

    def test_capacity_split(self):
        # Demand exceeds single route capacity (500) -> split flow across A->B->D and A->C->D
        res = self.opt.optimize_flow('A', 'D', demand_tons=700)
        self.assertTrue(res['success'])
        total_flow = sum(r['flow_tons'] for r in res['active_routes'] if r['target'] == 'D')
        self.assertAlmostEqual(total_flow, 700.0, places=1)

if __name__ == '__main__':
    unittest.main()
