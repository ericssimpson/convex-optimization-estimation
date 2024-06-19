import unittest
from src import feasible_point, hs_intersection, solve_convex_set, generate_linear_program

class TestOptimization(unittest.TestCase):
    def test_feasible_point(self):
        A = np.array([[1, 2], [-1, -2], [1, -1]])
        b = np.array([2, -1, 1])
        point = feasible_point(A, b)
        self.assertEqual(len(point), 2)

    def test_generate_linear_program(self):
        A, b, c = generate_linear_program(2, 3, -10, 10)
        self.assertEqual(A.shape, (3, 2))
        self.assertEqual(len(b), 3)
        self.assertEqual(len(c), 2)

if __name__ == '__main__':
    unittest.main()
