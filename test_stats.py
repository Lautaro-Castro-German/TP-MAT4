import unittest
import stats

class StatsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.x = [2.25, 2.87, 3.05, 3.43, 3.68, 3.76, 3.76, 4.5, 4.5, 5.26]
        self.y = [54.74, 59.01, 72.92, 50.85, 54.99, 60.56, 69.08, 77.03, 69.97, 90.7]
    
    def test_calculate_stats(self):
        s = stats.calculate_stats(self.x, self.y)
        
        n = s['n']
        x_mean = s['x_mean']
        y_mean = s['y_mean']
        Sxy = s['Sxy']
        Sxx = s['Sxx']
        Syy = s['Syy']
        b0 = s['b0']
        b1 = s['b1']
        r = s['r']
        R2 = s['R2']
        sigma2 = s['sigma2']
        t = s['t']

        self.assertEqual(n, 10)
        self.assertAlmostEqual(x_mean, 3.706, 2)
        self.assertAlmostEqual(y_mean, 65.985, 2)
        self.assertAlmostEqual(Sxy, 72.3327, 2)
        self.assertAlmostEqual(Sxx, 7.00764, 2)
        self.assertAlmostEqual(Syy, 1360.86625, 2)
        self.assertAlmostEqual(b0, 27.73175266, 2)
        self.assertAlmostEqual(b1, 10.32197716, 2)
        self.assertAlmostEqual(r, 0.740697814, 2)
        self.assertAlmostEqual(R2, 0.548633252, 2)
        self.assertAlmostEqual(sigma2, 76.78122162, 2)
        self.assertAlmostEqual(t, 2.30600, 2)