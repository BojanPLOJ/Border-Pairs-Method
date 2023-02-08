import unittest
import numpy as np
import matplotlib.pyplot as plt
from borderpairs_basic import BorderPairsMethodClassifier


class BorderPairsMethodClassifierTest(unittest.TestCase):
	def test_random_2d_data(self):
		'''
		Simple test to classify randomly generated data. 
		Test passes if all the training data is classified correctly.
		'''
		pts = np.random.randn(int(1e2), 1024)*0.5
		pts[::2, 0] += 0
		pts[1::2, 0][::2] += 1
		pts[1::2, 1][1::2] += 1
		
		tar = np.zeros(pts.shape[0])
		tar[1::2]=1

		plt.figure()
		plt.plot(pts[tar==0, 0], pts[tar==0, 1], 'o')
		plt.plot(pts[tar==1, 0], pts[tar==1, 1], 'o')
		
		bpm_obj = BorderPairsMethodClassifier()
		bpm_obj.fit(pts, tar)
		pred = bpm_obj.predict(pts)
		np.testing.assert_array_equal(np.squeeze(pred), tar, verbose=True)


if __name__ == '__main__':
	unittest.main()
