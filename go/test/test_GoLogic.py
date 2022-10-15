import unittest
import numpy as np
from numpy.testing import assert_array_equal

from GoLogic import Board
class TestGoGame(unittest.TestCase):
  def test_execute_move(self):
    b = Board(n=2)
    b.execute_move((0, 0), 1)
    assert_array_equal(b.to2darray(), np.array([[1, 0], [0, 0]]))

if __name__ == '__main__':
  unittest.main()