import unittest
import numpy as np
from numpy.testing import assert_array_equal

from ..GoLogic import Board

class TestGoGame(unittest.TestCase):
  def test_execute_move(self):
    b = Board(n=2)
    b.execute_move((0, 0), 1)
    assert_array_equal(b.to2darray(), np.array([[1, 0], [0, 0]]))

  def test_is_win(self):
    b = Board(n=3)
    b.execute_move((1, 0), 1)
    b.execute_move((2, 0), -1)
    b.execute_move((0, 1), 1)
    b.execute_move((1, 2), -1)
    b.execute_move((0, 2), 1)
    b.execute_move((2, 2), -1)
    b.execute_move((1, 1), 1)
    b.execute_move(None, -1)

    self.assertEqual(b.is_win(1), True)

if __name__ == '__main__':
  unittest.main()