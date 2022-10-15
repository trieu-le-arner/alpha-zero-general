import unittest
import numpy as np
from numpy.testing import assert_array_equal
from GoGame import GoGame as Game
from GoLogic import Board

class TestGoGame(unittest.TestCase):
  def test_getInitBoard(self):
    g = Game()
    b = g.getInitBoard()
    self.assertIsInstance(b, Board)

  def test_getBoardSize(self):
    g = Game(n=9)
    self.assertEqual(g.getBoardSize(), (9, 9))

  def test_getActionSize(self):
    g = Game(n=9)
    self.assertEqual(g.getActionSize(), 82)

  def test_getNextState(self):
    g = Game(n=2)
    b = g.getInitBoard()
    nextb, p = g.getNextState(b, 1, 0)
    
    self.assertIsInstance(nextb, Board)
    assert_array_equal(nextb.to2darray(), np.array([[1, 0], [0, 0]]))
    assert_array_equal(b.to2darray(), np.array([[0, 0], [0, 0]]))
    self.assertEqual(p, -1)

  def test_getNextStateWithPass(self):
    g = Game(n=2)
    b = g.getInitBoard()
    nextb, p = g.getNextState(b, 1, g.n * g.n)
    
    assert_array_equal(nextb, b)
    self.assertEqual(p, -1)

  def test_getNextStateOccupiedAction(self):
    with self.assertRaises(Exception) as context:
      g = Game(n=2)
      b = g.getInitBoard()
      b, p = g.getNextState(b, 1, 2)
      b, p = g.getNextState(b, p, 2)

    self.assertTrue("('Invalid move', (1, 0))" in str(context.exception))

  def test_getValidMoves(self):
    g = Game(n=2)
    b = g.getInitBoard()
    valids = g.getValidMoves(b, 1)
    assert_array_equal(valids, [1, 1, 1, 1, 0])

    b, p = g.getNextState(b, 1, 0)
    valids = g.getValidMoves(b, p)
    assert_array_equal(valids, [0, 1, 1, 1, 0])

    b, p = g.getNextState(b, p, 1)
    valids = g.getValidMoves(b, p)
    assert_array_equal(valids, [0, 0, 1, 1, 0])
    
    b, p = g.getNextState(b, p, 3)
    valids = g.getValidMoves(b, p)
    assert_array_equal(valids, [0, 0, 0, 0, 1])

  def test_getGameEndedPlayerWon(self):
    g = Game(n=2)
    b = g.getInitBoard()
    p = 1
    b, p = g.getNextState(b, p, 0)
    b, p = g.getNextState(b, p, 1)
    b, p = g.getNextState(b, p, 3)
    b, p = g.getNextState(b, p, 4)
    b, p = g.getNextState(b, p, 4)

    self.assertEqual(g.getGameEnded(b, -1), -1)
    self.assertEqual(g.getGameEnded(b, 1), 1)

  def test_getGameEndedOnGoing(self):
    g = Game(n=2)
    b = g.getInitBoard()
    p = 1
    b, p = g.getNextState(b, p, 0)

    self.assertEqual(g.getGameEnded(b, p), 0)

  def test_getCanonicalForm(self):
    g = Game(n=2)
    b = g.getInitBoard()
    p = 1
    b, p = g.getNextState(b, p, 0)
    b = g.getCanonicalForm(b, p)

    self.assertIsInstance(b, Board)
    self.assertEqual(b.env.turn(), 0)
    assert_array_equal(b.state()[0], [[0, 0], [0, 0]])
    assert_array_equal(b.state()[1], [[1, 0], [0, 0]])

  def test_getSymmetries(self):
    g = Game(n=2)
    b = g.getInitBoard()
    p = 1
    b, p = g.getNextState(b, p, 0)

    l = g.getSymmetries(b, [0.1, 0.2, 0.3, 0.4, 0.0])
    
    self.assertEqual(len(l), 8)
    
    assert_array_equal(l[0][0].env.state()[0], [[0, 0], [0, 1]])
    assert_array_equal(l[0][1], [0.4, 0.2, 0.3, 0.1, 0.0])

    assert_array_equal(l[1][0].env.state()[0], [[0, 0], [1, 0]])
    assert_array_equal(l[1][1], [0.2, 0.4, 0.1, 0.3, 0.0])

    assert_array_equal(l[2][0].env.state()[0], [[0, 0], [1, 0]])
    assert_array_equal(l[2][1], [0.3, 0.4, 0.1, 0.2, 0.0])

    assert_array_equal(l[3][0].env.state()[0], [[0, 0], [0, 1]])
    assert_array_equal(l[3][1], [0.4, 0.3, 0.2, 0.1, 0.0])

    assert_array_equal(l[4][0].env.state()[0], [[1, 0], [0, 0]])
    assert_array_equal(l[4][1], [0.1, 0.3, 0.2, 0.4, 0.0])

    assert_array_equal(l[5][0].env.state()[0], [[0, 1], [0, 0]])
    assert_array_equal(l[5][1], [0.3, 0.1, 0.4, 0.2, 0.0])

    assert_array_equal(l[6][0].env.state()[0], [[0, 1], [0, 0]])
    assert_array_equal(l[6][1], [0.2, 0.1, 0.4, 0.3, 0.0])

    assert_array_equal(l[7][0].env.state()[0], [[1, 0], [0, 0]])
    assert_array_equal(l[7][1], [0.1, 0.2, 0.3, 0.4, 0.0])

  def test_stringRepresentation(self):
    g = Game(n=2)
    b = g.getInitBoard()
    p = 1
    b, p = g.getNextState(b, p, 0)
    b, p = g.getNextState(b, p, 1)

    hash = g.stringRepresentation(b)
    self.assertEqual(hash, np.array([[1, -1], [0, 0]]).tostring())

if __name__ == '__main__':
  unittest.main()