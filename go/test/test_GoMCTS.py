import unittest

from ..GoGame import GoGame as Game
from MCTS import MCTS
from NeuralNet import NeuralNet
from utils import *

args = dotdict({
  'numIters': 3,
  'numEps': 25,               # Number of complete self-play games to simulate during a new iteration.
  'tempThreshold': 15,        #
  'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
  'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
  'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
  'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
  'cpuct': 1,
  'checkpoint': './temp/',
  'load_model': True,
  'load_folder_file': ('./temp', 'best.pth.tar'),
  'numItersForTrainExamplesHistory': 20,
  'numMCTSDepth': 100,
})

class DummyNNet(NeuralNet):
  def train(self, examples):
    pass

  def predict(self, board):
    return [0.25, 0.25, 0.25, 0.25, 0.0], 1


class TestGoGame(unittest.TestCase):
  def test_search(self):
    g = Game(n=2)
    b = g.getInitBoard()
    p = 1

    nnet = DummyNNet(game=g)
    mcts = MCTS(g, nnet, args)
    v = mcts.search(g.getCanonicalForm(b, p))
    
    self.assertEqual(v, -1)

  def test_getActionProb(self):
    g = Game(n=2)
    b = g.getInitBoard()
    p = 1

    nnet = DummyNNet(game=g)
    mcts = MCTS(g, nnet, args)
    probs = mcts.getActionProb(g.getCanonicalForm(b, p), temp=1)
    
    self.assertEqual(len(probs), g.getActionSize())

  # def test_getActionProbFullBoard(self):
  #   g = Game(n=2)
  #   b = g.getInitBoard()
  #   p = 1

  #   b, p = g.getNextState(b, p, 0)
  #   b, p = g.getNextState(b, p, 1)
  #   b, p = g.getNextState(b, p, 3)
  #   b, p = g.getNextState(b, p, 4)

  #   print(g.getGameEnded(b, 1))
  #   print(g.getGameEnded(b, -1))

  #   nnet = DummyNNet(game=g)
  #   mcts = MCTS(g, nnet, args)
  #   probs = mcts.getActionProb(g.getCanonicalForm(b, p), temp=1)
    
  #   self.assertEqual(probs, [0.0, 0.0, 0.0, 0.0, 1.0])

if __name__ == '__main__':
  unittest.main()