import sys
sys.path.append('..')
from Game import Game
from .GoLogic import Board
import numpy as np

"""
Game class implementation for the game of Go.

Author: Trieu Le, https://github.com/trieu-le-arner
Date: Oct 13, 2022.

Based on the OthelloGame by Surag Nair.
"""

class GoGame(Game):
  def __init__(self, n=9):
    self.n = n

  def getInitBoard(self):
    b = Board(self.n)
    return np.array(b.pieces)