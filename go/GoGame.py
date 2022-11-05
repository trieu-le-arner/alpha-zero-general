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
    return Board(n=self.n)

  def getBoardSize(self):
    return (self.n, self.n)

  # Return the number of actions
  # The semantic of +1 (the last returned action) is the pass action.
  def getActionSize(self):
    return self.n * self.n + 1

  def getNextState(self, board, player, action):
    assert player != 0
    
    # If player takes action on board, return next (board, player).
    # The selected action must be a valid move.
    next_board = Board(state=board.state(), n=self.n)

    if action == self.n * self.n: # The pass action
      next_board.execute_move(None, player)
      return (next_board, -player)
    
    move = (int(action / self.n), action % self.n)
    next_board.execute_move(move, player)
    return (next_board, -player)

  def getValidMoves(self, board, player):
    assert player != 0
    
    valids = [0] * self.getActionSize()
    valids[-1] = 1
    
    legalMoves = board.get_legal_moves(player)
    # if len(legalMoves) == 0:
    #   valids[-1] = 1 # Force the player to pass (other actions are disabled with 0)
    #   return np.array(valids)
    for x, y in legalMoves:
      valids[self.n * x + y] = 1
    return np.array(valids)

  def getGameEnded(self, board, player):
    """
    Returns:
      1 if the specified player won
      -1 if the specified player lost
      0 if the game has not ended
      a small positive value if the game tied
    """
    assert player != 0

    if board.is_win(player):
        return 1
    if board.is_win(-player):
        return -1
    
    # The game has not ended.
    # if board.has_legal_moves():
    #     return 0
    
    # Tie has a very little value. 
    # return 1e-4

    return 0

  def getCanonicalForm(self, board, player):
    assert player != 0
    return Board(state=board.canonical_state(), n=self.n)

  def getSymmetries(self, board, pi):
    # Flip and rotation
    state = board.state()
    assert(len(pi) == self.n**2 + 1)  # 1 for the pass action
    pi_board = np.reshape(pi[:-1], (self.n, self.n))
    
    l = []

    for i in range(1, 5):
      for j in [True, False]:
        newState = np.rot90(state, i, axes=(1, 2))
        newPi = np.rot90(pi_board, i)
        if j:
          newState = np.flip(newState, axis=2)
          newPi = np.flip(newPi, axis=1)
        
        l += [(Board(state=newState, n=self.n), list(newPi.ravel()) + [pi[-1]])]
    return l

  def stringRepresentation(self, board):
    # Use to hash
    return board.tostring()

  @staticmethod
  def display(board):
    s = board.to2darray()
    
    n = s.shape[0]

    print("   ", end="")
    for y in range(n):
      print (y,"", end="")
    print("")
    print("  ", end="")
    for _ in range(n):
      print ("-", end="-")
    print("--")
    for y in range(n):
      print(y, "|", end="")    # print the row #
      for x in range(n):
        piece = s[y][x]    # get the piece to print
        if piece == -1: print("X ", end="")
        elif piece == 1: print("O ", end="")
        else:
          if x == n:
            print("-", end="")
          else:
            print("- ", end="")
      print("|")

    print("  ", end="")
    for _ in range(n):
      print ("-", end="-")
    print("--")