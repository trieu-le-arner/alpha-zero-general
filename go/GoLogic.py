import gym

'''
Board class for the game of Go.
Default board size is 9x9.

Author: Trieu Le, https://github.com/trieu-le-arner
Date: Oct 13, 2022.

Based on the board for the game of Othello by Eric P. Nichols.

'''

class Board:
  def __init__(self, n=9):
    self.n = n
    
    env = gym.make('gym_go:go-v0', size=n, komi=0, reward_method='real')
    env.reset()

  def get_legal_moves(self, color):
    """
    Returns all the legal moves for the given color.
    (1 for black, -1 for white)
    Do not handle the "pass" move        
    """
    moves = set()
    invalid_moves = self.env.state()[3]
    assert invalid_moves.shape == (self.n, self.n)
    for row in range(self.n):
      for col in range(self.n):
        if invalid_moves[row][col] == 0:
          moves.add((row, col))

    return list(moves)

  def has_legal_moves(self):
    invalid_moves = self.env.state()[3]
    for row in range(self.n):
      for col in range(self.n):
        if invalid_moves[row][col] == 0:
          return True
    
    return False
    
  def is_win(self, color):
    """
    Check whether the given player has won the game
    @param color (1=black, -1=white)
    """
    if self.env.game_ended():
      return self.env.winner() == color
    else:
      return False


  def execute_move(self, move, color):
    """
    Perform the given move on the board; 
    @param color (1=black, -1=white)
      color gives the color of the piece to play (1=black, -1=white)
    """
    self.env.step(move)


    