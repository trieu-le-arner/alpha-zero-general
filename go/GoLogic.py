import sys
sys.path.append('..')

import gym
from gym_go import gogame
import numpy as np

'''
Board class for the game of Go.
Default board size is 9x9.

Author: Trieu Le, https://github.com/trieu-le-arner
Date: Oct 13, 2022.

Based on the board for the game of Othello by Eric P. Nichols.

'''

class Board:
  def __init__(self, state=None, n=9, komi=0):
    self.n = n
    self.komi = komi
    
    self.env = gym.make('gym_go:go-v0', size=n, komi=komi, reward_method='real')
    self.env.reset(state=state)

  def state(self):
    return self.env.state()

  def canonical_state(self):
    return self.env.canonical_state()

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
      if self.env.prev_player_passed():
        if self.env.turn() == 0 and color == 1 and self.env.winning() == 1:
          return True
        elif self.env.turn() == 1 and color == -1 and self.env.winning() == -1:
          return True
      # elif np.sum(self.state()[3]) >= self.n * self.n * 0.5:
      #   black_area, white_area = gogame.areas(self.state())
      #   area_difference = black_area - white_area
      #   komi_correction = area_difference - self.komi
      #   ealier_winner = 0
      #   if komi_correction >= self.n * self.n / 8.0:
      #     ealier_winner = 1
      #   elif komi_correction <= -1 * self.n * self.n / 8.0:
      #     ealier_winner = -1

      #   return ealier_winner == color
      
      return False


  def execute_move(self, move, color):
    """
    Perform the given move on the board; 
    @param color (1=black, -1=white)
      color gives the color of the piece to play (1=black, -1=white)
    """
    self.env.step(move)

  def to2darray(self):
    state = self.env.state()
    
    s = [None] * self.n
    for row in range(self.n):
      s[row] = [None] * self.n
      for col in range(self.n):
        if state[0][row][col] == 1:
          s[row][col] = 1
        elif state[1][row][col] == 1:
          s[row][col] = -1
        else:
          s[row][col] = 0
    
    return np.array(s)

  def tostring(self):
    return self.to2darray().tostring()