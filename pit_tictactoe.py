import Arena
from MCTS import MCTS
from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.TicTacToePlayers import *
from tictactoe.keras.NNet import NNetWrapper as nn


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = True

g = TicTacToeGame()

# all players
rp = RandomPlayer(g).play
hp = HumanTicTacToePlayer(g).play



# nnet players
n1 = nn(g)
n1.load_checkpoint('./temp/', 'best.pth.tar')
args1 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
player1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    # n2 = NNet(g)
    # n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
    # args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    # mcts2 = MCTS(g, n2, args2)
    # n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    # player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.
    player2 = rp

arena = Arena.Arena(player1, player2, g, display=TicTacToeGame.display)

print(arena.playGames(2, verbose=True)) 
