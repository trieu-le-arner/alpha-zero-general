import Arena
from MCTS import MCTS
from go.GoGame import GoGame
from go.GoPlayers import *
from go.keras.NNet import NNetWrapper as nn


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = True

g = GoGame(n=7)

# all players
rp = RandomPlayer(g).play
# hp = HumanGoPlayer(g).play
hp = GoPlayer(g).play

# nnet players
n1 = nn(g)
# n1.load_checkpoint('./temp/', 'best.pth.tar')
n1.load_checkpoint('./temp/', 'best.h5')
args = dotdict({       
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    'cpuct': 1.0,
    'numMCTSDepth': 50,
    'maxTurns': 35,
})
args1 = args
mcts1 = MCTS(g, n1, args1)

def get_move_in_arena(canonicalBoard, mcts, game):
    # The game has ended.
    if game.getGameEnded(canonicalBoard, 1) != 0:
        return game.n * game.n

    pi = mcts.getActionProb(canonicalBoard, temp=0)
    a = np.argmax(pi)
    # valids = game.getValidMoves(canonicalBoard, 1)
    # if valids[a] == 0:
    #     pi = pi * valids  # masking invalid moves
    #     sum_pi = np.sum(pi)
    #     if sum_pi > 0:
    #         pi = pi / sum_pi  # renormalize
    #         a = np.random.choice(len(pi), p=pi)
    #     else:
    #         pi = pi + valids
    #         pi = pi / np.sum(pi)
    #         a = np.random.choice(len(pi), p=pi)
    
    print(a // game.n, a % game.n)
    return a


player1 = lambda x: get_move_in_arena(x, mcts1, g)

if human_vs_cpu:
    player2 = hp
else:
    # player2 = rp
    n2 = nn(g)
    n2.load_checkpoint('./temp/', 'best.h5')
    args2 = args
    mcts2 = MCTS(g, n2, args2)
    player2 = lambda x: get_move_in_arena(x, mcts2, g)

arena = Arena.Arena(player1, player2, g, display=GoGame.display)

print(arena.playGames(2, verbose=True)) 
