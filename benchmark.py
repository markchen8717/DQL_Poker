from pypokerengine.api.game import setup_config, start_poker
from DQL import DQL_Agent
from DQL import DQN
from DQL import ReplayMemory
from dql_player import DQLPlayer
import numpy as np
from pypokerengine.api.emulator import Emulator, Event
from pypokerengine.utils.game_state_utils import restore_game_state
from pypokerengine.engine.poker_constants import PokerConstants as Const

# hyper parameters
STARTING_STACK = 1500
SB = 10  # small blind
ANTE = {"initial": 5, "growth": 3, "rate": 10}
# increase by a factor of "growth" for every "rate" rounds
REPLAY_MEMORY_SIZE = 100  # actions
REPLAY_MEMORY_BATCH_SIZE = 50
EPSILON = 0  # 0-1 probability of exploration
EPSILON_DECAY = 0.005  # 0-1 percentage to truncate epsilon by every new action 
'''new_ep = old_ep * (1-EPSILON_DECAY)'''
EPSILON_END = 0  # 0-1 minimum exploration probability
'''new_ep = max(new_ep,EPSILON_END)'''
DISCOUNT_FACTOR = 0.3  # 0-1 percetange to discount future q values by
'''future_q *= DISCOUNT_FACTOR'''
TOTAL_EPISODES = 100  # number of poker games
J = 10  # update target network weights for every J fits

# initialize dql agents with random weights
# make sure to give each DQL_Agent an unique name
hall_of_fame = [DQLPlayer(str(i),DQL_Agent(DQN(),
                                    ReplayMemory(
    REPLAY_MEMORY_SIZE, REPLAY_MEMORY_BATCH_SIZE),
    EPSILON, EPSILON_DECAY, EPSILON_END, DISCOUNT_FACTOR, J)) for i in range(8)]

#load player models
for player in hall_of_fame:
    player.agent.load_model("player_"+str(player.name)+".h5")

config = setup_config(max_round=2**32, initial_stack=1500, small_blind_amount=10)
for player in hall_of_fame:
    config.register_player(name=player.name, algorithm=player)

game_result = start_poker(config, verbose=1)

