from pypokerengine.api.game import setup_config, start_poker
from DQL import DQL_Agent
from DQL import DQN
from DQL import ReplayMemory
from dql_player import DQLPlayer
from collections import namedtuple
import numpy as np
from pprint import pprint
from pypokerengine.api.emulator import Emulator, Event
from pypokerengine.utils.game_state_utils import\
    restore_game_state, attach_hole_card, attach_hole_card_from_deck
from pypokerengine.engine.poker_constants import PokerConstants as Const

'''
The DQL algorithm:

Initialize replay memory capacity.
Initialize the policy network with random weights.
Clone the policy network, and call it the target network.
For each episode:
    Initialize the starting state.
    For each time step:
        Select an action.
            Via exploration or exploitation
        Execute selected action in an emulator.
        Observe reward and next state.
        Store experience in replay memory.
        Sample random batch from replay memory.
        Preprocess states from batch.
        Pass batch of preprocessed states to policy network.
        Calculate loss between output Q-values and target Q-values.
            Requires a pass to the target network for the next state
        Gradient descent updates weights in the policy network to minimize loss.
            After x time steps, weights in the target network are updated to the weights in the policy network.
'''

# hyper parameters
STARTING_STACK = 1500
SB = 10  # small blind
ANTE = {"initial": 5, "growth": 2, "rate": 10}
# increase by a factor of "growth" for every "rate" rounds
REPLAY_MEMORY_SIZE = 50  # poker hands
REPLAY_MEMORY_BATCH_SIZE = 15
EPSILON = 1  # 0-1 probability of exploration
EPSILON_DECAY = 0.005  # 0-1 percentage to truncate epsilon by every new action '''FIX'''
'''new_ep = old_ep * EPSILON_DECAY'''
EPSILON_END = 0.15  # 0-1 minimum exploration probability '''FIX'''
'''new_ep = max(new_ep,EPSILON_END)'''
DISCOUNT_FACTOR = 0.3  # 0-1 percetange to discount future q values by
'''future_q *= DISCOUNT_FACTOR'''
TOTAL_EPISODES = 10  # number of poker games
J = 5  # update target network weights for every J fits

# initialize dql agents with random weights
# make sure to give each DQL_Agent an unique name
hall_of_fame = [DQLPlayer(DQL_Agent(str(i),DQN(),
                                    ReplayMemory(
    REPLAY_MEMORY_SIZE, REPLAY_MEMORY_BATCH_SIZE),
    EPSILON, EPSILON_DECAY, EPSILON_END, DISCOUNT_FACTOR, J)) for i in range(8)]

for i in range(1,TOTAL_EPISODES+1):
    print(">>>>>>>>>>Initializing game %d<<<<<<<<<<<<"%(i))
    # set up the emulator
    emulator = Emulator()
    emulator.set_game_rule(player_num=len(
        hall_of_fame), max_round=2**32, small_blind_amount=SB, ante_amount=ANTE["initial"])

    # simulate 1 round to obtain the starting game state and player info
    config = setup_config(
        max_round=1, initial_stack=STARTING_STACK, small_blind_amount=SB)
    for player in hall_of_fame:
        config.register_player(
            name=player._name, algorithm=player)
    game_result = start_poker(config, verbose=0)
    # obtain simulated player info
    for player in game_result['players']:
        for _player in hall_of_fame:
            if _player.name == player['name']: 
                _player.uuid = player['uuid']
                emulator.register_player(_player.uuid, _player)
                break
    current_state = restore_game_state(hall_of_fame[0].last_pre_flop_state)

    print("Simulating the game")
    while True:
        current_state, events = emulator.start_new_round(current_state)
        if Event.GAME_FINISH == events[-1]["type"]:
            break
        #update game rules
        round_num = current_state["round_count"]
        new_ante = int(ANTE["initial"] * (ANTE["growth"]
                                          ** (round_num//ANTE["rate"])))
        emulator.set_game_rule(player_num=len(
            hall_of_fame), max_round=2**32, small_blind_amount=SB, ante_amount=new_ante)
        print("Playing round %d"%(round_num))
        current_state, events = emulator.run_until_round_finish(current_state)
        #fit players
        round_state = events[-1]["round_state"]
        winners = events[1]["winners"]
        for player in hall_of_fame:
            player.receive_round_result_message(winners,None,round_state)
        if Event.GAME_FINISH == events[-1]["type"]:
            break

    print("Finished playing game %d"%(i))
    #calculate score
    player_to_stack = {}
    winner = ("uuid",0)
    for player in events[-1]['players']:
        if player["stack"] > winner[1]:
            winner = (player["uuid"],player["stack"])
        player_to_stack[player["uuid"]] = player["stack"]
    for _player in hall_of_fame:
        if _player._uuid == winner[0]:
            winner = (_player._name,winner[1])
    print("Player %s has won!"%(winner[0]))
    player_to_wins ={}
    for player in hall_of_fame:
        if player._uuid == winner[0]:
            player.wins+=1
        player_to_wins [player._uuid] = player.wins
    print("Score board:", player_to_wins)

    print("Saving DQL models")
    for player in hall_of_fame:
        player.agent.save_policy("player_"+player._name+".h5")
