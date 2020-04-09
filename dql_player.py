from pypokerengine.players import BasePokerPlayer
from pprint import pprint
from DQL import State


class DQLPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"

    def __init__(self, name, DQL_Agent):
        self.agent = DQL_Agent
        self.name = name
        self.wins = 0
        self.prev_state = None
        self.prev_action = None
        self.uuid = None
        self.last_pre_flop_state = None
        self.round_start_stack = None
        self.hole_card = None
        self.prev_stack = None

    def print(self):
        print("DQL_Player:")
        pprint(vars(self))
        print("Agent:\t")
        self.agent.print()

    def reset_state_vars(self):
        self.round_start_stack = None
        self.last_pre_flop_state = None
        self.hole_card = None
        self.prev_action = None
        self.prev_state = None

    '''
        valid_actions
        [
            {'action': 'fold', 'amount': 0},
            {'action': 'call', 'amount': 0},
            {'action': 'raise', 'amount': {'max': 95, 'min': 20}}
        ]
        See AI_CALLBACK_FORMAT.md for more details
    '''
    def declare_action(self, valid_actions, hole_card, round_state):
        curr_state = State(round_state,self)
        action = self.agent.get_action(
            valid_actions, curr_state, self)  #returns an action object
        
        #backtrack and obtain reward for previous action
        if not self.prev_state is None and not self.prev_action is None:
            reward = self.agent.get_reward(
                self.prev_state, self.prev_action, curr_state, self)
            # update replay memory for previous state action pair
            self.agent.update_replay_mem(
                self.prev_state, self.prev_action, reward,curr_state)

        #Update current state action pair
        self.prev_action = action
        self.prev_state = curr_state
        return action.action, action.amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.reset_state_vars()  # reset state vars in the beginning of each hand
        self.hole_card = hole_card
        for player in seats:
            if player["uuid"] == self.uuid:
                self.round_start_stack = player["stack"]
                self.prev_stack = player["stack"]
                break

    def receive_street_start_message(self, street, round_state):
        #obtain the round_state for simulation
        if street == "preflop":
            self.last_pre_flop_state = round_state

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        if(not self.prev_action is None and not self.prev_state is None):
            #backtrack and obtain reward for previous action
            round_state["winners"] = winners
            curr_state = State(round_state,self)
            reward = self.agent.get_reward(
                self.prev_state, self.prev_action, curr_state, self)
            # update replay memory for previous state action pair
            self.agent.update_replay_mem(
                self.prev_state, self.prev_action, reward, None) #None signifies end of round
        self.agent.sample_and_fit(self)
