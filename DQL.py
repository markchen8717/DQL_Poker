import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import namedtuple
import random
from pprint import pprint
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate


Experience = namedtuple('Experience_Tuple', [
                        'old_state', 'action', 'reward', 'next_state'])

'''See AI_CALLBACK_FORMAT.md for more details about some of the following parameters'''


class State():
    def __init__(self, round_state, DQL_player):
        self.DQL_player = DQL_player
        self.round_state = round_state
        self.model_input = None
        self._pot = round_state["pot"]['main']['amount']

    '''Implement me'''
    # Implement the following function base on your model's input needs

    def generate_model_input(self):
        if self.model_input is None:
            self.model_input = []
            hole_card = gen_cards(self.DQL_player.hole_card)
            community_card = gen_cards(self.round_state["community_card"])
            nb_player = sum(
                [1 if player['state'] != 'folded' else 0 for player in self.round_state["seats"]])
            x1 = estimate_hole_card_win_rate(
                nb_simulation=100, nb_player=nb_player, hole_card=hole_card, community_card=community_card)
            self.model_input.append(x1)

            player_index = [player['uuid']
                            for player in self.round_state["seats"]].index(self.DQL_player.uuid)
            btn_index = self.round_state["dealer_btn"]
            x2 = (nb_player - ((btn_index-player_index+nb_player) %  # a score 0-1 on position, 1 being the button
                               nb_player)) / nb_player
            self.model_input.append(x2)

            stack_total = 0
            player_stack = 0
            for player in self.round_state["seats"]:
                stack_total += player["stack"]
                if player["uuid"] == self.DQL_player.uuid:
                    player_stack = player["stack"]
            x3 = player_stack / stack_total
            self.model_input.append(x3)

            street_map = {'preflop': 0.25, 'flop': 0.5,
                          'turn': 0.75, 'river': 1.0}
            x4 = street_map[self.round_state["street"]]
            self.model_input.append(x4)

            x5 = self.round_state["pot"]['main']['amount'] / stack_total
            self.model_input.append(x5)

        return self.model_input


class Action():
    '''Implement me'''
    '''See AI_CALLBACK_FORMAT.md for more details about some of the following parameters'''
    # curr_state is a State object
    # not every action our model suggests is valid, hence we have the attributes
    # initial_index and final_index refering to action indices which may change
    # after action validification
    '''
    valid_actions
    [
        {'action': 'fold', 'amount': 0},
        {'action': 'call', 'amount': 0},
        {'action': 'raise', 'amount': {'max': 95, 'min': 20}}
    ]
    '''

    def __init__(self, valid_actions, curr_state, initial_index):
        self.action = None
        self.amount = None
        self.initial_index = initial_index
        self.final_index = None
        self.curr_state = curr_state
        self.valid_actions = valid_actions
        # Implement the following action_map base on what each action index
        # represents by your model.
        self.action_map = {0: {'action': 'fold', 'amount': 0, 'index': 0},
                           1: {'action': 'call', 'amount': valid_actions[1]['amount'], 'index': 1},
                           2: {'action': 'raise', 'amount': int((1/3)*curr_state._pot), 'index': 2},
                           3: {'action': 'raise', 'amount': int((3/4)*curr_state._pot), 'index': 3},
                           4: {'action': 'raise', 'amount': int(curr_state._pot), 'index': 4}}
        self.validify_action()

    '''Implement me'''

    def validify_action(self):
        # check whether the suggested action is valid
        # remeber to make shallow copy
        action_info = self.action_map[self.initial_index].copy()
        if(action_info["index"] >= 2):
            # make sure we are raising within limits
            _max = self.valid_actions[2]['amount']['max']
            _min = self.valid_actions[2]['amount']['min']
            action_info["amount"] = max(_min, action_info["amount"])
            action_info["amount"] = min(_max, action_info["amount"])
            if(_max == -1 and _min == -1):  # go all in by calling
                action_info = self.action_map[1].copy()
            elif action_info["amount"] == _min:
                action_info["index"] = 2
            elif action_info["amount"] == _max:
                action_info["index"] = 4

        self.action = action_info["action"]
        self.amount = action_info["amount"]
        self.final_index = action_info["index"]


class DQN():
    '''Implement me'''

    def __init__(self):
        self.num_of_actions = 5
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(
            100, input_shape=(5,), activation="sigmoid"))
        self.model.add(keras.layers.Dense(50, activation="relu"))
        self.model.add(keras.layers.Dense(5, activation="linear"))
        self.model.compile(
            optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

    def print(self):
        pass

    def update_weights(self, keras_model_weights):
        self.model.set_weights(keras_model_weights)

    def get_weights(self):
        return self.model.get_weights()

    # input_states_lst is a python list of state objects
    def predict(self, input_states_lst, DQL_player):
        # remeber that keras takes np lists as inputs
        np_input_states_lst = np.array(
            [state.generate_model_input() for state in input_states_lst])
        return self.model.predict(np_input_states_lst)

    # outputs_lst is a python list of correct q values 
    # for every action in each state from input_states_lst
    def fit(self, input_states_lst, outputs_lst, DQL_player):
        np_input_states_lst = np.array(
            [state.generate_model_input() for state in input_states_lst])
        np_outputs_lst = np.array(outputs_lst)
        self.model.fit(np_input_states_lst,
                       np_outputs_lst, epochs=1)


class ReplayMemory():
    def __init__(self, size, batch_size):
        if size < batch_size:
            raise Exception(
                'Replay memory size cannot be less than batch size')
        self.mem = []
        self.mem_size = size
        self.batch_size = batch_size

    def push(self, exp_tuple):
        self.mem.append(exp_tuple)
        self.mem = self.mem[max(0, len(self.mem)-self.mem_size):]

    def get_sample(self):
        if len(self.mem) < self.batch_size:
            return None
        return random.sample(self.mem, self.batch_size)


class DQL_Agent():
    def __init__(self, policy_net, replay_mem, epsilon, epsilon_decay_factor, epsilon_end, discount_factor, target_update_rate):
        self.policy_net = policy_net
        self.target_net = DQN()
        self.update_target()
        self.replay_mem = replay_mem
        self.epsilon = epsilon
        self.ep_decay = epsilon_decay_factor
        self.epsilon_end = epsilon_end
        self.q_discount = discount_factor
        self.sample_n_fit_count = 0
        self.target_update_rate = target_update_rate

    def load_model(self, h5_file_name):
        self.policy_net.model = keras.models.load_model(h5_file_name)

    def print(self):
        pprint(vars(self))
        print("Policy Net: \t")
        self.policy_net.print()
        print("Target Net: \t")
        self.policy_net.print()

    '''Implement me'''
    #Implement a reward function viable for your model
    def get_reward(self, prev_state, prev_action, new_state, DQL_player):
        prev_stack = DQL_player.prev_stack #stack size before making the action
        uuid_to_new_stack = {}
        for player in new_state.round_state["seats"]:
            uuid_to_new_stack[player['uuid']] = player['stack']
        new_stack = uuid_to_new_stack[DQL_player.uuid] #stack size after making the action
        if prev_stack == 0:
            reward = new_stack
        else:
            reward = (new_stack - prev_stack) / prev_stack
        DQL_player.prev_stack = new_stack #update stack
        return reward

    def get_action(self, valid_actions, state, DQL_player):
        # print("epsilon:",self.epsilon)
        rand_float = np.random.rand(1)[0]
        if(rand_float <= self.epsilon):
            # explore
            # this returns a random action index, check the DQN() class to see what each index maps to
            suggested_action_index = np.random.randint(
                low=0, high=self.policy_net.num_of_actions)
        else:
            # exploit
            # this returns the action index with the highest q value associated with the given state
            # the predict function expects a list of input states, and will return a list of list of action indices for each respective input state
            # we obtain the the action index with the greatest q value
            suggested_action_index = np.argmax(
                self.policy_net.predict([state], DQL_player)[0])
        self.epsilon = self.epsilon * (1-self.ep_decay)
        self.epsilon = max(self.epsilon, self.epsilon_end)
        action = Action(valid_actions, state, suggested_action_index)
        return action

    def update_replay_mem(self, state, action, reward, next_state):
        exp_tup = Experience(old_state=state, action=action,
                             reward=reward, next_state=next_state)
        self.replay_mem.push(exp_tup)

    '''
        state
        {"hole_card": hole_card, "round_state": round_state}

        action
        {'action': 'raise', 'amount': int(pot), 'index': 4}
    '''

    def sample_and_fit(self, DQL_player):
        exp_sample_lst = self.replay_mem.get_sample()  # random sample list of exp tuples
        if(exp_sample_lst is None):
            return
        # a list of all the old states in the sample
        input_states_lst = [exp_tup.old_state for exp_tup in exp_sample_lst]
        # a list of list of action q values for each old state from the policy network
        unprocessed_outputs_lst = self.policy_net.predict(
            input_states_lst, DQL_player)
        processed_outputs_lst = []
        #for each output, we must replace the chosen action q value 
        # with the max action q value in the new state from the target net
        for i in range(len(exp_sample_lst)):
            exp_sample = exp_sample_lst[i]
            max_future_q = 0
            # obtain the max future q value from target net from feeding it the new state
            if not exp_sample.next_state is None:
                max_future_q = np.max(
                    self.target_net.predict([exp_sample.next_state], DQL_player)[0])
            unprocessed_output = unprocessed_outputs_lst[i] #list of old_state action q values
            #update the chose action q value
            unprocessed_output[exp_sample.action.final_index] = exp_sample.reward + (
                self.q_discount * max_future_q)
            processed_outputs_lst.append(unprocessed_output)
        self.policy_net.fit(
            input_states_lst, processed_outputs_lst, DQL_player)

        self.sample_n_fit_count += 1
        if self.sample_n_fit_count % self.target_update_rate == 0:
            self.update_target()

    def update_target(self):
        self.target_net.update_weights(self.policy_net.get_weights())

    def save_policy(self, h5_file_name):
        self.policy_net.model.save(h5_file_name)
