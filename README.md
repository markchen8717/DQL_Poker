# DQL_Poker
A Python Tensorflow2 with Keras environment to create, train, and test Deep Q-learning agents that play no limit Texas hold 'em.

## Prerequisites
-   install pipenv from pip
-   create a virtual environment inside the project directory using the provided Pipfiles by running `pipenv install`

## DQL Algorithm:
-   Initialize replay memory capacity.
-   Initialize the policy network with random weights.
-   Clone the policy network, and call it the target network.
-   For each episode:
    -   Initialize the starting state.
    -   For each time step:
        -   Select an action.
            -   Via exploration or exploitation
        -   Execute selected action in an emulator.
        -   Observe reward and next state.
        -   Store experience in replay memory.
        -   Sample random batch from replay memory.
        -   Preprocess states from batch.
        -   Pass batch of preprocessed states to policy network.
        -   Calculate loss between output Q-values and target Q-values.
            -   Requires a pass to the target network for the next state
        -   Gradient descent updates weights in the policy network to minimize loss.
            -   After x time steps, weights in the target network are updated to the weights in the policy network.

This is the general DQL algorithm that I've implemented inside DQL_Poker.
For more details regards to DQL, you may visit this amazing course by [Deep Lizard.](https://deeplizard.com/learn/playlist/PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv)

## Usage:
-   A placeholder model is implemented for demonstration. You may implement your own model by editing functions marked with `Implement me` inside the `DQL.py` file
-   Poker simulation is done with [PyPokerEngine](https://github.com/ishikota/PyPokerEngine)
    -   You may refer to the provided [documentation](https://ishikota.github.io/PyPokerEngine/)
