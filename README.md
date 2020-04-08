# DQL_Poker
A Python environment to create, train, and test Deep Q-learning agents that play no limit Texas hold 'em

## Prerequisites
-   install pipenv from pip
-   create a virtual environment inside the project directory using the provided Pipfiles by running `pipenv install`

## The DQL algorithm:

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

