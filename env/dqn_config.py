from .custom_env import SUMO_PARAMS

import torch.nn as nn
import torch.optim as optim
from torch import no_grad, as_tensor

CONFIG = "platoon_simplified"

# """CHANGE HYPER PARAMETERS HERE""" ###################################################################################
HYPER_PARAMS = {
    'gpu': '0',                                 # GPU #
    'n_env': 1,                                 # Multi-processing environments
    'lr': 2e-04,                                # Learning rate
    'gamma': 0.9,                              # Discount factor
    'eps_start': 1,                            # Epsilon start
    'eps_min': 0.01,                            # Epsilon min
    'eps_dec': 3e6,                             # Epsilon decay
    'eps_dec_exp': True,                        # Epsilon exponential decay
    'bs': 32,                                   # Batch size
    'min_mem': 5000,                          # Replay memory buffer min size
    'max_mem': 50000,                         # Replay memory buffer max size
    'target_update_freq': 5000,                # Target network update frequency
    'target_soft_update': True,                 # Target network soft update
    'target_soft_update_tau': 1e-03,            # Target network soft update tau rate
    'save_freq': 1000,                         # Save frequency
    'log_freq': 5000,                           # Log frequency
    'save_dir': './save/' + CONFIG + "/",       # Save directory
    'log_dir': './logs/train/' + CONFIG + "/",  # Log directory
    'load': False,                               # Load model
    'repeat': 0,                                # Repeat action
    'max_episode_steps': 1000,                  # Time limit episode steps
    'max_total_steps': 0,                       # Max total training steps if > 0, else inf training
    'algo': 'PerDuelingDoubleDQNAgent'          # DQNAgent
                                                # DoubleDQNAgent
                                                # DuelingDoubleDQNAgent
                                                # PerDuelingDoubleDQNAgent
}

########################################################################################################################


# """CHANGE NETWORK CONFIG HERE""" #####################################################################################
def network_config(input_dim):
    # """CHANGE NETWORK HERE""" ########################################################################################
    """
    cnn_dims = (
        (16, 4, 1),
        (32, 2, 1)
    )

    fc_dims = (32, 16)

    activation = nn.ELU()

    cnn = nn.Sequential(
        nn.Conv2d(input_dim.shape[0], cnn_dims[0][0], kernel_size=cnn_dims[0][1], stride=cnn_dims[0][2]),
        activation,
        nn.Conv2d(cnn_dims[0][0], cnn_dims[1][0], kernel_size=cnn_dims[1][1], stride=cnn_dims[1][2]),
        activation,
        nn.Flatten()
    )

    with no_grad():
        n_flatten = cnn(as_tensor(input_dim.sample()[None]).float()).shape[1]

    net = nn.Sequential(
        cnn,
        nn.Linear(n_flatten, fc_dims[0]),
        activation,
        nn.Linear(fc_dims[0], fc_dims[1]),
        activation
    )"""


    activation = nn.LeakyReLU()

    '''net = nn.Sequential(
        nn.Linear(14, 64),
        activation,
        nn.Linear(64, 128),
        activation,
        nn.Linear(128, 64),
        activation,
        nn.Linear(64, 64),
        activation,
        nn.Linear(64, 48),
        activation,
    )'''
    net = nn.Sequential(
        nn.Linear(14, 32),
        activation,
        nn.Linear(32, 64),
        activation,
        nn.Linear(64, 128),
        activation,
        nn.Linear(128, 64),
        activation,
        nn.Linear(64, 48),
        activation,
    )
    """"""
    """
    cnn_dims = (
        (32, 4, 2),
        (64, 2, 2),
        (128, 2, 1)
    )

    fc_dims = (128, 64)

    activation = nn.ELU()

    cnn = nn.Sequential(
        nn.Conv2d(input_dim.shape[0], cnn_dims[0][0], kernel_size=cnn_dims[0][1], stride=cnn_dims[0][2]),
        activation,
        nn.Conv2d(cnn_dims[0][0], cnn_dims[1][0], kernel_size=cnn_dims[1][1], stride=cnn_dims[1][2]),
        activation,
        nn.Conv2d(cnn_dims[1][0], cnn_dims[2][0], kernel_size=cnn_dims[2][1], stride=cnn_dims[2][2]),
        activation,
        nn.Flatten()
    )

    with no_grad():
        n_flatten = cnn(as_tensor(input_dim.sample()[None]).float()).shape[1]

    net = nn.Sequential(
        cnn,
        nn.Linear(n_flatten, fc_dims[0]),
        activation,
        nn.Linear(fc_dims[0], fc_dims[1]),
        activation
    )
    """
    ####################################################################################################################

    # """CHANGE FC DUELING LAYER OUTPUT DIM HERE""" ####################################################################
    fc_out_dim = 48
    ####################################################################################################################

    # """CHANGE OPTIMIZER HERE""" ######################################################################################
    optim_func = optim.Adam
    ####################################################################################################################

    # """CHANGE LOSS HERE""" ###########################################################################################
    loss_func = nn.SmoothL1Loss
    ####################################################################################################################

    return net, fc_out_dim, optim_func, loss_func

########################################################################################################################
