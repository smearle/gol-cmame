import argparse
import copy
import itertools
import os
import pickle
import time
from pdb import set_trace as T

import cma
import cv2
#import game_of_life
#from game_of_life.envs.env import GoLImitator
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import tqdm
from ribs.archives import GridArchive  # , SlidingBoundaryArchive
from ribs.emitters import ImprovementEmitter, OptimizingEmitter
from ribs.visualize import grid_archive_heatmap
from torch import ByteTensor, Tensor
from torch.nn import Conv2d, CrossEntropyLoss

import utils

#from _multiprocessing import Pool

#INFER = False
MAP_WIDTH = 128
global TRAIN_RENDER
TRAIN_RENDER = False
#SAVE_PATH = 'gol_cmame_brute_multi_2'


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data)
    # bias_init(module.bias.data)

    return module


class World(torch.nn.Module):
    def __init__(self,
                 num_proc=1, state=None):
        super(World, self).__init__()
        self.CUDA = CUDA
#       self.map_width = map_width
#       self.map_height = map_height
#       self.prob_life = prob_life / 100
#       self.num_proc = num_proc
#       state_shape = (num_proc, 1, map_width, map_height)
#       if self.cuda:
#           self.y1 = torch.ones(state_shape).cuda()
#           self.y0 = torch.zeros(state_shape).cuda()
#       else:
#           self.y1 = torch.ones(state_shape)
#           self.y0 = torch.zeros(state_shape)
        device = torch.device("cuda:0" if self.cuda else "cpu")
        self.conv_init_ = lambda m: init(m,
                                         torch.nn.init.dirac_, None,
                                         # nn.init.calculate_gain('relu')
                                         )

        conv_weights = [[[[1, 1, 1],
                          [1, 9, 1],
                          [1, 1, 1]]]]
        self.transition_rule = Conv2d(
            1, 1, 3, 1, 1, bias=False, padding_mode='circular')
        self.conv_init_(self.transition_rule)
        self.transition_rule.to(device)
        self.populate_cells(state=state)
        conv_weights = torch.FloatTensor(conv_weights)

        if self.cuda:
            conv_weights.cuda()
        conv_weights = conv_weights
        self.transition_rule.weight = torch.nn.Parameter(
            conv_weights, requires_grad=False)
        self.channels = 1
        self.get_neighbors = self.get_neighbors_map().to(device)
        self.to(device)

    def populate_cells(self, state):
        if state is None:
            return
        self.state = torch.Tensor(state.astype(np.bool))

        if self.CUDA:
            self.state.cuda()

        return

#       if self.cuda:
#           self.state = torch.cuda.FloatTensor(size=
#                                               (self.num_proc, 1, self.map_width, self.map_height)).uniform_(0, 1)
#         # self.builds = torch.cuda.FloatTensor(size=
#         #                                      (self.num_proc, 1, self.map_width, self.map_height)).fill_(0)
#         # self.failed = torch.cuda.FloatTensor(size=
#         #                                      (self.num_proc, 1, self.map_width, self.map_height)).fill_(0)
#       else:
#           self.state = torch.FloatTensor(size=
#                                          (self.num_proc, 1, self.map_width, self.map_height)).uniform_(0, 1)
#         # self.builds = torch.FloatTensor(size=
#         #                                 (self.num_proc, 1, self.map_width, self.map_height)).fill_(0)
#         # self.failed = torch.FloatTensor(size=
#         #                                 (self.num_proc, 1, self.map_width, self.map_height)).fill_(0)
#       self.state = torch.where(self.state < self.prob_life, self.y1, self.y0).float()

    def repopulate_cells(self):
        self.state.float().uniform_(0, 1)
        self.state = torch.where(
            self.state < self.prob_life, self.y1, self.y0).float()
#       self.builds.fill_(0)
#       self.failed.fill_(0)

#   def build_cell(self, x, y, alive=True):
#       if alive:
#           self.state[0, 0, x, y] = 1
#       else:
#           self.state[0, 0, x, y] = 0

    def _tick(self):
        self.step()
#       self.state = self.forward(self.state)
    # print(self.state[0][0])

    def step(self):
        state = self.state

        if self.cuda:
            neighbors = self.get_neighbors(state.cuda())[0, ...]
        else:
            # Get neighbor counts of cells
            neighbors = self.get_neighbors(state)[0, ...]

        # Alive cell with less than two neighbors should die
        rule1 = (neighbors < 2).type(Tensor)
        mask1 = (rule1 * state[0, ...]).type(ByteTensor)

        # Alive cell with more than two neighbors should die
        rule2 = (neighbors > 3).type(Tensor)
        mask2 = (rule2 * state[0, ...]).type(ByteTensor)

        # Dead cell with exactly three neighbors should spawn
        rule3 = (neighbors == 3).type(Tensor)
        mask3 = (rule3 * (1 - state[0, ...])).type(ByteTensor)

        # Update state
        state[0, mask1] = 0
        state[0, mask2] = 0
        state[0, mask3] = 1

        return state

    def get_neighbors_map(self):
        channels = self.channels
        neighbors_filter = Conv2d(channels, channels, 3, padding=1)
        neighbors_filter.weight = torch.nn.Parameter(Tensor([[[[1, 1, 1],
                                                               [1, 0, 1],
                                                               [1, 1, 1]]]]),
                                                     requires_grad=False)
        neighbors_filter.bias = torch.nn.Parameter(torch.Tensor(
            np.zeros(neighbors_filter.bias.shape)), requires_grad=False)

        if CUDA:
            neighbors_filter.cuda()

        return neighbors_filter

    def forward(self, x):
        with torch.no_grad():
            if self.cuda:
                x = x.cuda()
#           x = pad_circular(x, 1)
            x = x.float()
            # print(x[0])
            x = self.transition_rule(x)
            # print(x[0])
            # Mysterious leakages appear here if we increase the batch size enough.
            x = x.round()  # so we hack them back into shape
            # print(x[0])
            x = self.GoLu(x)

            return x

    def GoLu(self, x):
        '''
        Applies the Game of Life Unit activation function, element-wise:
                   _
        __/\______/ \_____
       0 2 4 6 8 0 2 4 6 8
        '''
        x_out = copy.deepcopy(x).fill_(0).float()
        ded_0 = (x >= 2).float()
        bth_0 = ded_0 * (x < 3).float()
        x_out = x_out + (bth_0 * (x - 2).float())
        ded_1 = (x >= 3).float()
        bth_1 = ded_1 * (x < 4).float()
        x_out = x_out + abs(bth_1 * (x - 4).float())
        alv_0 = (x >= 10).float()
        lif_0 = alv_0 * (x < 11).float()
        x_out = x_out + (lif_0 * (x - 10).float())
        alv_1 = (x >= 11).float()
        lif_1 = alv_1 * (x < 12).float()
        x_out = x_out + lif_1
        alv_2 = (x >= 12).float()
        lif_2 = alv_2 * (x < 13).float()
        x_out = x_out + abs(lif_2 * (x - 13).float())
        assert (x_out >= 0).all() and (x_out <= 1).all()
        #x_out = torch.clamp(x_out, 0, 1)

        return x_out

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self, rend_idx):
        rend_state = self.state[rend_idx].cpu()
        rend_state = np.vstack(
            (rend_state * 1, rend_state * 1, rend_state * 1))
        rend_arr = rend_state

        rend_arr = rend_arr.transpose(1, 2, 0)
        rend_arr = rend_arr.astype(np.uint8)
        rend_arr = rend_arr * 255

        return rend_arr
     #  cv2.imshow("Game of Life", rend_arr)

#       if self.record and not self.gif_writer.done:
#           gif_dir = ('{}/gifs/'.format(self.record))
#           im_dir = os.path.join(gif_dir, 'im')
#           im_path = os.path.join(im_dir, 'e{:02d}_s{:04d}.png'.format(self.gif_ep_count, self.num_step))
#           cv2.imwrite(im_path, rend_arr)

#           if self.gif_ep_count == 0 and self.num_step == self.max_step:
#               self.gif_writer.create_gif(im_dir, gif_dir, 0, 0, 0)
#               self.gif_ep_count = 0
#       cv2.waitKey(1)


class GoLImitator(gym.core.Env):
    ''' A gym environment in which the player is expected to learn the Game of Life. '''

    def __init__(self, n_forward_frames=1, train_brute=False):
        self.train_brute = train_brute

        if train_brute:
            self.map_width = 3
            n_forward_frames = 1
            self.prob_life = None
        else:
            self.map_width = MAP_WIDTH
#           self.prob_life = np.random.randint(0, 100)
        self.n_forward_frames = FORWARD_FRAMES
        self.max_step = 1 * self.n_forward_frames
        # how many frames do we let the NN predict on its own before allowing it to observe the actual game-state?
        # could increase this over the course of training to fine tune a model
        self.n_step = 0
        self.actions = None
        self.observation_space = self.action_space = gym.spaces.Box(
            0, 1, shape=(self.map_width, self.map_width))
        screen_width = 8*self.map_width
        # FIXME: remove need for this
        self.view_agent = False
        self.render_gui = RENDER
        self.gol = World()

        if self.render_gui:
            cv2.namedWindow("Game of Life", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Game of Life", screen_width, screen_width)
            cv2.namedWindow("NN GoL", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("NN GoL", screen_width, screen_width)

    def reset(self, state=None):
        self.n_step = 0
        self.gol.populate_cells(state)
        self.actions = self.gol.state
        obs = self.actions

        return obs

    def step(self, actions):
        self.actions = actions
        self.gol._tick()
#       print(self.n_step)
        reward = 0
#       loss_hist = []

      # if not INFER: # and self.n_step != 0:# and self.n_step % self.n_forward_frames == 0:
        # if self.train_brute:
        #    #loss = (abs(self.gol.state[:, 0, 1, 1].cpu() - actions[:, 0, 1, 1].cpu())).sum()
        #     loss = (abs(self.gol.state[:, 0, :, :].cpu() - actions[:, 0, :, :].cpu())).sum()
        # else:

        if self.n_step == self.n_forward_frames:
            if not INFER and self.train_brute:
                loss = (abs(self.gol.state[:, 0, 1, 1].cpu(
                ) - actions[:, 0, 1, 1].cpu().numpy())).sum()
            else:
                loss = (abs(self.gol.state - actions.cpu().numpy())).sum()
        else:
            loss = 0
#       loss_hist.append(loss)
        reward += -loss
        obs = self.gol.state
#       else:
#           obs = actions
        info = {}
        done = self.n_step >= self.max_step
        self.n_step += 1

        return obs, reward, done, info

    def render(self, mode=None):
        if INFER:
            rend_idx = 0
        else:
            rend_idx = np.random.randint(self.gol.state.shape[0])
    #   rend_arr_1 = np.array(self.gol.state, dtype=np.uint8)
    #   rend_arr_1 = np.vstack((rend_arr_1 * 255, rend_arr_1 * 255, rend_arr_1 * 255))
    #   rend_arr_1 = rend_arr_1.transpose(1, 2, 0)
        rend_arr_1 = self.gol.render(rend_idx=rend_idx)
        cv2.imshow("Game of Life", rend_arr_1)
       #actions = self.actions.squeeze(0)
        actions = self.actions
        actions = actions[rend_idx]
        rend_arr_2 = np.array(actions.cpu(), dtype=np.float)
        rend_arr_2 = np.vstack((rend_arr_2, rend_arr_2, rend_arr_2))
        rend_arr_2 = rend_arr_2.transpose(1, 2, 0)
        cv2.imshow("NN GoL", rend_arr_2)
        cv2.waitKey(1)


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == torch.nn.Conv2d:
        torch.nn.init.orthogonal_(m.weight)

    if CUDA:
        m.cuda()
        m.to('cuda:0')


class NNCellPredictor(torch.nn.Module):
    def __init__(self, n_chan=1):
        super().__init__()
        filter_mask = torch.Tensor(
            [[1, 1, 1],
             [1, 0, 1],
             [1, 1, 1]]
        )
        self.c1 = MaskedConv2d(1, 20, padding_mode='circular', filter_mask=filter_mask)
        self.c2 = nn.Conv2d(20, 10, kernel_size=1, stride=1, padding=0, bias=True)
        self.c3 = nn.Conv2d(10, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.apply(init_weights)

    def forward(self, x):
        x = self.c1(x)
        x = nn.functional.relu(x)
        x = self.c2(x)
        x = nn.functional.relu(x)
        x = self.c3(x)


        return x


class MaskedConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode, filter_mask):

        super().__init__()
        self.kernel_size = tuple(filter_mask.shape)
        self.filter_mask = nn.Parameter(filter_mask)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=1,
            bias=True,
            padding_mode=padding_mode,
        )
#       self.register_buffer('filter_mask', filter_mask)

    def forward(self, x):
        x = x.cuda()
        self._mask_conv_filter()

        return self.conv(x)

    def _mask_conv_filter(self):
        with torch.no_grad():
            self.conv.weight = nn.Parameter(
                self.conv.weight * self.filter_mask)


def ca_predictor_test():
    predictor = NNCellPredictor()
    predictor.cuda()
    optimizer = torch.optim.SGD(predictor.parameters(), lr=0.00001, momentum=0)
    cross_entropy_loss = nn.BCEWithLogitsLoss()
    gol = World()
    gol.cuda()

    prob_life = 0.2
    width = 32
    n_batch = 32
    n_chan = 1

    n_init_states = 100
    train_data = np.random.random((n_init_states, n_batch, n_chan, width, width)) < prob_life
    for j in range(10000):
        for k in range(n_init_states):
            err = 0
            loss = 0
            optimizer.zero_grad()
            np.random.shuffle(train_data)
            init_states = train_data[k]
            gol.populate_cells(init_states)
            for i in range(EPISODE_STEPS):
                obs = gol.state
                pred = predictor(obs)
                gol._tick()
                trg = gol.state
                pred = pred.permute(0, 2, 3, 1)
                pred = pred.view(n_batch * width ** 2, n_chan)
                trg = trg.permute(0, 2, 3, 1)
                trg = trg.view(n_batch * width ** 2, n_chan)
                err += abs(nn.functional.sigmoid(trg) - pred.cpu()).mean()
                loss += cross_entropy_loss(pred, trg.cuda())
            batch_size = n_init_states * EPISODE_STEPS
            loss = loss / batch_size
            err = err / batch_size

            print('err {}'.format(err))
            print('loss {}'.format(loss))
            loss.backward()


class NNGoL(torch.nn.Module):
    def __init__(self):
        self.m = 1
        super().__init__()
#       self.embed = Conv2d(1, 2, 1, 1, 0, bias=True)
        self.l1 = Conv2d(1, 2 * self.m, 3, 1, 1, bias=True,
                         padding_mode='circular')
        self.l2 = Conv2d(2 * self.m, self.m, 1, 1, 0, bias=True)
        self.l3 = Conv2d(self.m, 1, 1, 1, 0, bias=True)
#       self.layers = [self.embed, self.l1, self.l2, self.l3]
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

        if CUDA:
            self.cuda()
            self.to('cuda:0')

    def forward(self, x):
        if CUDA:
            x = x.cuda()
#       self.embed.cuda()
#       x = self.embed(x)
#       x = torch.nn.functional.relu(x)
#       x = x.repeat((1, 2, 1, 1))
#       x[:, 0, :, :] = (x[:, 0, :, :] + 1) % 2
        self.l1.cuda()
        x = self.l1(x)
        x = torch.nn.functional.relu(x)
        self.l2.cuda()
        x = self.l2(x)
#       x = torch.nn.functional.relu(x)
#       x = torch.nn
        self.l3.cuda()
        x = self.l3(x)
        x = torch.sigmoid(x)

        return x


def set_weights(nn, weights):
    with torch.no_grad():
        n_el = 0

        for layer in nn.layers:
            l_weights = weights[n_el:n_el + layer.weight.numel()]
            n_el += layer.weight.numel()
            l_weights = l_weights.reshape(layer.weight.shape)
            layer.weight = torch.nn.Parameter(torch.Tensor(l_weights))
            layer.weight.requires_grad = False
            b_weights = weights[n_el:n_el + layer.bias.numel()]
            n_el += layer.bias.numel()
            b_weights = b_weights.reshape(layer.bias.shape)
            layer.bias = torch.nn.Parameter(torch.Tensor(b_weights))
            layer.bias.requires_grad = False

    return nn


def simulate(env, nn, model, seed=None, state=None):
    """Simulates the lunar lander model.

    Args:
        env (gym.Env): A copy of the lunar lander environment.
        model (np.ndarray): The array of weights for the linear policy.
        seed (int): The seed for the environment.
    Returns:
        total_reward (float): The reward accrued by the lander throughout its
            trajectory.
        impact_x_pos (float): The x position of the lander when it touches the
            ground for the first time.
        impact_y_vel (float): The y velocity of the lander when it touches the
            ground for the first time.
    """

#   if seed is not None:
#       env.seed(seed)

#   action_dim = env.action_space.shape
#   obs_dim = env.observation_space.shape
#   model = model.reshape((action_dim, obs_dim))

    total_reward = 0.0
    obs = env.reset(state=state)

    if RENDER:
        env.render()
    done = False
    bcs = []

#   obs = torch.Tensor(obs).unsqueeze(0)

    while not done:
        #       action = model @ obs  # Linear policy.
        #       action = nn(torch.Tensor(obs))
        action = nn(obs)

        # mean and stddev output activation over batch

        if BC == 2:
            mean_act = action.mean(axis=-1).mean(axis=-1).cpu().numpy()

            bcs.append((mean_act.mean(), mean_act.std()))
        elif BC == 3:
            #           combinations = 2 ** (3 * 3)
            #           channels = 1
            #           structure_similarity = get_structure_similarity(combinations, channels)
            #           state = action
            #           entropy_1 = get_entropy(
            #               state, structure_similarity, combinations
            #           )

            #           print(action.shape)
            entropy = torch.distributions.Categorical(
                probs=action.view(action.shape[0], -1)).entropy()
#           print(entropy.shape)
            entropy_1 = entropy.mean().cpu() * 100
            entropy_2 = entropy.std().cpu() * 100
#           print(entropy_1, entropy_2)
            bcs.append((entropy_1, entropy_2))
        elif BC == 4:
            bcs.append(0.5)
        else:
            raise Exception
        obs, reward, done, info = env.step(action)

        if RENDER:
            env.render()
        total_reward += reward

#   print(total_reward)
    # average loss per step per cell
#   total_reward = total_reward / ((env.max_step / env.n_forward_frames) * env.map_width**2 * state.shape[0])

    # mean BCs over episode steps

    if BC == 4:
        bc = np.mean(bcs)
    else:
        bc = (np.mean(bcs[0]), np.mean(bcs[1]))
#   print(bc)

    return total_reward, bc


def get_bcs(nn):
    if BC == 1:
        param = list(nn.parameters())[0]
        eigval_sum = np.linalg.eig(param)[0].sum()
        bc_a = eigval_sum.real
        bc_b = eigval_sum.imag

        return (bc_a)

    elif BC == 0:
        means = []
        stds = []

        for param in nn.parameters():
            means.append(param.mean())
            stds.append(param.std())
        mean = np.mean(means)
        # FIXME: why?
        std = np.nanmean(stds)

        return mean, std

    else:
        raise Exception


def set_nograd(nn):
    for param in nn.parameters():
        param.requires_grad = False


def get_init_weights(nn):
    init_weights = []
#   n_par = 0
    for lyr in nn.layers:
        #       n_par += np.prod(lyr.weight.shape)
        #       n_par += np.prod(lyr.bias.shape)
        init_weights.append(lyr.weight.view(-1).cpu().numpy())
        init_weights.append(lyr.bias.view(-1).cpu().numpy())
    init_weights = np.hstack(init_weights)

    return init_weights


def get_neighbors_map(channels):
    neighbors_filter = Conv2d(channels, channels, 3, padding=1)
    neighbors_filter.weight = torch.nn.Parameter(Tensor([[[[1, 1, 1],
                                                           [1, 0, 1],
                                                           [1, 1, 1]]]]),
                                                 requires_grad=False)
    neighbors_filter.bias = torch.nn.init.zeros_(neighbors_filter.bias)

    return neighbors_filter


def get_structure_similarity(combinations, channels):
    tensors = torch.zeros([combinations, 1, 3, 3])
    elems = list(map(Tensor, itertools.product([0, 1], repeat=9)))

    for i, elem in enumerate(elems):
        tensors[i] = elem.view(1, channels, 3, 3)
    structure_similarity = Conv2d(
        channels, combinations, 3, stride=3, groups=channels)
    structure_similarity.weight = torch.nn.Parameter(
        tensors, requires_grad=False)
    structure_similarity.bias = torch.nn.init.zeros_(structure_similarity.bias)

    return structure_similarity


def get_entropy(state, structure_similarity, combinations):
    configs = structure_similarity(state)
    match_weights = structure_similarity.weight.view(combinations, -1).sum(-1)
    distribution = torch.zeros([combinations])

    # Smooth distribution incase configuration doesn't exist
    distribution.fill_(utils.EPSILON)

    for i, weight in enumerate(match_weights):
        config = configs[0][i]
        mask = config == weight
        distribution[i] += config[mask].shape[0]
    distribution /= distribution.sum()

    entropy = utils.entropy(distribution, 2)
    info = "Max Event Probability: {} | Entropy: {}".format(
        distribution.max(), entropy)
#   print(info)
    return entropy


class EvolverCMAME():
    def __init__(self):
        self.epsilon = 1e-10
        self.n_forward_frames = 10
        init_nn = NNGoL()
        set_nograd(init_nn)
        init_weights = get_init_weights(init_nn)

        env = GoLImitator(train_brute=TRAIN_BRUTE)

        if BC == 0:
            archive = GridArchive(
                [10, 10],
                [(-5, 5), (0, 5)],
            )
        elif BC == 1:
            archive = GridArchive(
                [200],
                [(-10, 10)],
            )

        # mean/std of output
        elif BC == 2:
            archive = GridArchive(
                #                   [50, 50],
                [50, 50],
                [(0, 1), (0, 1)]
            )

        # entropy of output
        elif BC == 3:
            archive = GridArchive(
                [50, 50],
                #            [(480, 490), (485, 486)]#
                [(0, 150)] * 2,
            )

        # just CMAES
        elif BC == 4:
            archive = GridArchive(
                [1],
                [(0, 1)],
            )

        emitters = [
            #               ImprovementEmitter(
            OptimizingEmitter(
                archive,
                init_weights,
                0.01,
                batch_size=30,
            ) for _ in range(5)
        ]

#       env = gym.make("GoLImitator-v0")
        self.seed = 420
        action_dim = env.action_space.shape
        obs_dim = env.observation_space.shape
        assert action_dim == obs_dim

        from ribs.optimizers import Optimizer

        optimizer = Optimizer(archive, emitters)
        self.optimizer = optimizer
        self.archive = archive
        self.init_nn = init_nn
        self.env = env

    def restore(self):
        #       self.env = gym.make("GoLImitator-v0", n_forward_frames=self.n_forward_frames)

        if INFER:
            n_forward_frames = EPISODE_STEPS
        else:
            n_forward_frames = self.n_forward_frames
        self.env = GoLImitator(
            n_forward_frames=n_forward_frames, train_brute=TRAIN_BRUTE)
#       env.n_forward_frames = self.n_forward_frames

    def infer(self):
        df = self.archive.as_pandas()
#       high_performing = df[df["behavior_1"] > 0.1].sort_values("objective", ascending=False)
        high_performing = df.sort_values("objective", ascending=False)
        rows = high_performing
        models = np.array(rows.loc[:, "solution_0":])
#       models = self.optimizer.ask()
        i = 0

        while True:
            model = self.archive.get_random_elite()[0]
#           model = models[np.random.randint(len(models))]
#           model = models[i]
            init_nn = set_weights(self.init_nn, model)
            init_states = (np.random.random(
                size=(1, 1, MAP_WIDTH, MAP_WIDTH)) < 0.2).astype(np.uint8)
            _, _ = simulate(self.env, init_nn, model,
                            self.seed, state=init_states)
            i += 1

            if i == len(models):
                i = 0

    def evolve(self, eval_elites=False):
        if TRAIN_BRUTE:
            init_states = np.zeros((2**9, 1, 3, 3))
            s_idxs = np.arange(2**9)

            for x in range(3):
                for y in range(3):
                    s = np.where((s_idxs // (2**(x*3+y))) % 2 == 0)
                    init_states[s, 0, x, y] = 1

            for i, s in enumerate(init_states):
                for j, t in enumerate(init_states):
                    if not i == j:
                        assert (s != t).any()
            n_sims = 2**9
        else:
            n_sims = 12
            init_states = np.random.choice(
                (0, 1), (n_sims, 1, MAP_WIDTH, MAP_WIDTH), p=(0.9, 0.1))

        self.eval_elites = eval_elites
        init_nn = self.init_nn
        optimizer = self.optimizer
        archive = self.archive
        seed = self.seed
        start_time = time.time()
        total_itrs = 10000

        for itr in tqdm.tqdm(range(1, total_itrs + 1)):
            # Request models from the optimizer.
            sols = optimizer.ask()

            # Evaluate the models and record the objectives and BCs.
            objs = []

            if BC == 4:
                bcs = np.zeros((len(sols), 1))
            else:
                bcs = np.zeros((len(sols), 2))

#           pool_in = []

            for (i, model) in enumerate(sols):

                m_objs = []  # , m_bcs = [], []
                m_bcs = []

                init_nn = set_weights(init_nn, model)

#               for i in range(n_sims):
#               pool_in.append(self.env, init_nn, model, seed, )
                obj, bc = simulate(self.env, init_nn, model,
                                   seed, state=init_states)
                m_objs.append(obj)

                if BC == 4:
                    bcs[i] = bc
                else:
                    bcs[i, :] = bc

                obj = np.mean(m_objs)
#               bc = np.mean(m_bcs)
                objs.append(obj)
#               bcs.append([bc])

#           print(np.min(bcs), np.max(bcs))
            optimizer.tell(objs, bcs)

            df = archive.as_pandas(include_solutions=True)

            # Logging.

            if itr % 1 == 0:
                elapsed_time = time.time() - start_time
                print(f"> {itr} itrs completed after {elapsed_time:.2f} s")
                print(f"  - Archive Size: {len(df)}")
                print(f"  - Max Score: {df['objective'].max()}")
                print(f"  - Mean Score: {df['objective'].mean()}")
                del(self.env)
                self.env = None
                pickle.dump(self, open(SAVE_PATH, 'wb'))
#               if df['objective'].max() > - self.epsilon:
#                   print("incrementing number of forward frames")
#                   self.n_forward_frames += 1
#                   self.eval_elites = True
                self.restore()

        plt.figure(figsize=(8, 6))
        grid_archive_heatmap(archive, vmin=-300, vmax=300)
        # Makes more sense if larger velocities are on top.
        plt.gca().invert_yaxis()
        plt.ylabel("Impact y-velocity")
        plt.xlabel("Impact x-position")


BC = 4  # basically CMAES

if __name__ == '__main__':
    opts = argparse.ArgumentParser(description='Game of Life')
    opts.add_argument(
        '-s',
        '--size',
        help='Size of world grid',
        default=700)
    opts.add_argument(
        '-p',
        '--prob',
        help='Probability of life in the initial seed',
        default=.05)
    opts.add_argument(
        '-tr',
        '--tick_ratio',
        help='Ticks needed to update on time step in game',
        default=1)
    opts.add_argument(
        '-re',
        '--record_entropy',
        help='Should record entropy of configurations',
        default=True,
    )
    opts.add_argument(
        '-i',
        '--infer',
        action='store_true',
    )
    opts.add_argument(
        '-e',
        '--exp_name',
        default='scrizzatch'
    )
    opts.add_argument(
        '--episode_steps',
        default=100,
    )
    opts.add_argument('--n_forward_frames',
                      default=1,
                      )
    opts = opts.parse_args()
    global INFER
    global EVO_DIR
    global render
    global CUDA
    global EPISODE_STEPS
    global TRAIN_BRUTE
    global FORWARD_FRAMES
    FORWARD_FRAMES = int(opts.n_forward_frames)
    EPISODE_STEPS = int(opts.episode_steps)
    INFER = opts.infer
    exp_name = '{}_{}frames'.format(opts.exp_name, opts.n_forward_frames)
    SAVE_PATH = os.path.join('experiments', exp_name)

    if INFER:
        RENDER = True
        CUDA = True
        TRAIN_BRUTE = False
    else:
        RENDER = TRAIN_RENDER
        CUDA = True
        TRAIN_BRUTE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ca_predictor_test()
    T()

    try:
        evolver = pickle.load(open(SAVE_PATH, 'rb'))
        evolver.render_gui = RENDER
        evolver.restore()

        if INFER:  # and not TRAIN_BRUTE:
            evolver.env.n_forward_frames = int(opts.episode_steps)
            evolver.infer()
        evolver.evolve(eval_elites=False)
    except FileNotFoundError as e:
        print(e)
#       evolver = EvolverCMAES()
        evolver = EvolverCMAME()
        evolver.evolve()
