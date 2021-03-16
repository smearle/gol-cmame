import argparse
import itertools
import copy
import utils
from torch import Tensor, ByteTensor
import pickle
import time
import os
from pdb import set_trace as T

import cma
import cv2
#import game_of_life
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from ribs.archives import GridArchive#, SlidingBoundaryArchive
from ribs.emitters import ImprovementEmitter, OptimizingEmitter
from ribs.visualize import grid_archive_heatmap
from utils import FlexArchive
from torch.nn import Conv2d
#from _multiprocessing import Pool

#INFER = False
MAP_WIDTH = 128
global TRAIN_RENDER
TRAIN_RENDER = False
#SAVE_PATH = 'gol_cmame_brute_multi_2'


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data)
    #bias_init(module.bias.data)
    return module



class Parity(gym.core.Env):
    ''' A gym environment in which the player is expected to learn the Game of Life. '''
    def __init__(self):
        self.n_inputs = N_INPUTS
        self.observation_space = gym.spaces.MultiBinary(self.n_inputs)
        self.action_space = gym.spaces.Discrete(2)
        self.render_gui = RENDER
        self.trg_nn = NNParityTrg()
        self.trg_nn.set_trg_weights()

    def reset(self, state=None):
        if state is None:
            obs = self.input = np.random.randint(0, 2, (64, self.n_inputs))
        else:
            obs = self.input = state

        return obs


    def step(self, actions):
        self.actions = actions
        self.label = (np.sum(self.input, axis=1) % 2 == 1).astype(int).reshape(-1, 1)
#       trg_loss = np.sum(abs(self.trg_nn(self.input).cpu().detach().numpy() - self.label))
#       assert trg_loss < 0.05
        loss = np.sum(abs(self.label - actions.cpu().numpy()))
        reward = -loss
        obs = None
        info = {}
        done = True

        return obs, reward, done, info

    def render(self, mode=None):
        if not hasattr(self, 'label'):
            return
        print('Inputs: {}\nLabel: {}\nOutput: {}'.format(self.input, self.label, self.actions))



def init_weights(m):
    if type(m) == torch.nn.Linear:
#       torch.nn.init.xavier_uniform_(m.weight)
#       torch.nn.init.normal_(m.weight, mean=1)
        m.weight.data.fill_(0)
        m.bias.data.fill_(0.0)

    if type(m) == torch.nn.Conv2d:
        torch.nn.init.orthogonal_(m.weight)
    if CUDA:
        m.cuda()
        m.to('cuda:0')


class NNParity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n_inputs = N_INPUTS
        self.layers = []
        for i in range(int(np.log2(n_inputs))):
            self.layers.append(torch.nn.Linear(int(n_inputs/2**i), int(n_inputs/2**i)))
            self.layers.append(torch.nn.Linear(int(n_inputs / 2 ** i), int(n_inputs / 2 **(i+1))))
        [setattr(self, 'l{}'.format(i), l) for (i, l) in enumerate(self.layers)]
        self.apply(init_weights)
        if CUDA:
            self.cuda()
            self.to('cuda:0')

    def forward(self, x):
        if CUDA:
            x = torch.Tensor(x).cuda()
        for l in self.layers:
            l.cuda()
            x = torch.sigmoid(l(x))

        return x


    def set_trg_weights(self):
        for i in range(len(self.layers) // 2):
            l0, l1 = self.layers[2*i], self.layers[2*i+1]
            l0.weight.data.fill_(0)
            l0.bias.data.fill_(0)
            l1.weight.data.fill_(0)
            l1.bias.data.fill_(0)
            for j in range(l0.weight.shape[0] // 2):

                pos = j * 2
                l0.weight[pos, pos] = 20
                l0.weight[pos, pos+1] = 20
                l0.bias[pos] = -10
                l0.weight[pos+1, pos] = -20
                l0.weight[pos+1, pos+1] = -20
                l0.bias[pos+1] = 30
                l1.weight[j, pos] = 20
                l1.weight[j, pos+1] = 20
                l1.bias[j] = -30

    def perturb_weights(self, scale):
#       pass
        for l in self.layers:
            l.weight.data = l.weight.data.cpu() + np.random.normal(0, scale, l.weight.data.shape)
            l.bias.data = l.bias.data.cpu() + np.random.normal(0, scale, l.bias.data.shape)

class NNParityTrg(NNParity):
    def __init__(self):
        super().__init__()




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

        obs, reward, done, info = env.step(action)
        if RENDER:
            env.render()
        total_reward += reward

    #   print(total_reward)
    # average loss per step per cell
    #   total_reward = total_reward / ((env.max_step / env.n_forward_frames) * env.map_width**2 * state.shape[0])

    # mean BCs over episode steps
    bc = (0)
    #   print(bc)

    return total_reward, bc

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

class EvolverCMAME():
    def __init__(self):
        self.n_sims = 10
        self.epsilon = 1e-10
        self.n_forward_frames = 10
        self.n_inputs = 128
        init_nn = NNParity()
        set_nograd(init_nn)
        init_nn.set_trg_weights()
        init_nn.perturb_weights(PERTURB)
        init_weights = get_init_weights(init_nn)

        env = Parity()

        archive = FlexArchive(
            [1],
            [(0, 1)],
            )

        emitters = [
           #ImprovementEmitter(
            OptimizingEmitter(
                archive,
                init_weights,
                0.05,
                batch_size=30,
            ) for _ in range(5)
        ]

        #       env = gym.make("Parity-v0")
        self.seed = 420

        from ribs.optimizers import Optimizer

        optimizer = Optimizer(archive, emitters)
        self.optimizer = optimizer
        self.archive = archive
        self.init_nn = init_nn
        self.env = env

    def restore(self):
        #       self.env = gym.make("Parity-v0", n_forward_frames=self.n_forward_frames)
        if self.env is None:
            self.env = Parity()
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
            _, _ = simulate(self.env, init_nn, model, self.seed)
            i += 1
            if i == len(models):
                i = 0


    def evolve(self, eval_elites=False):
#       if TRAIN_BRUTE:
#           init_states = np.zeros((2**9, 1, 3, 3))
#           s_idxs = np.arange(2**9)
#           for x in range(3):
#               for y in range(3):
#                   s = np.where((s_idxs // (2**(x*3+y))) % 2 == 0)
#                   init_states[s, 0, x, y] = 1
#           for i, s in enumerate(init_states):
#               for j, t in enumerate(init_states):
#                   if not i == j:
#                       assert (s != t).any()
#           n_sims = 2**9
#       else:
#           n_sims = 12
#           init_states = np.random.choice((0, 1), (n_sims, 1, MAP_WIDTH, MAP_WIDTH), p=(0.9, 0.1))

        self.eval_elites = eval_elites
        init_nn = self.init_nn
        optimizer = self.optimizer
        archive = self.archive
        seed = self.seed
        start_time = time.time()
        total_itrs = 10000

        init_states = np.random.randint(0, 2, (self.n_sims, 64, N_INPUTS))
        for itr in tqdm.tqdm(range(1, total_itrs + 1)):
            # Request models from the optimizer.
            sols = optimizer.ask()

            # Evaluate the models and record the objectives and BCs.
            objs, bcs = [], []
            bcs = np.zeros((len(sols), 1))

            #           pool_in = []

            for (i, model) in enumerate(sols):

                m_objs = []  #, m_bcs = [], []
                m_bcs = []

                init_nn = set_weights(init_nn, model)

                for j in range(self.n_sims):
                    obj, bc = simulate(self.env, init_nn, model, seed, state=init_states[j])
                    m_bcs.append(bc)
                    m_objs.append(obj)
                bcs[i, :] = bc

                obj = np.mean(m_objs)
                bc = np.mean(m_bcs)
                objs.append(obj)
            #               bcs.append([bc])

            #           print(np.min(bcs), np.max(bcs))
            #           print(bcs)
            optimizer.tell(objs, bcs)

            if eval_elites and not archive.empty:
                # Re-evaluate elites in case we have made some change
                # prior to reloading which may affect fitness
                elites = [archive.get_random_elite() for _ in range(len(archive._solutions))]
                for (model, score, behavior_values) in elites:
                    init_nn = set_weights(init_nn, model)
                    m_objs = []  #, m_bcs = [], []
                    for j in range(self.n_sims):
                        obj = simulate(self.env, init_nn, model, seed, state=init_states[j])
                        m_objs.append(obj)
                    obj = np.mean(m_objs)
                    archive.update_elite(behavior_values, obj)

                   #    m_objs.append(obj)
                   #bc_a = get_bcs(init_nn)
                   #obj = np.mean(m_objs)
                   #objs.append(obj)
                   #bcs.append([bc_a])


            df = archive.as_pandas(include_solutions=False)
            max_score = df['objective'].max()

            df = archive.as_pandas(include_solutions=True)

            # Logging.

            if itr % 1 == 0:
                elapsed_time = time.time() - start_time
                print(f"> {itr} itrs completed after {elapsed_time:.2f} s")
                print(f"  - Archive Size: {len(df)}")
                print(f"  - Max Score: {df['objective'].max()}")
                print(f"  - Mean Score: {df['objective'].mean()}")
            if itr % 10 == 0:
                env = self.env
                self.env = None
                pickle.dump(self, open(SAVE_PATH, 'wb'))
                #               if df['objective'].max() > - self.epsilon:
                #                   print("incrementing number of forward frames")
                #                   self.n_forward_frames += 1
                #                   self.eval_elites = True
                self.env = env
                self.restore()


        plt.figure(figsize=(8, 6))
        grid_archive_heatmap(archive, vmin=-300, vmax=300)
        plt.gca().invert_yaxis()  # Makes more sense if larger velocities are on top.
        plt.ylabel("Impact y-velocity")
        plt.xlabel("Impact x-position")



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
        default='parity_0'
    )
    opts.add_argument(
        '--n_inputs',
        default=16,
    )
    opts.add_argument(
        '--perturb',
        default=0.1
    )
    opts = opts.parse_args()
    global INFER
    global EVO_DIR
    global render
    global CUDA
    global N_INPUTS
    global TRAIN_BRUTE
    PERTURB = float(opts.perturb)
    N_INPUTS = int(opts.n_inputs)
    INFER = opts.infer
    SAVE_PATH = os.path.join('experiments', 'n_in_{}_perturb_{}_{}'.format(N_INPUTS, PERTURB, opts.exp_name))
    if INFER:
        RENDER = True
        CUDA = True
        TRAIN_BRUTE = False
    else:
        RENDER = TRAIN_RENDER
        CUDA = True
        TRAIN_BRUTE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        evolver = pickle.load(open(SAVE_PATH, 'rb'))
        evolver.render_gui = RENDER
        evolver.restore()
        if INFER:# and not TRAIN_BRUTE:
            evolver.infer()
        evolver.evolve(eval_elites=False)
    except FileNotFoundError as e:
        print(e)
        #       evolver = EvolverCMAES()
        evolver = EvolverCMAME()
        evolver.evolve()
