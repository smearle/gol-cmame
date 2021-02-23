import copy
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
import tqdm
from ribs.archives import GridArchive, SlidingBoundaryArchive
from ribs.emitters import ImprovementEmitter, OptimizingEmitter
from torch.nn import Conv2d

RENDER = False
CUDA = True

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data)
    #bias_init(module.bias.data)
    return module


class World(torch.nn.Module):
    def __init__(self,
                 num_proc=1, state=None):
        super(World, self).__init__()
        self.cuda = CUDA
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
                                         #nn.init.calculate_gain('relu')
                                         )

        conv_weights = [[[[1, 1, 1],
                          [1, 9, 1],
                          [1, 1, 1]]]]
        self.transition_rule = Conv2d(1, 1, 3, 1, 1, bias=False, padding_mode='circular')
        self.conv_init_(self.transition_rule)
        self.transition_rule.to(device)
        self.populate_cells(state=state)
        conv_weights = torch.FloatTensor(conv_weights)
        if self.cuda:
            conv_weights.cuda()
        conv_weights = conv_weights
        self.transition_rule.weight = torch.nn.Parameter(conv_weights, requires_grad=False)
        self.to(device)

    def populate_cells(self, state):
        if state is None:
            return
        self.state = torch.Tensor(state)
        if self.cuda:
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
        self.state = torch.where(self.state < self.prob_life, self.y1, self.y0).float()
#       self.builds.fill_(0)
#       self.failed.fill_(0)

#   def build_cell(self, x, y, alive=True):
#       if alive:
#           self.state[0, 0, x, y] = 1
#       else:
#           self.state[0, 0, x, y] = 0

    def _tick(self):
        self.state = self.forward(self.state)
    #print(self.state[0][0])

    def forward(self, x):
        with torch.no_grad():
            if self.cuda:
                x = x.cuda()
#           x = pad_circular(x, 1)
            x = x.float()
            #print(x[0])
            x = self.transition_rule(x)
            #print(x[0])
            # Mysterious leakages appear here if we increase the batch size enough.
            x = x.round() # so we hack them back into shape
            #print(x[0])
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
        x_out = x_out + abs(lif_2 * (x -13).float())
        assert (x_out >= 0).all() and (x_out <=1).all()
        #x_out = torch.clamp(x_out, 0, 1)
        return x_out

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self, rend_idx):
        rend_state = self.state[rend_idx].cpu()
        rend_state = np.vstack((rend_state * 1, rend_state * 1, rend_state * 1))
        rend_arr = rend_state

        rend_arr = rend_arr.transpose(2, 1, 0)
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
    def __init__(self, n_forward_frames=1, train_brute=True):
        if train_brute:
            self.train_brute = True
            self.map_width = 3
            n_forward_frames = 1
            self.prob_life = None
        else:
            self.map_width = 16
            self.prob_life = np.random.randint(0, 100)
        self.n_forward_frames = n_forward_frames
        self.max_step = 1 * self.n_forward_frames
        # how many frames do we let the NN predict on its own before allowing it to observe the actual game-state?
        # could increase this over the course of training to fine tune a model
        self.n_step = 0
        self.actions = None
        self.observation_space = self.action_space = gym.spaces.Box(0, 1, shape=(self.map_width, self.map_width))
        screen_width = 8*self.map_width
        #FIXME: remove need for this
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
        reward = 0

        if self.n_step != 0 and self.n_step % self.n_forward_frames == 0:
            if self.train_brute:
                loss = (abs(self.gol.state[:, 0, 1, 1].cpu() - actions[:, 0, 1, 1].cpu())).sum()
            else:
                loss = (abs(self.gol.state - actions.numpy())).sum()
            reward += -loss
            obs = self.gol.state
        else:
            obs = actions
        info = {}
        done = self.n_step >= self.max_step
        self.n_step += 1

        return obs, reward, done, info

    def render(self, mode=None):
        rend_idx = np.random.randint(self.gol.state.shape[0])
    #   rend_arr_1 = np.array(self.gol.state, dtype=np.uint8)
    #   rend_arr_1 = np.vstack((rend_arr_1 * 255, rend_arr_1 * 255, rend_arr_1 * 255))
    #   rend_arr_1 = rend_arr_1.transpose(1, 2, 0)
        rend_arr_1 = self.gol.render(rend_idx=rend_idx)
        cv2.imshow("Game of Life", rend_arr_1)
       #actions = self.actions.squeeze(0)
        actions = self.actions
        actions = actions[rend_idx]
        rend_arr_2 = np.array(actions, dtype=np.float)
        rend_arr_2 = np.vstack((rend_arr_2 , rend_arr_2 , rend_arr_2 ))
        rend_arr_2 = rend_arr_2.transpose(1, 2, 0)
        cv2.imshow("NN GoL", rend_arr_2)
        cv2.waitKey(1)



#class FlexArchive(GridArchive):
#    def __init__(self, *args, **kwargs):
#        self.score_hists = {}
#        super().__init__(*args, **kwargs)
#
#    def update_elite(self, behavior_values, obj):
#        index = self._get_index(behavior_values)
#        self.update_elite_idx(index, obj)
#
#    def update_elite_idx(self, index, obj):
#        if index not in self.score_hists:
#            self.score_hists[index] = []
#        score_hists = self.score_hists[index]
#        score_hists.append(obj)
#        obj = np.mean(score_hists)
#        self._solutions[index][2] = obj
#        self._objective_values[index] = obj
#
#        while len(score_hists) > 500:
#            score_hists.pop(0)
#
#    def add(self, solution, objective_value, behavior_values):
#        index = self._get_index(behavior_values)
#
#        if index in self.score_hists:
#            self.score_hists[index] = [objective_value]
#
#        return super().add(solution, objective_value, behavior_values)


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == torch.nn.Conv2d:
        torch.nn.init.orthogonal_(m.weight)
    if CUDA:
        m.cuda()


class NNGoL(torch.nn.Module):
    def __init__(self):
        self.m = 10
        super().__init__()
        self.l1 = Conv2d(1, 2 * self.m, 3, 1, 1, bias=True, padding_mode='circular')
        self.l2 = Conv2d(2 * self.m, self.m, 1, 1, 0, bias=True)
        self.l3 = Conv2d(self.m, 1, 1, 1, 0, bias=True)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)
        if CUDA:
            self.cuda()

    def forward(self, x):
        if CUDA:
            x.cuda()
        x = self.l1(x)
        x = torch.nn.functional.relu(x)
        x = self.l2(x)
        x = torch.nn.functional.relu(x)
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

    if seed is not None:
        env.seed(seed)

#   action_dim = env.action_space.shape
#   obs_dim = env.observation_space.shape
#   model = model.reshape((action_dim, obs_dim))


    total_reward = 0.0
    obs = env.reset(state=state)
    if RENDER:
        env.render()
    done = False
    act_sums = []

#   obs = torch.Tensor(obs).unsqueeze(0)

    while not done:
#       action = model @ obs  # Linear policy.
#       action = nn(torch.Tensor(obs))
        action = nn(obs)
        act_sums.append(action.sum())
        obs, reward, done, info = env.step(action)
        if RENDER:
            env.render()
        total_reward += reward

    # average loss per step per cell
    total_reward = total_reward / ((env.max_step / env.n_forward_frames) * env.map_width**2 * state.shape[0])
    bc = 100 * np.mean(act_sums) / (env.map_width**2)

    return total_reward, bc

BC = 2

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
        #FIXME: why?
        std = np.nanmean(stds)

        return mean, std

    elif BC == 2:
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


class EvolverCMAME():
    def __init__(self):
        self.epsilon = 1e-10
        self.n_forward_frames = 1
        init_nn = NNGoL()
        set_nograd(init_nn)
        init_weights = get_init_weights(init_nn)

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

        elif BC == 2:
            archive = GridArchive(
                    [200],
                    [(0, 100)],
                    )

        emitters = [
#               ImprovementEmitter(
                OptimizingEmitter(
                    archive,
                    init_weights,
                    0.05,
                    batch_size=30,
                    ) for _ in range(5)
                ]

#       env = gym.make("GoLImitator-v0")
        env = GoLImitator(train_brute=True)
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
        self.env = GoLImitator(n_forward_frames=self.n_forward_frames)
#       env.n_forward_frames = self.n_forward_frames


    def evolve(self, eval_elites=False):
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
        self.eval_elites = eval_elites
        init_nn = self.init_nn
        optimizer = self.optimizer
        archive = self.archive
        seed = self.seed
        start_time = time.time()
        total_itrs = 1000
#       init_states = np.random.randint(0, 2, (n_sims, 1, 16, 16))

        for itr in tqdm.tqdm(range(1, total_itrs + 1)):
            # Request models from the optimizer.
            sols = optimizer.ask()

            # Evaluate the models and record the objectives and BCs.
            objs, bcs = [], []

            for model in sols:

                m_objs = []  #, m_bcs = [], []
                m_bcs = []

                init_nn = set_weights(init_nn, model)

#               for i in range(n_sims):
                obj, bc = simulate(self.env, init_nn, model, seed, state=init_states)
                m_objs.append(obj)
                m_bcs.append(bc)

                obj = np.mean(m_objs)
                bc = np.mean(m_bcs)
                objs.append(obj)
                bcs.append([bc])


           #if not archive.empty:
           #    if self.eval_elites:
           #        # Re-evaluate elites in case we have made some change
           #        # prior to reloading which may affect fitness
           #        elites = [archive.get_random_elite() for _ in range(len(archive._solutions))]
           #    else:
           #        elites = [archive.get_random_elite() for _ in range(10)]

           #    for (model, score, behavior_values) in elites:
           #        init_nn = set_weights(init_nn, model)
           #        m_objs = []  #, m_bcs = [], []
           #        m_bcs = []

           #        for i in range(n_sims):
           #            obj, bc = simulate(self.env, init_nn, model, seed, state=init_states[i])
           #            m_objs.append(obj)
           #            m_bcs.append(bc)

           #        obj = np.mean(m_objs)
           #        behavior_values = np.mean(bc)

           #        if not self.eval_elites:
           #            # if re-evaluating elites, throw away old scores
           #            obj = (score + obj) / 2
           #        else:
           #            self.eval_elites = False
           #        archive.update_elite(behavior_values, obj)

           #       #    m_objs.append(obj)
           #       #bc_a = get_bcs(init_nn)
           #       #obj = np.mean(m_objs)
           #       #objs.append(obj)
           #       #bcs.append([bc_a])


           #df = archive.as_pandas(include_solutions=False)
           #max_score = df['objective'].max()



#          #if max_score == 0:
           #if not archive.empty:
           #    df = archive.as_pandas(include_solutions=True)
#          #    print('found perfect individual')
           #    idx = df['objective'].argmax()
           #    best_ind = df.iloc[df['objective'].argmax()]
           #    idxs = [int(best_ind['index_{}'.format(i)]) for i in range(2)]
           #    behavior_values = [best_ind['behavior_{}'.format(i)] for i in range(2)]
           #    model = archive._solutions[idxs[0], idxs[1]]
           #    score = max_score
           #    init_nn = set_weights(init_nn, model)

           #    while score == max_score:
           #        m_objs = []
           #        m_bcs = []

           #        for i in range(n_sims):
           #            obj, bc = simulate(self.env, init_nn, model, seed, state=init_states[i])
           #            m_objs.append(obj)
           #            m_bcs.append(bc)

           #        obj = np.mean(m_objs)
           #        behavior_values = np.mean(m_bcs)
           #        score = (score + obj) / 2
           #    archive.update_elite(np.array(behavior_values), score)

            # Send the results back to the optimizer.
         #  bcs = [bc[0] for bc in bcs]
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

        from ribs.visualize import grid_archive_heatmap

        plt.figure(figsize=(8, 6))
        grid_archive_heatmap(archive, vmin=-300, vmax=300)
        plt.gca().invert_yaxis()  # Makes more sense if larger velocities are on top.
        plt.ylabel("Impact y-velocity")
        plt.xlabel("Impact x-position")


class EvolverCMAES:
    def __init__(self):
        self.nn = nn = NNGoL()
        set_nograd(nn)
        init_weights = get_init_weights(nn)
        self.es = cma.CMAEvolutionStrategy(init_weights, 1,
                {
                'verb_disp': 1,
               #'verb_plot': 1,
                }
                )
        self.env = GoLImitator()

    def restore(self):
        self.env = GoLImitator()

    def evolve(self, eval_elite=False):
        n_sims = 20
        init_states = np.zeros((n_sims, 1, 16, 16))
        nn = self.nn
        es = self.es
        env = self.env

        while not self.es.stop():
            solutions = es.ask()
            objs = []

            for model in solutions:
                set_weights(nn, model)
                objs.append(-np.mean([simulate(env, nn, model, state=init_states[i]) for i in range(20)]))
            es.tell(solutions, objs)
            es.logger.add()
            es.disp()
        es.result_pretty()
        cma.plot()


SAVE_PATH = 'gol_cmame_brute_multi_0'

if __name__ == '__main__':
    try:
        evolver = pickle.load(open(SAVE_PATH, 'rb'))
        evolver.restore()
        evolver.evolve(eval_elites=False)
    except FileNotFoundError as e:
        print(e)
#       evolver = EvolverCMAES()
        evolver = EvolverCMAME()
        evolver.evolve()
