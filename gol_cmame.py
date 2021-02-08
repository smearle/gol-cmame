
#import game_of_life
#from game_of_life.envs.env import GoLImitator
import gym
import time
import numpy as np
import tqdm
from ribs.archives import GridArchive
import matplotlib.pyplot as plt
#from ribs.emitters import ImprovementEmitter
from ribs.emitters import OptimizingEmitter
from torch.nn import Conv2d
import torch
from pdb import set_trace as T
import pickle
from ribs.archives import GridArchive, SlidingBoundaryArchive
import cv2


class World:
    ''' An implementation of Conway's Game of Life'''
    class LocationOccupied(RuntimeError): pass

    # Python doesn't have a concept of public/private variables

    def __init__(self, width, height, prob_life=20, env=None, state=None):
        self.env = env
        self.width = width
        self.height = height
        self.prob_life = prob_life
        self.tick = 0
        self.cells = np.full(shape=(width, height), fill_value=None)
        self.cached_directions = [
            [-1, 1],  [0, 1],  [1, 1], # above
            [-1, 0],           [1, 0], # sides
            [-1, -1], [0, -1], [1, -1] # below
        ]
        if state is not None:
            self.state = state
        else:
            raise Exception
            self.state = np.zeros(shape=(1, width, height), dtype=np.uint8)
        self.populate_cells()
        self.prepopulate_neighbours()

    def seed(self, seed=None):
        np.random.seed(seed)

    def _tick(self):
        state_changed = False
        # First determine the action for all cells
        for row in self.cells:
            for cell in row:
                alive_neighbours = self.alive_neighbours_around(cell)
                if cell.alive is False and alive_neighbours == 3:
                    cell.next_state = 1
                    state_changed = True
                elif alive_neighbours < 2 or alive_neighbours > 3:
                    if cell.alive:
                        state_changed = True
                    cell.next_state = 0
                    # FIXME: should be in env class
                    if self.env is not None and self.env.view_agent:
                        self.env.agent_builds[cell.x, cell.y] = 0
        self.state_changed = state_changed

        # Then execute the determined action for all cells
        for row in self.cells:
            for cell in row:
                if cell.next_state == 1:
                    cell.alive = True
                elif cell.next_state == 0:
                    cell.alive = False
                x, y = cell.x, cell.y
                self.state[0][x][y] = int(cell.alive)

        self.tick += 1

    # Implement first using string concatenation. Then implement any
    # special string builders, and use whatever runs the fastest
    def render(self):
        rendering = ''
        for y in list(range(self.height)):
            for x in list(range(self.width)):
                cell = self.cell_at(x, y)
                rendering += cell.to_char()
            rendering += "\n"
        return rendering

        # The following works but performs no faster than above
        # rendering = []
        # for y in list(range(self.height)):
        #     for x in list(range(self.width)):
        #         cell = self.cell_at(x, y)
        #         rendering.append(cell.to_char())
        #     rendering.append("\n")
        # return ''.join(rendering)

    # Python doesn't have a concept of public/private methods

    def populate_cells(self):
        for y in list(range(self.height)):
            for x in list(range(self.width)):
                alive = (np.random.randint(0, 100) <= self.prob_life)
                self.add_cell(x, y, alive)

    def repopulate_cells(self):
        ''' When resetting the env.'''
        for y in list(range(self.height)):
            for x in list(range(self.width)):
                alive = (np.random.randint(0, 100) <= self.prob_life)
                self.build_cell(x, y, alive)

    def prepopulate_neighbours(self):
        for row in self.cells:
            for cell in row:
                self.neighbours_around(cell)

    def add_cell(self, x, y, alive=False):
        cell = self.cell_at(x, y)
        if cell != None:
            self.state[0][x][y] = int(cell.alive)
            raise World.LocationOccupied
        cell = Cell(x, y, alive)
        self.cells[x, y] = cell
        self.state[0][x][y] = int(alive)
        return self.cell_at(x, y)

    def build_cell(self, x, y, alive=True):
        cell = Cell(x, y, alive)
        self.cells[x, y] = cell
        self.state[0][x][y] = int(alive)
        return self.cell_at(x, y)

    def cell_at(self, x, y):
        x = x % self.width
        y = y % self.width
        return self.cells[x][y]

    def neighbours_around(self, cell):
        if cell.neighbours is None:
            cell.neighbours = []
            for rel_x,rel_y in self.cached_directions:
                neighbour = self.cell_at(
                    cell.x + rel_x,
                    cell.y + rel_y
                )
                if neighbour is not None:
                    cell.neighbours.append(neighbour)

        return cell.neighbours

    # Implement first using filter/lambda if available. Then implement
    # foreach and for. Retain whatever implementation runs the fastest
    def alive_neighbours_around(self, cell):
        # The following works but is slower
        # filter_alive = lambda neighbour: neighbour.alive
        # return len(list(filter(filter_alive, neighbours)))

        alive_neighbours = 0
        for neighbour in self.neighbours_around(cell):
            if neighbour.alive:
                alive_neighbours += 1
        return alive_neighbours

    def set_state(self, state):
        pass

class Cell:

    def __init__(self, x, y, alive = False):
        self.x = x
        self.y = y
        self.alive = alive
        self.next_state = None
        self.neighbours = None

    def to_char(self):
        return 'o' if self.alive else ' '

class GoLImitator(gym.core.Env):
    ''' A gym environment in which the player is expected to learn the Game of Life. '''
    def __init__(self, n_forward_frames=1):
        self.map_width = 16
        self.prob_life = np.random.randint(0, 100)
        self.n_forward_frames = n_forward_frames
        self.max_step = 5 * self.n_forward_frames
        # how many frames do we let the NN predict on its own before allowing it to observe the actual game-state?
        # could increase this over the course of training to fine tune a model
        self.n_step = 0
        self.actions = None
        self.observation_space = self.action_space = gym.spaces.Box(0, 1, shape=(self.map_width, self.map_width))
        screen_width = 8*self.map_width
        #FIXME: remove need for this
        self.view_agent = False
        self.render_gui = True
        if self.render_gui:
            cv2.namedWindow("Game of Life", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Game of Life", screen_width, screen_width)
            cv2.namedWindow("NN GoL", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("NN GoL", screen_width, screen_width)


    def reset(self, state=None):
        self.max_step = 5 * self.n_forward_frames
        self.n_step = 0
        self.prob_life = np.random.randint(15, 85)
        self.gol = World(self.map_width, self.map_width, prob_life=self.prob_life, env=self, state=state) 
        self.actions = self.gol.state
        obs = self.actions

        return obs


    def step(self, actions):
        self.actions = actions
        self.gol._tick()
        reward = 0
        if self.n_step != 0 and self.n_step % self.n_forward_frames == 0:
            loss = (abs(self.gol.state - actions.numpy())).sum()
            reward += -loss
            obs = self.gol.state
            obs = torch.Tensor(obs).unsqueeze(0)
        else:
            obs = actions
        info = {}
        done = self.n_step >= self.max_step
        self.n_step += 1

        return obs, reward, done, info

    def render(self, mode=None):
        rend_arr_1 = np.array(self.gol.state, dtype=np.uint8)
        rend_arr_1 = np.vstack((rend_arr_1 * 255, rend_arr_1 * 255, rend_arr_1 * 255))
        rend_arr_1 = rend_arr_1.transpose(1, 2, 0)
        cv2.imshow("Game of Life", rend_arr_1)
        actions = self.actions.squeeze(0)
        rend_arr_2 = np.array(actions, dtype=np.float)
        rend_arr_2 = np.vstack((rend_arr_2 , rend_arr_2 , rend_arr_2 ))
        rend_arr_2 = rend_arr_2.transpose(1, 2, 0)
        cv2.imshow("NN GoL", rend_arr_2)
        cv2.waitKey(1)



class FlexArchive(GridArchive):
    def __init__(self, *args, **kwargs):
        self.score_hists = {}
        super().__init__(*args, **kwargs)

    def update_elite(self, behavior_values, obj):
        index = self._get_index(behavior_values)
        self.update_elite_idx(index, obj)

    def update_elite_idx(self, index, obj):
        if index not in self.score_hists:
            self.score_hists[index] = []
        score_hists = self.score_hists[index]
        score_hists.append(obj)
        obj = np.mean(score_hists)
        self._solutions[index][2] = obj
        self._objective_values[index] = obj
        while len(score_hists) > 500:
            score_hists.pop(0)

    def add(self, solution, objective_value, behavior_values):
        index = self._get_index(behavior_values)
        if index in self.score_hists:
            self.score_hists[index] = [objective_value]
        return super().add(solution, objective_value, behavior_values)


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.orthogonal(m.weight)

class NNGoL(torch.nn.Module):
    def __init__(self):
        self.m = 50
        super().__init__()
        self.l1 = Conv2d(1, 2 * self.m, 3, 1, 1, bias=True, padding_mode='circular')
        self.l2 = Conv2d(2 * self.m, self.m, 1, 1, 0, bias=True)
        self.l3 = Conv2d(self.m, 1, 1, 1, 0, bias=True)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, x): 
        x = self.l1(x)
        x = torch.nn.functional.relu(x)
        x = self.l2(x)
        x = torch.nn.functional.relu(x)
        x = self.l3(x)
        x = torch.nn.functional.sigmoid(x)

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
 
def simulate(env, nn, model, state, seed=None):
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
    done = False

    obs = torch.Tensor(obs).unsqueeze(0)
    while not done:
#       action = model @ obs  # Linear policy.
        action = nn(torch.Tensor(obs))
        obs, reward, done, info = env.step(action)
        env.render()
        total_reward += reward

    # average loss per step per cell
    total_reward = total_reward / ((env.max_step / env.n_forward_frames) * env.map_width**2)
    return total_reward

BC = 0

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

def set_nograd(nn):
    for param in nn.parameters():
        param.requires_grad = False

def get_init_weights(nn):
    init_weights = []
#   n_par = 0 
    for lyr in nn.layers:
#       n_par += np.prod(lyr.weight.shape)
#       n_par += np.prod(lyr.bias.shape)
        init_weights.append(lyr.weight.view(-1).numpy())
        init_weights.append(lyr.bias.view(-1).numpy())
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
            archive = FlexArchive(
                    [10, 10],
                    [(-5, 5), (0, 5)],
                    )
        elif BC == 1:
            archive = FlexArchive(
                    [200],
                    [(-10, 10)],
                    )

        emitters = [
#               ImprovementEmitter(
                OptimizingEmitter(
                    archive,
                    init_weights,
                    0.5,
                    batch_size=10,
                    ) for _ in range(1)
                ]

#       env = gym.make("GoLImitator-v0")
        env = GoLImitator()
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
        n_sims = 50
        self.eval_elites = eval_elites
        init_nn = self.init_nn
        optimizer = self.optimizer
        archive = self.archive
        seed = self.seed
        start_time = time.time()
        total_itrs = 500

        for itr in tqdm.tqdm(range(1, total_itrs + 1)):
            # Request models from the optimizer.
            sols = optimizer.ask()

            # Evaluate the models and record the objectives and BCs.
            objs, bcs = [], []
            for model in sols:

                m_objs = []  #, m_bcs = [], []

                init_nn = set_weights(init_nn, model)
                for _ in range(n_sims):
                    obj = simulate(self.env, init_nn, model, seed)
                    m_objs.append(obj)

                bc = get_bcs(init_nn)
                obj = np.mean(m_objs)
                objs.append(obj)
                bcs.append([bc])


            if not archive.empty:
                if self.eval_elites:
                    # Re-evaluate elites in case we have made some change
                    # prior to reloading which may affect fitness
                    elites = [archive.get_random_elite() for _ in range(len(archive._solutions))]
                else:
                    elites = [archive.get_random_elite() for _ in range(10)]
                for (model, score, behavior_values) in elites:
                    init_nn = set_weights(init_nn, model)
                    m_objs = []  #, m_bcs = [], []
                    for _ in range(n_sims):
                        obj = simulate(self.env, init_nn, model, seed)
                        m_objs.append(obj)
                    obj = np.mean(m_objs)
                    if not self.eval_elites:
                        # if re-evaluating elites, throw away old scores
                        obj = (score + obj) / 2
                    else:
                        self.eval_elites = False
                    archive.update_elite(behavior_values, obj)
                        
                   #    m_objs.append(obj)
                   #bc_a = get_bcs(init_nn)
                   #obj = np.mean(m_objs)
                   #objs.append(obj)
                   #bcs.append([bc_a])

                
            df = archive.as_pandas(include_solutions=False)
            max_score = df['objective'].max()



#           if max_score == 0:
            if not archive.empty:
                df = archive.as_pandas(include_solutions=True)
#               print('found perfect individual')
                idx = df['objective'].argmax()
                best_ind = df.iloc[df['objective'].argmax()]
                idxs = [int(best_ind['index_{}'.format(i)]) for i in range(2)]
                behavior_values = [best_ind['behavior_{}'.format(i)] for i in range(2)]
                model = archive._solutions[idxs[0], idxs[1]]
                score = max_score
                init_nn = set_weights(init_nn, model)
                while score == max_score:
                    m_objs = []
                    for _ in range(n_sims):
                        obj = simulate(self.env, init_nn, model, seed)
                        m_objs.append(obj)
                    score = (score + obj) / 2
           #    T()
                archive.update_elite(np.array(behavior_values), score)

            # Send the results back to the optimizer.
            bcs = [bc[0] for bc in bcs]
            optimizer.tell(objs, bcs)


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

import cma

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
                objs.append(-np.mean([simulate(env, nn, model, init_states[i]) for i in range(20)]))
            es.tell(solutions, objs)
            es.logger.add()
            es.disp()
        es.result_pretty()
        cma.plot()


SAVE_PATH = 'gol_cmaes_evolver_0'

if __name__ == '__main__':
    try:
        evolver = pickle.load(open(SAVE_PATH, 'rb'))
        evolver.restore()
        evolver.evolve(eval_elites=False)
    except FileNotFoundError as e:
        print(e)
        evolver = EvolverCMAES()
        evolver.evolve()

