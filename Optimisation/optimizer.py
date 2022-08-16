"""
(mu+lambda)-Evolution Strategy with different mutation variants
"""
import logging
import random

import numpy as np

from SimpleOpt2D import SimpleWalk2DDynGoal

from typing import List


def calc_y_coord(x, x_old, y_old, step):
    """
    calculates a suitable y coordinate for a given x and a step length
    """
    y = np.sqrt(step - (x - x_old) ** 2) + y_old
    return y


class Solution:
    fitness_: float = 0

    def __init__(self,
                 start: np.ndarray,
                 steps: np.ndarray):
        self.start = start
        self.steps = steps

    def __copy__(self):
        copy = Solution(self.start.copy(), self.steps.copy())
        copy.fitness_ = self.fitness_
        return copy


class ES:
    """
    ES for SimpleWalk2DDynGoal environment
    """

    def __init__(self, env: SimpleWalk2DDynGoal, hard_bounds: float = False):
        """
        Initialize ES parameters
        """

        self.env = env
        self.has_hard_bounds = hard_bounds

        # early_stopping allows to stop the optimization process if a part of a solution (less steps than allowed) has
        # reached the goal
        self.early_stopping = True
        self.finished = False

        # for counting the number of function evaluations (= episodes)
        self.function_evaluations = 0

        """
        parameters of ES: iterations, steps, mu, lambda, mutation_sigma, p_cross
        """
        # number of iterations
        self.iterations = 100

        # number of steps/length of solution
        self.steps = 20

        # set number of parents (mu) and number of offspring (la or lambda)
        self.mu = 4
        self.la = 4

        # set sigma of gaussian distribution in mutation
        self.mutation_sigma = 0.1

        # crossover probability (only if crossover should be applied; requires mu and lambda >= 2)
        self.p_cross = 0.8

        """
        variables for ES components
        """
        # current iteration
        self.current_iter = 0

        # assume that environments are quadratic for now
        self.x_min = self.env.env_size()[0]
        self.x_max = self.env.env_size()[1]

        self.population = list()
        self.best = None  # best solution so far
        self.best_fitness = []  # best fitness values for each iteration; (iteration, fitness)
        self.short_individual = []  # steps until goal is reached if fewer than 20 steps

    def run(self):
        """
        runs ES for specified number of iterations
        """
        self.initialize()
        self.population = [self.evaluate(ind) for ind in self.population]
        while self.current_iter < self.iterations:
            selected = self.select(self.population, self.la)
            if np.random.uniform(0, 1) < self.p_cross:
                selected = self.recombine(selected)
            children = [self.mutate(parent.__copy__()) for parent in selected]
            children = [self.evaluate(child) for child in children]
            self.replace(children)
            if self.finished:
                break
            self.current_iter += 1

    def initialize(self):
        """
        creates initial solutions for each random starting point
        """
        # # random starting points for mu solutions
        # # self.starts = [(np.random.uniform(x_min, x_max),
        # #                 np.random.uniform(x_min, x_max))
        # #                for j in range(self.mu)]
        # starts = np.random.uniform(x_min, x_max, size=(self.mu, 2))

        starts = []
        for _ in range(self.mu):
            # random starting point
            start = np.random.uniform(self.x_min, self.x_max, size=2)
            starts.append(start)
            # random step sizes for all steps
            distances = list([np.random.uniform(0, self.env.max_speed(),
                                                size=self.steps)])
            # ratios to distribute distance on x and y
            ratios = np.random.uniform(0, 1, size=self.steps)
            # randomly move towards negative values
            neg_x = np.array([-1 if np.random.uniform(0, 1) > 0.5 else 1 for
                              _ in range(self.steps)])
            neg_y = np.array([-1 if np.random.uniform(0, 1) > 0.5 else 1 for
                              _ in range(self.steps)])
            # apply the step modifiers
            steps_x = distances * ratios * neg_x
            steps_y = distances * (1 - ratios) * neg_y
            if self.has_hard_bounds:
                self.enforce_bounds(start, steps_x, self.x_min, self.x_max)
                self.enforce_bounds(start, steps_y, self.x_min, self.x_max)
            steps = np.stack([steps_x, steps_y], axis=1).reshape((2, self.steps))
            self.population.append(Solution(start, steps))

        logging.info(f"Initial Solutions: {self.population}")
        logging.debug(f"Starting points: {starts}")

    def enforce_bounds(self, start, steps, x_min, x_max):
        for i in range(self.steps):
            cur_x = start[0] + np.sum(steps[:i])
            if cur_x > x_max or cur_x < x_min:
                steps[i] = steps[i] * -1

    def select(self, population, la):
        """
        randomly select solutions for mutations
        a solution can be selected several times (but is then mutated differently)
        """
        parents = np.random.choice(population, size=la)
        logging.debug(f"Selected solutions: {parents}")
        return parents

    def recombine(self, parents: List[Solution]):
        """
        recombine steps of two parent solutions to two child solutions using uniform crossover
        """
        pairs = []
        children = []
        for v, w in zip(parents[::2], parents[1::2]):
            pairs.append((v, w))

        for pair in pairs:
            for i in range(self.steps):
                if np.random.uniform(0, 1) < 0.5:
                    pair[0].steps.T[i], pair[1].steps.T[i] = pair[1].steps.T[i], pair[0].steps.T[i].copy()
            logging.debug(f"pair 0 {pair[0].steps}")
            logging.debug(f"pair 1 {pair[1].steps}")
            children.append(pair[0])
            children.append(pair[1])

        if self.has_hard_bounds:
            for child in children:
                self.enforce_bounds(child.start, child.steps[0], self.x_min, self.x_max)
                self.enforce_bounds(child.start, child.steps[1], self.x_min, self.x_max)

        return children

    def mutate(self, child: Solution):
        """
        mutates a solution's steps
        """
        # mutate x and y steps by drawing from normal distribution with mu being the previous value and mutation_sigma
        child.steps = np.random.normal(loc=child.steps.reshape(self.steps*2),
                                       scale=self.mutation_sigma,
                                       size=self.steps*2).reshape(2, self.steps)

        # calculate the distances that result from new x/y values; distances must be <= env.max_speed()
        dist = np.sqrt(child.steps[0]**2 + child.steps[1]**2)
        for i, d in enumerate(dist):
            # if step is too large, calculate new x and y for maximal step length
            if d > 1:
                child.steps[0][i] = np.random.uniform(0, self.env.max_speed())
                child.steps[1][i] = np.sqrt(self.env.max_speed()**2 - child.steps[0][i])
                # give x and y a 50% chance of being negative
                if np.random.uniform(0, 1) < 0.5:
                    child.steps[0][i] = - child.steps[0][i]
                if np.random.uniform(0, 1) < 0.5:
                    child.steps[1][i] = - child.steps[1][i]

        if self.has_hard_bounds:
            self.enforce_bounds(child.start, child.steps[0], self.x_min, self.x_max)
            self.enforce_bounds(child.start, child.steps[1], self.x_min, self.x_max)

        return child

    def replace(self, children: List[Solution]):
        """
        replaces solutions using + strategy, keeping best from mu + lambda
        """
        self.population.extend(children)
        self.population.sort(key=lambda i: i.fitness_)
        self.population = self.population[:self.mu]
        self.best = self.population[0].__copy__()
        self.best_fitness.append((self.current_iter, self.best.fitness_))

    def evaluate(self, child: Solution):
        """
        evaluate solutions using distance_to_opt from environment as optimization goal (minimization)
        """

        self.env.state[0:2] = child.start  # set starting point
        state = self.env.reset()
        # done = False
        score = 0
        distances = []
        logging.debug(f"state: {state}")
        self.function_evaluations += 1

        for action in child.steps.T:
            n_state, reward, done, info = self.env.step(action)
            score += reward  # remaining from RL; don't know if useful/necessary
            distances.append(info['distance_to_goal'])

        # the closest the optimizer managed to get to the target is counted
        # as its fitness
        child.fitness_ = np.min(distances)

        logging.info(f'Fitness of solution {child.steps} starting at {child.start}'
                     f' was {child.fitness_}')

        if self.early_stopping and self.env.ask_for_goal():
            logging.info(
                "Goal reached in iteration {}".format(self.current_iter))
            logging.info("Required FEs: {}".format(self.function_evaluations))
            self.short_individual = child.steps.T[:np.argmin(distances)]
            logging.info(f"Steps until Goal: {self.short_individual}")
            # self.env.render()
            self.finished = True

        return child


def main():
    logging.basicConfig(level=logging.DEBUG)

    env = SimpleWalk2DDynGoal()

    es = ES(env)
    es.run()
    logging.info(f'best solution: {es.best.steps} starting at {es.best.start}')
    logging.info(f'total FEs: {es.function_evaluations}')
    logging.info(f'iteration and fitness: {es.best_fitness}')
    es.evaluate(es.best)
    env.render()
    env.close()


if __name__ == '__main__':
    main()
