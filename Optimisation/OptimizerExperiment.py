import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SimpleOpt2D import SimpleWalk2DDynGoal
from optimizer import ES

from typing import List


# allows to perform multiple runs and gathers their data for analysis
def main():
    logging.basicConfig(level=logging.INFO)

    results = pd.DataFrame(columns=['Run', 'FEs', 'Iteration', 'Fitness', 'Solution', 'Start', 'Steps to Goal'])

    runs = 50

    for i in range(runs):
        env = SimpleWalk2DDynGoal()

        es = ES(env)
        es.run()
        logging.info(f'best solution: {es.best.steps} starting at {es.best.start}')
        logging.info(f'total FEs: {es.function_evaluations}')
        logging.info(f'iteration and fitness: {es.best_fitness}')
        es.evaluate(es.best)
        results.loc[i] = [i, es.function_evaluations, es.best_fitness[-1][0], es.best_fitness[-1][1], es.best.steps,
                          es.best.start, es.short_individual]
        #env.render()
        env.close()

    results.hist(column='FEs')
    plt.show()


if __name__ == '__main__':
    main()
