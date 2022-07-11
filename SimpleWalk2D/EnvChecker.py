import logging
from SimpleWalk2D import SimpleWalk2DDynGoal

def main():
    logging.basicConfig(level=logging.DEBUG)

    env = SimpleWalk2DDynGoal()

    episodes = 1
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0 
        logging.debug('state: {}'.format(state))
        
        while not done:
            # env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            logging.debug('Steps taken: {}'.format(info['steps_taken']))
            logging.debug('distance to goal: {}'.format(info['distance_to_goal']))
            score+=reward
        logging.info('Episode:{}'.format(episode)) #, score))
        logging.info('Score: {}'.format(score))
        env.render()
    env.close()

    logging.debug('stabel_baselines3 env_checker')
    from stable_baselines3.common import env_checker
    env_checker.check_env(env)

if __name__ == '__main__':
    main()