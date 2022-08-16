"""
Example for the training procedure with stable baselines3 and pytorch.
Both dependencies are not included in the `environment.yml` conda specification
and need to be installed manually.
"""
import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from SimpleWalk2D import SimpleWalk2DDynGoal


class TrainAndLoggingCallback(BaseCallback):
    
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok = True)
            
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
            
        return True

def main():

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)



    env = SimpleWalk2DDynGoal()



    env_name = 'SW2DDynGoal'

    CHECKPOINT_DIR = './train/train_' + env_name
    LOG_DIR = './train/log_' + env_name

    callback = TrainAndLoggingCallback(check_freq=10_000, save_path=CHECKPOINT_DIR)

    log_path = os.path.join('Training', 'Logs')

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_path,
        learning_rate=0.001,
        n_steps =256
        )
    logger.setLevel(logging.DEBUG)

    model.learn(
        total_timesteps=100_000,
        callback = callback
        )

    model.save('PPO')
    logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    main()
