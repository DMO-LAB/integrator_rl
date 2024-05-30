import neptune
from datetime import datetime
import os
import pickle
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
load_dotenv()

from stable_baselines3.common.logger import configure

tmp_path = "/tmp/integrator_rl/logs"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


# neptune_run = neptune.init_run(
#             project="DMO-LAB/integrator-rl",
#             api_token=os.getenv("NEPTUNE_API_KEY"),
#             name="RL Cantera Training",
#             description="Training a reinforcement learning model with Cantera environment"
#         )

class NeptuneCallback(BaseCallback):
    def __init__(self, run, verbose=0):
        super(NeptuneCallback, self).__init__(verbose)
        self.run = run

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            self.run['train/reward'].log(self.locals['rewards'])
            self.run['train/episode_length'].log(self.locals['dones'].sum())
        return True

# Define custom callback for logging additional metrics
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.writer = SummaryWriter(log_dir='logs')

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            self.writer.add_scalar('reward', self.locals['rewards'], self.n_calls)
            self.writer.add_scalar('episode_length', self.locals['dones'].sum(), self.n_calls)
        return True

    def _on_training_end(self) -> None:
        self.writer.close()