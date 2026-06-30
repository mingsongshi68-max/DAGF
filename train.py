"""
Training script for DAGF
"""
from run import DAGF_run



if __name__ == '__main__':
  DAGF_run(model_name='DAGF', dataset_name='mosi', is_tune=False, seeds=[1111], model_save_dir="./pt",
         res_save_dir="./result", log_dir="./log", mode='train', is_training=True)
