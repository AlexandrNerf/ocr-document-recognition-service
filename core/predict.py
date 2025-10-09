import hydra
import rootutils
import argparse
from hydra.utils import instantiate
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".core-root", pythonpath=True)
parser = argparse.ArgumentParser()

@hydra.main(config_path='config', config_name='config', version_base='1.2.0')
def run(config: DictConfig) -> None:
    instantiate(config['core']).run()

if __name__ == '__main__':
    run()
