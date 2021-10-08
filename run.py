from src.experiment import ExperimentHandler
from collections import namedtuple
import json

with open('./config.json') as jsonFile:
    train_config = json.load(jsonFile)
    

ConfigTruple = namedtuple('Config', train_config)
config = ConfigTruple(**train_config)

E = ExperimentHandler()
E.train_corruption(config)
E.eval_GCDC(config)
E.eval_corruption(config)
