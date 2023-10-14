from .agent import Agent
from .gpt2 import MyGPT2Model
from .custom_gpt2 import MyGPT2DynamicDropModel
from .sg import MyGPT2ModelWithSGLoss
from .unlikelihood import *

def load_model(args):
    model_name = args['models'][args['model']]['model_name']
    model = globals()[model_name](**args)
    agent = Agent(model, args)
    return agent
