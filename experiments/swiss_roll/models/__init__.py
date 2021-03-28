from src.models import ExactCMEProcess, VariationalCMEProcess, VariationalGP
from src.likelihoods import CMEProcessLikelihood, VBaggGaussianLikelihood
from src.utils import Registry
"""
Registry of experiment models
"""
MODELS = Registry()
TRAINERS = Registry()
PREDICTERS = Registry()


def build_model(cfg):
    model = MODELS[cfg['name']](**cfg)
    return model


def train_model(cfg):
    TRAINERS[cfg['name']](**cfg)


def predict(cfg):
    prediction = PREDICTERS[cfg['name']](**cfg)
    return prediction


from .exact_cme_process import *
from .variational_cme_process import *
from .linear_interpolation import *
from .vbagg import *