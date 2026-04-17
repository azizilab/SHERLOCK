from .cvae import cVAE
from .base._base_trainer import BaseTrainer
from .base._pyro_trainer import PyroTrainer

# Registry mapping
MODEL_REGISTRY = {
    "cvae": { "model": cVAE, "trainer": BaseTrainer},

    # not implemented 
    "cpa-vae": {"model" : None, "trainer": None},
    "svae+": {"model" : None},
    "sa-vae": {"model" : None},
    "contrastive-vi": {"model" : None},
    "sengen": {"model" : None},
}

__all__ = ["MODEL_REGISTRY"]
