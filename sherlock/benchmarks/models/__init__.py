from .cvae import cVAE, CVAETrainer

# Registry mapping
MODEL_REGISTRY = {
    "cvae": { "model": cVAE, "trainer": CVAETrainer,},

    # not implemented 
    "cpa-vae": None,
    "svae+": None,
    "sa-vae": None,
    "contrastive-vi": None,
    "sengen": None,
}

__all__ = ["MODEL_REGISTRY"]
