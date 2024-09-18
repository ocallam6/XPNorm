import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


import hmac
import matplotlib.pyplot as plt
from src.NS_Single_Star import NS_Sampler
from src import NF_Cos_dist
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.colors as mcolors


ns=NS_Sampler()

import ultranest

sampler = ultranest.ReactiveNestedSampler(['a0','cos_dist','K','GK','BPK','RPK','JK','HK','x1K','x2K','x3K','x4k','x5k'], ns.log_likelihood, ns.priors)
x=sampler.run(show_status=True # how many times to go back and improve
)