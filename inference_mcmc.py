from cgi import test
from tokenize import Double


from itertools import combinations









from src import HMC_Single_Star
if __name__=="__main__":
    hmc=HMC_Single_Star.HMC_Sampler()
    hmc.run_model()
    hmc.plot_profile()


