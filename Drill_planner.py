import pandas as pd
import  matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sberpm
import pm4py



class Drill_Planner:


    def __init__(self,
                 dataholder,
                 miner_type='InductiveMiner',
                 nrows = None,

                 networkx_graph = False
                 ):

        self.data

        self.miner_type = miner_type

