import sys
import os
import numpy as np
import argparse
import os
import matplotlib as plt
import pickle
from matplotlib import pyplot as plt
import pandas as pd


if __name__ == '__main__' :        
    parser = argparse.ArgumentParser(description = "Train many model")
    parser.add_argument("-config", type = str, help = "<str> configuration file", required = True)
    pargs = parser.parse_args()

    print ("el argumento es este {}".format(pargs.config))

    plt.figure(figsize=(20,5))
    plt.suptitle(pargs.config)

    print("Plotting Acurracy")
    plt.subplot(1, 2, 2)
    plt.xlabel('# Epocas')
    plt.legend(loc="upper left", title="Accuracy", frameon=False)
    plt.plot([1,4,5,3], label='train_accuracy')
    plt.show()

    print ("showding datatable")
    dict = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
       "area": [8.516, 17.10, 3.286, 9.597, 1.221],
       "population": [200.4, 143.5, 1252, 1357, 52.98] }

    brics = pd.DataFrame(dict)
    print(brics)





