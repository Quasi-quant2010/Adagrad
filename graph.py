# -*- coding: utf-8 -*- 

import sys, gzip, codecs, optparse, os
from collections import defaultdict

import numpy as np
import pandas as pd
import cPickle, json

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#import seaborn as sns


if __name__ == '__main__':

    parser = optparse.OptionParser()
    parser.add_option("--data", dest="data", type=str, default="adagrad_iters.txt")
    (options, args) = parser.parse_args()


    plt.ylabel("LogLoss", fontsize=15)
    plt.xlabel("Iteration", fontsize=15)
    plt.minorticks_on()
    plt.grid(True)

    for idx, fname in enumerate(["Batch_L2.dat", "SGD_L2.dat"]):
        df = pd.read_csv(fname, names=['iteration', fname.split(".")[0]], delimiter="\t")
        #tmp.append(df)
        plt.plot(df["iteration"], df[fname.split(".")[0]], marker=idx+6, label=fname.split(".")[0])

    #plt.yscale("log")
    plt.legend(loc="best")
    plt.savefig("loss.png")
    plt.close()
