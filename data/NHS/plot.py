import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
import numpy as np
import copy
import torch
import torch.nn as nn
from sklearn.metrics import auc, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':    
    dirs = ["512_run", "1440_runs"]
    image_size = ["512", "1440"]
 
    
    for i, dir in enumerate(dirs):
        fpr = np.load(f"{dir}/pat_fpr.npy")
        tpr = np.load(f"{dir}/pat_tpr.npy")
        plt.plot(fpr, tpr, label = f"Image: {image_size[i]}", linewidth=3)
    plt.legend()
    plt.title("Patient AUC")
    plt.savefig("pat_auc.pdf", format="pdf", bbox_inches="tight")
    #plt.savefig("pat_auc.png")
    plt.clf()


