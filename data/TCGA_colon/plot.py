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
    dirs = ["512_runs", "2k_runs", "6k_runs", "2k_downsampled_runs"]   
    image_size = ["512", "2k", "6k", "2k->512"]
    
    for i, dir in enumerate(dirs):
        fpr = np.load(f"{dir}/patch_fpr_0.npy")
        tpr = np.load(f"{dir}/patch_tpr_0.npy")
        plt.plot(fpr, tpr, label = f"Class: 0 Image: {image_size[i]}", linewidth=3)
    plt.legend()
    plt.title("Patch AUC Class 0")
    plt.savefig("patch_auc_0.pdf", format="pdf", bbox_inches="tight")
    plt.clf()

    for i, dir in enumerate(dirs):
        fpr = np.load(f"{dir}/patch_fpr_1.npy")
        tpr = np.load(f"{dir}/patch_tpr_1.npy")
        plt.plot(fpr, tpr, label = f"Class: 1 Image: {image_size[i]}", linewidth=3)
    plt.legend()
    plt.title("Patch AUC Class 1")
    plt.savefig("patch_auc_1.pdf", format="pdf", bbox_inches="tight")
    plt.clf()

    for i, dir in enumerate(dirs):
        fpr = np.load(f"{dir}/patch_fpr_2.npy")
        tpr = np.load(f"{dir}/patch_tpr_2.npy")
        plt.plot(fpr, tpr, label = f"Class: 2 Image: {image_size[i]}", linewidth=3)
    plt.legend()
    plt.title("Patch AUC Class 2")
    plt.savefig("patch_auc_2.pdf", format="pdf", bbox_inches="tight")
    plt.clf()

    for i, dir in enumerate(dirs):
        fpr = np.load(f"{dir}/pat_fpr_0.npy")
        tpr = np.load(f"{dir}/pat_tpr_0.npy")
        plt.plot(fpr, tpr, label = f"Class: 0 Image: {image_size[i]}", linewidth=3)
    plt.legend()
    plt.title("Patient AUC Class 0")
    plt.savefig("pat_auc_0.pdf", format="pdf", bbox_inches="tight")
    plt.clf()

    for i, dir in enumerate(dirs):
        fpr = np.load(f"{dir}/pat_fpr_1.npy")
        tpr = np.load(f"{dir}/pat_tpr_1.npy")
        plt.plot(fpr, tpr, label = f"Class: 1 Image: {image_size[i]}", linewidth=3)
    plt.legend()
    plt.title("Patient AUC Class 1")
    plt.savefig("pat_auc_1.pdf", format="pdf", bbox_inches="tight")
    plt.clf()

    for i, dir in enumerate(dirs):
        fpr = np.load(f"{dir}/pat_fpr_2.npy")
        tpr = np.load(f"{dir}/pat_tpr_2.npy")
        plt.plot(fpr, tpr, label = f"Class: 2 Image: {image_size[i]}", linewidth=3)
    plt.legend()
    plt.title("Patient AUC Class 2")
    plt.savefig("pat_auc_2.pdf", format="pdf", bbox_inches="tight")
    plt.clf()

