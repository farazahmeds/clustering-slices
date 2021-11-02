import os
import lpips
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity as ssim
from sklearn import cluster
from sklearn.cluster import KMeans, SpectralClustering
import random
from abc import ABC, abstractmethod
from pathlib import Path


cwd = os.getcwd()


class PerceptualSimilarityClustering:
    def __init__(
        self, path, n_clusters
    ):
        self.path = path
        self.n_clusters = n_clusters

    def return_samples(self) -> 'list of lists, containing clusters':

        loss_fn_alex = lpips.LPIPS(net="alex")
        loss_fn_alex.cuda()

        path = self.path
        img = nib.load(path)
        img = img.get_fdata()  # get image array
        print (img.shape)
        x, y, z = img[:, :, :].shape
        total_slices_ = np.arange(z).tolist()  # get total slice no

        def perc_sim(img1, img2):
            x, y = img1.shape
            img1, img2 = torch.from_numpy(img1), torch.from_numpy(img2)
            img1, img2 = img1.float(), img2.float()
            img1, img2 = img1.expand(3, x, y), img2.expand(3, x, y)
            img1, img2 = img1.cuda(), img2.cuda()

            per_score = loss_fn_alex(img1, img2)
            per_score = per_score.tolist()
            per_score = np.concatenate(
                np.concatenate(np.concatenate(per_score))
            )  
            perscore = np.ndarray.item(per_score)
            per_score = round(perscore, 3)
            return per_score

        vol = []

        for slice in total_slices_:

            val = []  

            for i in range(img.shape[-1]): 
                score = perc_sim(img[:, :, slice], img[:, :, i])
                val.append(score)  
            vol.append(val)

        clustering = SpectralClustering(
            n_clusters=self.n_clusters,
            assign_labels="discretize",
            random_state=0,
            affinity="precomputed",
        ).fit(vol)
        n_classes = []

        for num in range(self.n_clusters):
            k = [
                i for i, j in enumerate(clustering.labels_) if j == num
            ]  # returns the slice_index for the slice belonging to jth class
            n_classes.append(
                k
            )  # populate 'n_classes=[]' with which slice index belongs to which class

        return n_classes


class SSIMClustering:
    def __init__(
        self, path, n_clusters
    ):
        super().__init__()
        self.path = path
        self.n_clusters = n_clusters

    def return_samples(self):

        img = Path(cwd, self.path)
        img = nib.load(img)

        img = img.get_fdata()  # get image array

        x, y, z = img[:, :, :].shape

        total_slices_ = np.arange(z).tolist()  # get total slice no

        vol = []

        for slice in total_slices_:

            val = []  # Results for SSIM

            for i in range(img.shape[-1]):  # i runs for entire volume size (155)
                k = ssim(
                    img[:, :, slice], img[:, :, i]
                )  # Perform SSIM between each slice and rest of the slices, example slice=1, i=1 SSIM between slice 1 and 1
                k = round(k, 3)  # round the SSIM to 3 places
                val.append(k)  # append val with kth SSIM

            vol.append(val) 

        clustering = SpectralClustering(
            n_clusters=self.n_clusters,
            assign_labels="discretize",
            random_state=0,
            affinity="precomputed",
        ).fit(vol)

        n_classes = []

        for num in range(self.n_clusters):
            k = [
                i for i, j in enumerate(clustering.labels_) if j == num
            ]  # returns the slice_index for the slice belonging to jth class
            n_classes.append(
                k
            )  # populate 'n_classes=[]' with which slice index belongs to which class

        return n_classes
