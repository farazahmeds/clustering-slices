import os
import lpips
import nibabel as nib
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from sklearn import cluster
from sklearn.cluster import KMeans, SpectralClustering
from pathlib import Path
from abc import ABC, abstractmethod

cwd = os.getcwd()


class Clustering(ABC):

    def __init__(self):
        pass

    def compute_total_slices(self):
        self.img = nib.load(self.path)
        self.img = self.img.get_fdata()
        self.nth_slice = self.img.shape[-1]
        self.total_slices = np.arange(self.nth_slice).tolist()

    def spectral_clustering(self):
        clustering = SpectralClustering(
            n_clusters=self.n_clusters,
            assign_labels="discretize",
            random_state=0,
            affinity="precomputed",
        ).fit(self.vol)

        n_classes = []
        for num in range(self.n_clusters):
            k = [
                i for i, j in enumerate(clustering.labels_) if j == num
            ]  # returns the slice_index for the slice belonging to jth class
            n_classes.append(
                k
            )  # populate 'n_classes=[]' with which slice index belongs to which class

        return n_classes

    @abstractmethod
    def return_samples(self):
        pass


class PerceptualSimilarity(Clustering):
    
    '''based on https://github.com/richzhang/PerceptualSimilarity'''

    def __init__(
            self, path, n_clusters
    ):
        super().__init__()
        self.path = path
        self.n_clusters = n_clusters

    def return_samples(self) -> 'list of lists, containing clusters':
        
        loss_fn_alex = lpips.LPIPS(net="alex")
        loss_fn_alex.cuda()

        super().compute_total_slices()

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

        self.vol = []

        for slice in self.total_slices:

            val = []

            for i in range(self.img.shape[-1]):
                score = perc_sim(self.img[:, :, slice], self.img[:, :, i])
                val.append(score)
            self.vol.append(val)

        return super().spectral_clustering()


class SSIM(Clustering):

    def __init__(
            self, path, n_clusters
    ):
        super().__init__()
        self.path = path
        self.n_clusters = n_clusters

    def return_samples(self):

        super().compute_total_slices()
        self.vol = []

        for slice in self.total_slices:

            val = []  # Results for SSIM

            for i in range(self.img.shape[-1]):  # i runs for entire volume size (155)
                k = ssim(
                    self.img[:, :, slice], self.img[:, :, i]
                )  # Perform SSIM between each slice and rest of the slices, example slice=1, i=1 SSIM between slice 1 and 1
                k = round(k, 3)  # round the SSIM to 3 places
                val.append(k)  # append val with kth SSIM

            self.vol.append(val)

        return super().spectral_clustering()
