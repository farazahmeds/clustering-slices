
'''Cluster slices within NIFTI based on perceptual similarity or Structural Similarity Index Measure'''

from clustering import PerceptualSimilarity
from clustering import SSIM


def main():

    cluster_vol = SSIM('BraTS19_2013_2_1_t1ce.nii.gz', 24) # k=5 clusters
    cluster_vol = cluster_vol.return_samples()
    print (cluster_vol)



if __name__ == '__main__':
    main()
