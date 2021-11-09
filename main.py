
'''Cluster slices within NIFTI based on perceptual similarity or Structural Similarity Index Measure'''

from clustering import PerceptualSimilarity
from clustering import SSIM


def main():

    cluster_vol = SSIM('someones_epi.nii.gz', 5) # k=5 clusters
    print (cluster_vol.return_samples())




if __name__ == '__main__':
    main()
