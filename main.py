
'''Cluster slices within NIFTI based on perceptual similarity or Structural Similarity Index Measure'''

from clustering import PerceptualSimilarity
from clustering import SSIM


def main():

    cluster_perc_sim = PerceptualSimilarity('data/someones_epi.nii.gz', n_clusters=5)
    print(cluster_perc_sim.return_samples())

    cluster_SSIM = SSIM('data/someones_epi.nii.gz', n_clusters=5)
    print(cluster_SSIM.return_samples())
    

if __name__ == '__main__':
    main()
