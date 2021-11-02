'''Cluster slices within NIFTI based on perceptual similarity or Structural Similarity Index Measure'''

from clustering import PerceptualSimilarityClustering
from clustering import SSIMClustering

def main():

    k = PerceptualSimilarityClustering('data/BraTS19_2013_2_1_t1.nii.gz', 5) # k=5 clusters
    k = k.return_samples()


if __name__ == '__main__':
    main()
