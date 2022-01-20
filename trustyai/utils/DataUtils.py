# pylint: disable = invalid-name, import-error
"""DataUtils module"""
from org.kie.kogito.explainability.utils import DataUtils as du

getMean = du.getMean
getStdDev = du.getStdDev
gaussianKernel = du.gaussianKernel
euclideanDistance = du.euclideanDistance
hammingDistance = du.hammingDistance
doublesToFeatures = du.doublesToFeatures
exponentialSmoothingKernel = du.exponentialSmoothingKernel
generateRandomDataDistribution = du.generateRandomDataDistribution


def generateData(mean, stdDeviation, size, jrandom):
    """Generate data"""
    return list(du.generateData(mean, stdDeviation, size, jrandom))


def perturbFeatures(originalFeatures, perturbationContext):
    """Perform perturbations on a fixed number of features in the given input."""
    return du.perturbFeatures(originalFeatures, perturbationContext)


def getLinearizedFeatures(originalFeatures):
    """Transform a list of eventually composite / nested features into a
    flat list of non composite / non nested features."""
    return du.getLinearizedFeatures(originalFeatures)


def sampleWithReplacement(values, sampleSize, jrandom):
    """Sample (with replacement) from a list of values."""
    return du.sampleWithReplacement(values, sampleSize, jrandom)
