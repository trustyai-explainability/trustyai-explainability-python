# pylint: disable = invalid-name, import-error
"""DataUtils module"""
from org.kie.kogito.explainability.utils import DataUtils as du
from java.util import ArrayList

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
    jlist = ArrayList(originalFeatures)
    return du.perturbFeatures(jlist, perturbationContext)


def getLinearizedFeatures(originalFeatures):
    """Transform a list of eventually composite / nested features into a
    flat list of non composite / non nested features."""
    jlist = ArrayList(originalFeatures)
    return du.getLinearizedFeatures(jlist)


def sampleWithReplacement(values, sampleSize, jrandom):
    """Sample (with replacement) from a list of values."""
    jlist = ArrayList(values)
    return du.sampleWithReplacement(jlist, sampleSize, jrandom)
