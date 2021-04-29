# pylint: disable = invalid-name, import-error
"""DataUtils module"""
from org.kie.kogito.explainability.utils import DataUtils as du
from java.util import ArrayList

generateData = du.generateData
getMean = du.getMean
getStdDev = du.getStdDev
gaussianKernel = du.gaussianKernel
euclideanDistance = du.euclideanDistance
hammingDistance = du.hammingDistance
doublesToFeatures = du.doublesToFeatures
exponentialSmoothingKernel = du.exponentialSmoothingKernel


def perturbFeatures(originalFeatures, perturbationContext):
    """Perform perturbations on a fixed number of features in the given input."""
    jlist = ArrayList(originalFeatures)
    return du.perturbFeatures(jlist, perturbationContext)
