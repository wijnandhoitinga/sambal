import sys, numpy
from distutils.version import LooseVersion

assert sys.version_info >= (3, 3)
assert LooseVersion(numpy.version.version) >= LooseVersion('1.8'), 'nutils requires numpy 1.8 or higher, got %s' % numpy.version.version

__all__ = []
