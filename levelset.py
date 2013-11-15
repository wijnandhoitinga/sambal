from finity import *

@util.withrepr
def SchwarzP ( ndims, L ):
  assert ndims in [2,3] and L > 0.
  L = float(L)
  return lambda x : function.cos(numpy.pi*x[0]/L)+function.cos(numpy.pi*x[1]/L)+1 if ndims==2 else function.cos(numpy.pi*x[0]/L)+function.cos(numpy.pi*x[1]/L)+function.cos(numpy.pi*x[2]/L)
