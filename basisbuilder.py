#! /usr/bin/env python


from nutils import topology, element, function
import numpy


class BasisBuilder( object ):

  def __init__( self, ndims ):
    self.ndims = ndims

  def __mul__( self, other ):
    return ProductFunc( self, other )

  def build( self, topo ):
    assert isinstance( topo, topology.StructuredTopology )
    assert topo.ndims == self.ndims

    dofshape = self.getdofshape( topo.structure.shape )
    slices = self.getslices( topo.structure.shape )
    stdelems = self.getstdelems( topo.structure.shape )

    ndofs = numpy.product(dofshape)
    dofs = numpy.arange( ndofs ).reshape( dofshape )
    idx = numpy.frompyfunc( lambda *s: dofs[s].ravel(), len(slices), 1 )( *numpy.ix_( *slices ) )
    return function.function(
      fmap = dict( numpy.broadcast( topo.structure, stdelems ) ),
      nmap = dict( numpy.broadcast( topo.structure, idx ) ),
      ndofs = ndofs,
      ndims = topo.ndims )


class ProductFunc( BasisBuilder ):

  def __init__( self, func1, func2 ):
    assert isinstance( func1, BasisBuilder )
    assert isinstance( func2, BasisBuilder )
    self.func1 = func1
    self.func2 = func2
    BasisBuilder.__init__( self, ndims=func1.ndims+func2.ndims )

  def getdofshape( self, shape ):
    assert len(shape) == self.ndims
    return self.func1.getdofshape( shape[:self.func1.ndims] ) \
         + self.func2.getdofshape( shape[self.func1.ndims:] )

  def getslices( self, shape ):
    assert len(shape) == self.ndims
    return self.func1.getslices( shape[:self.func1.ndims] ) \
         + self.func2.getslices( shape[self.func1.ndims:] )

  def getstdelems( self, shape ):
    assert len(shape) == self.ndims
    return self.func1.getstdelems( shape[:self.func1.ndims] )[(Ellipsis,)+(numpy.newaxis,)*self.func2.ndims] \
         * self.func2.getstdelems( shape[self.func1.ndims:] )


class Spline( BasisBuilder ):

  def __init__( self, degree ):
    self.degree = degree
    BasisBuilder.__init__( self, ndims=1 )

  def getdofshape( self, (nelems,) ):
    return nelems + self.degree,

  def getslices( self, (nelems,) ):
    return [[ slice(i,i+self.degree+1) for i in range(nelems) ]]

  def getstdelems( self, (nelems,) ):
    return element.PolyLine.spline( degree=self.degree, nelems=nelems )
    

class ModSpline2( BasisBuilder ):

  def __init__( self, *ifaces ):
    self.ifaces = ifaces
    BasisBuilder.__init__( self, ndims=1 )

  def sorted_ifaces( self, nelems ):
    ifaces = numpy.sort([ nelems+iface if iface < 0 else iface for iface in self.ifaces ])
    assert ifaces[0] >= 2 and numpy.all( numpy.diff(ifaces) >= 4 ) and ifaces[-1] <= nelems-2
    return ifaces

  def getdofshape( self, (nelems,) ):
    return nelems + 2,

  def getslices( self, (nelems,) ):
    slices = [ slice(i,i+3) for i in range(nelems) ]
    for n in self.sorted_ifaces( nelems ):
      slices[n-2] = slice(n-2,n+2)
      slices[n+1] = slice(n,n+4)
    return [ slices ]

  def getstdelems( self, (nelems,) ):
    stdelems = element.PolyLine.spline( degree=2, nelems=nelems )
    for n in self.sorted_ifaces( nelems ):
      stdelems[n-2] = stdelems[n-2].extract( [[1,0,0,0],[0,1,0,0],[0,0,1,1]] )
      stdelems[n-1] = stdelems[n-1].extract( [[1,0,0],[0,1,1],[0,-1,1]] )
      stdelems[n+0] = stdelems[n+0].extract( [[1,1,0],[-1,1,0],[0,0,1]] )
      stdelems[n+1] = stdelems[n+1].extract( [[-1,1,0,0],[0,0,1,0],[0,0,0,1]] )
    return stdelems


# EXAMPLE
#
# domain, geom = mesh.rectilinear( [ xnodes, ynodes, znodes ] )
# basis1 = ( ModSpline2(2,-2) * Spline(1) * Spline(0) ).build( domain )
# basis2 = ( Spline(1) * ModSpline2(4) * Spline(0) ).build( domain )
