#! /usr/bin/env python


from nutils import topology, element, function, plot, util, mesh
import numpy


def objvec( items ):
  items = tuple(items)
  A = numpy.empty( len(items), dtype=object )
  A[...] = items
  return A


class BasisBuilder( object ):

  def __init__( self, ndims ):
    self.ndims = ndims

  def __mul__( self, other ):
    return ProductFunc( self, other )

  def build( self, topo ):
    assert isinstance( topo, topology.StructuredTopology )
    assert topo.ndims == self.ndims

    dofshape = self.getdofshape( topo.structure.shape )
    slices = map( objvec, self.getslices( topo.structure.shape ) )
    stdelems = self.getstdelems( topo.structure.shape )

    ndofs = numpy.product(dofshape)
    dofs = numpy.arange( ndofs ).reshape( dofshape )
    idx = objvec( dofs[s] for s in slices[0] ) if self.ndims == 1 \
     else numpy.frompyfunc( lambda *s: dofs[numpy.ix_(*s)].ravel(), len(slices), 1 )( *numpy.ix_( *slices ) )

    return function.function(
      fmap = { elem.transform: ((funcs,None),) for elem, funcs in numpy.broadcast( topo.structure, stdelems ) },
      nmap = { elem.transform: dofs for elem, dofs in numpy.broadcast( topo.structure, idx ) },
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
    return numpy.asarray( self.func1.getstdelems( shape[:self.func1.ndims] ), dtype=object )[(Ellipsis,)+(numpy.newaxis,)*self.func2.ndims] \
         * numpy.asarray( self.func2.getstdelems( shape[self.func1.ndims:] ), dtype=object )


class Spline( BasisBuilder ):

  def __init__( self, degree, rmfirst=False, rmlast=False, periodic=False ):
    self.degree = degree
    assert not periodic or not rmfirst and not rmlast
    self.rmfirst = rmfirst
    self.rmlast = rmlast
    self.periodic = periodic
    BasisBuilder.__init__( self, ndims=1 )

  def getdofshape( self, (nelems,) ):
    ndofs = nelems if self.periodic else nelems + self.degree - self.rmfirst - self.rmlast
    return ndofs,

  def getslices( self, (nelems,) ):
    if self.rmlast:
      N, = self.getdofshape( [nelems] )
    else:
      N = numpy.inf
    slices = [ numpy.arange(max(0,i),min(N,i+self.degree+1)) for i in numpy.arange(nelems)-self.rmfirst ]
    if self.periodic:
      idofs = numpy.arange(nelems+self.degree) % nelems
      slices[-self.degree:] = [ idofs[s] for s in slices[-self.degree:] ]
    return slices,

  def getstdelems( self, (nelems,) ):
    stdelems = element.PolyLine.spline( degree=self.degree, nelems=nelems, periodic=self.periodic )
    if self.rmfirst:
      stdelems[0] = stdelems[0].extract( numpy.eye(stdelems[0].nshapes)[:,1:] )
    if self.rmlast:
      stdelems[-1] = stdelems[-1].extract( numpy.eye(stdelems[-1].nshapes)[:,:-1] )
    return stdelems


class BSpline( BasisBuilder ):

  def __init__( self, degree, knotvalues=None, knotmultiplicities=None ):
    self.degree = degree
    self.knotvalues = knotvalues
    self.knotmultiplicities = knotmultiplicities
    BasisBuilder.__init__( self, ndims=1 )

  def getdofshape( self, (n,) ):
    p = self.degree
    m = self._knotmultiplicities(n)
    return sum(m[1:-1]) + p+1,

  def getslices( self, (n,) ):
    p = self.degree
    m = self._knotmultiplicities(n)
    offsets = numpy.cumsum(m[:-1]) - (p+1)
    return [ offset + numpy.arange(p+1) for offset in offsets ],

  def getstdelems( self, (n,) ):
    p = self.degree
    k = self._knotvalues(n)
    m = self._knotmultiplicities(n)
    km = numpy.concatenate([ [ki] * mi for ki, mi in zip(k,m) ])
    offsets = numpy.cumsum(m[:-1]) - p
    return [ element.PolyLine( self._localsplinebasis( km[offset:offset+2*p] - km[offset] ) ) for offset in offsets ]

  def _knotmultiplicities( self, n ):
    p = self.degree
    m = numpy.ones( n+1, dtype=int ) if self.knotmultiplicities is None else numpy.array( self.knotmultiplicities )
    m[0] = m[-1] = p+1
    assert len(m) == n+1, 'knot vector size does not match the topology size'
    assert min(m) > 0 and max(m) <= p+1, 'incorrect multiplicity encountered'
    return m

  def _knotvalues( self, n ):
    k = numpy.arange(n+1) if self.knotvalues is None else numpy.array( self.knotvalues )
    assert len(k) == n+1, 'knot vector size does not match the topology size'
    return k

  @staticmethod
  def _localsplinebasis( lknots ):
    lknots = numpy.asarray( lknots, dtype=float )
    p = len(lknots) // 2
    assert len(lknots) == 2*p, 'expected even number of local knots'
    #Based on Algorithm A2.2 Piegl and Tiller
    N = [ numpy.poly1d([1.]) ]
    if p > 0:
      assert numpy.all( lknots[:-1]-lknots[1:] < numpy.spacing(1) ), 'local knot vector should be non-decreasing'
      assert lknots[p]-lknots[p-1] > numpy.spacing(1), 'element size should be positive'
      xi = numpy.poly1d( [lknots[p]-lknots[p-1],lknots[p-1]] )
      left  = []
      right = []
      for i in range(p):
        left.append( xi - lknots[p-i-1] )
        right.append( -xi + lknots[p+i] )
        saved = 0.
        for r in range(i+1):
          temp = N[r] / (lknots[p+r]-lknots[p+r-i-1])
          N[r] = saved + right[r]*temp
          saved = left[i-r]*temp
        N.append( saved )
    assert all( Ni.order == p for Ni in N )
    return numpy.array([Ni.coeffs for Ni in N]).T[::-1]


class Mod( BasisBuilder ):

  def __init__( self, bbuilder, vertex ):
    assert bbuilder.ndims == 1
    self.bbuilder = bbuilder
    self.vertex = vertex
    BasisBuilder.__init__( self, ndims=1 )

  def getdofshape( self, shape ):
    return self.bbuilder.getdofshape( shape )

  def getmoddofs( self, shape ):
    slices, = self.bbuilder.getslices( shape )
    s1 = slices[self.vertex-1]
    s2 = slices[self.vertex]
    isect = set(s1) & set(s2)
    assert len(isect) == 2
    return sorted(isect)

  def getslices( self, shape ):
    slices, = self.bbuilder.getslices( shape )
    n1, n2 = self.getmoddofs( shape )
    return [ numpy.hstack([s,n1]) if n2 in s and n1 not in s else numpy.hstack([s,n2]) if n1 in s and n2 not in s else s for s in slices ],

  def getstdelems( self, shape ):
    stdelems = self.bbuilder.getstdelems( shape )
    slices, = self.bbuilder.getslices( shape )
    n1, n2 = self.getmoddofs( shape )
    modstdelems = []
    for stdelem, s in zip( stdelems, slices ):
      i1 = s == n1
      i2 = s == n2
      if not i1.any() and not i2.any():
        modstdelems.append( stdelem )
        continue
      A = numpy.eye( stdelem.nshapes )
      if not i1.any() or not i2.any():
        A = numpy.hstack([ A, A[:,i1|i2] ])
        A[:,i2] *= -1
      else:
        A[:,i1|i2] = numpy.hstack([ A[:,i1]+A[:,i2], A[:,i1]-A[:,i2] ])
      modstdelems.append( stdelem.extract( A ) )
    return modstdelems


def example():

  verts = 0,1,3,4,10
  domain, geom = mesh.rectilinear( [ verts ] )
  basis = BSpline( degree=2, knotvalues=verts, knotmultiplicities=[1,2,3,1,1] ).build(domain)
  x, y = domain.elem_eval( [ geom[0], basis ], ischeme='bezier9' )
  with plot.PyPlot( '1D' ) as plt:
    plt.plot( x, y, '-' )

  verts = numpy.arange(10)
  domain, geom = mesh.rectilinear( [ verts ] )
  basis = Mod( Mod( Spline( degree=2, rmlast=True ), 2 ), -3 ).build(domain)
  x, y = domain.elem_eval( [ geom[0], basis ], ischeme='bezier9' )
  with plot.PyPlot( '1D' ) as plt:
    plt.plot( x, y, '-' )

  verts = numpy.arange(10)
  domain, geom = mesh.rectilinear( [ verts ] )
  basis = Mod( Mod( Spline( degree=2, periodic=True ), 2 ), 4 ).build(domain)
  x, y = domain.elem_eval( [ geom[0], basis ], ischeme='bezier9' )
  with plot.PyPlot( '1D' ) as plt:
    plt.plot( x, y, '-' )

  domain, geom = mesh.rectilinear( [ numpy.arange(5) ] * 2 )
  basis = ( Spline( degree=1, rmfirst=True ) * Mod( Spline( degree=2, periodic=True ), 2 ) ).build(domain)
  x, y = domain.elem_eval( [ geom, basis ], ischeme='bezier5' )
  with plot.PyPlot( '2D' ) as plt:
    for i, yi in enumerate( y.T ):
      plt.subplot( 4, 4, i+1 )
      plt.mesh( x, yi )
      plt.gca().set_axis_off()


if __name__ == '__main__':
  util.run( example )
