from nutils import *

class TopoMap( function.ArrayFunc ):
  """Topology mapping ArrayFunc wrapper

  ArrayFunc wrapper for function evaluation on an arbitrary topology. 

  Args:
    func (ArrayFunc): ArrayFunc to be wrapper
    func_topo: Topology on which func can be evaluated
    geometry: Geometric map evaluable on the target topology
    bounding_box: Extends of the geometry 

  Returns:
    func (:class:`TopoMap`): ArrayFunc 

  """

  def __init__( self, func, func_topo, geometry, bounding_box ):

    assert isinstance( func_topo, topology.StructuredTopology )
    assert geometry.ndim==1
    assert geometry.shape[0]==func_topo.ndims
    assert isinstance(bounding_box,list)
    assert len(bounding_box)==func_topo.ndims
    self.bb = numpy.array(bounding_box) #[[xmin,xmax],[ymin,ymax],...]
    assert self.bb.shape[1]==2

    self.func_topo = func_topo.structure
    self.func = func
    function.ArrayFunc.__init__( self, args=[geometry], shape=func.shape )

  def evalf( self, x ):
    shp    = numpy.array(self.func_topo.shape)
    scale  = shp/(self.bb[:,1]-self.bb[:,0])
    indf   = (x-self.bb[:,0])*scale
    indi   = numpy.maximum(numpy.minimum(numpy.floor(indf),shp-1),0).astype(int)
    points = indf-indi

    elems  = numpy.array([self.func_topo[tuple(ind)] for ind in indi])
    values = numpy.zeros( shape=(points.shape[:1]+self.func.shape) )
    for elem in numpy.unique( elems ):
      mask = (elems==elem)
      values[mask] = self.func.eval(elem,points[mask])
    
    return values


def voxread ( fname ):

  assert fname.endswith('.vox'), 'Expected a vox files'

  #Reading the data from the vox file
  with open( fname ) as fin:
    fin = open( fname )
    title = fin.readline()
    spacing = tuple(float(fin.readline().strip()) for i in range(3))
    shape = tuple(int(fin.readline().strip()) for i in range(3))
    sdata = fin.readline().strip()
    assert len(sdata)==numpy.prod(shape), 'Incorrect data size'
  
  #Convert to numpy array
  data = (numpy.fromstring( sdata, dtype=numpy.uint8 )==83).astype(float).reshape( shape )-0.5

  log.info( 'Original data shape: (%s)' % ','.join(map(str,data.shape)) )

  #Construct the domain and geometry
  topo, geom = mesh.rectilinear( [ numpy.linspace( 0, sh*sp, sh+1 ) for sp,sh in zip(spacing,shape)] )

  #Cosntruct the ArrayFunc
  mapping = { elem.transform:value for elem,value in zip(topo.structure.ravel(),data.ravel()) }
  func = function.Elemwise( mapping, shape=() )

  return topo, geom, func

def get_boundingbox( topo, geom ):

  bb = []

  if topo.ndims == 2:
    bb.append( [ topo.boundary['left']  .boundary['left'].integrate( geom[0], geometry=geom, ischeme='gauss1' ),
                 topo.boundary['right'] .boundary['left'].integrate( geom[0], geometry=geom, ischeme='gauss1' )  ] )
    bb.append( [ topo.boundary['bottom'].boundary['left'].integrate( geom[1], geometry=geom, ischeme='gauss1' ),
                 topo.boundary['top']   .boundary['left'].integrate( geom[1], geometry=geom, ischeme='gauss1' )  ] )
  if topo.ndims == 3:
    bb.append( [ topo.boundary['left']  .boundary['left'].boundary['left'].integrate( geom[0], geometry=geom, ischeme='gauss1' ),
                 topo.boundary['right'] .boundary['left'].boundary['left'].integrate( geom[0], geometry=geom, ischeme='gauss1' )  ] )
    bb.append( [ topo.boundary['bottom'].boundary['left'].boundary['left'].integrate( geom[1], geometry=geom, ischeme='gauss1' ),
                 topo.boundary['top']   .boundary['left'].boundary['left'].integrate( geom[1], geometry=geom, ischeme='gauss1' )  ] )
    bb.append( [ topo.boundary['front'] .boundary['left'].boundary['left'].integrate( geom[2], geometry=geom, ischeme='gauss1' ),
                 topo.boundary['back']  .boundary['left'].boundary['left'].integrate( geom[2], geometry=geom, ischeme='gauss1' )  ] )

  return bb

def coarsegrain ( topo, geom, func, ncg ):

  if ncg < 1:
    return topo, geom, func

  bb = get_boundingbox( topo, geom )
  cgshape = [d//(2**ncg) for d in topo.structure.shape]

  log.info( 'Coarse grained data shape: (%s)' % ','.join(map(str,cgshape)) )

  #Create the coarse grain domain
  cgtopo, cggeom = mesh.rectilinear( [numpy.linspace( b[0], b[1], sh+1 ) for b,sh in zip(bb,cgshape)] )
  cgfunc = TopoMap( func, func_topo=topo, geometry=cggeom, bounding_box=bb )
  cgdata = cgtopo.elem_mean( cgfunc, geometry=cggeom, ischeme='uniform%i'%(2**ncg) )

  #Cosntruct the ArrayFunc
  mapping = { elem.transform:value for elem,value in zip(cgtopo.structure.ravel(),cgdata) }
  cgfunc = function.Elemwise( mapping, shape=() )

  return cgtopo, cggeom, cgfunc
