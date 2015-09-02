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

class VoxelData ( object ):

  def __init__ ( self, data, bounding_box ):
    self.data         = data
    self.bounding_box = bounding_box

    self.ndim  = self.data.ndim
    self.shape = self.data.shape

  @property
  def lengths ( self ):
    return tuple(bb[1]-bb[0] for bb in self.bounding_box)

  @property
  def volume( self ):
    return numpy.prod( self.lengths )

  @property
  def spacing ( self ):
    return tuple( l/float(sh) for l, sh in zip( self.lengths, self.shape ) )

  @cache.property
  def mesh ( self ):
    return mesh.rectilinear( [ numpy.linspace( b[0], b[1], sh+1 ) for b,sh in zip(self.bounding_box,self.shape)] )

  @property
  def topo ( self ):
    return self.mesh[0]

  @property
  def geom ( self ):
    return self.mesh[1]

  @cache.property
  def func ( self ):
    mapping = { elem.transform:value for elem,value in zip(self.topo.structure.ravel(),self.data.ravel()) }
    return function.Elemwise( mapping, shape=() )

  def __getitem__ ( self, Slice ):

    if Slice is Ellipsis:
      return self

    if isinstance(Slice,slice):
      if Slice==slice(None):
        return self
      Slice = (Slice,)*self.ndim

    if isinstance(Slice,tuple):
      assert len(Slice)==self.ndim

      bounding_box = []
      for d in range(self.ndim):
        left_verts = numpy.linspace(self.bounding_box[d][0],self.bounding_box[d][1]-self.spacing[d],self.shape[d])[Slice[d]]
        bounding_box.append( (left_verts[0],left_verts[-1]+self.spacing[d]) )

      sliced = VoxelData( self.data[Slice], bounding_box )
      numpy.testing.assert_allclose( sliced.spacing, self.spacing, rtol=0., atol=1e-14 )

      return sliced

    raise Exception('Unsupported slicing operation')

  def coarsegrain ( self, ncg ):

    if ncg < 1:
      return self

    cgshape = [d//(2**ncg) for d in self.data.shape]

    log.info( 'Coarse grained data shape: (%s)' % ','.join(map(str,cgshape)) )

    #Create the coarse grain domain
    cgtopo, cggeom = mesh.rectilinear( [numpy.linspace( b[0], b[1], sh+1 ) for b,sh in zip(self.bounding_box,cgshape)] )
    cgfunc = TopoMap( self.func, func_topo=self.topo, geometry=cggeom, bounding_box=self.bounding_box )
    cgdata = cgtopo.elem_mean( cgfunc, geometry=cggeom, ischeme='uniform%i'%(2**ncg) )

    return VoxelData( cgdata, self.bounding_box )

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
  bb = [ [0,sh*sp] for sp,sh in zip(spacing,shape)]

  return VoxelData( data, bb )

def jsonread( fname ):

  import json
  from os import path

  assert path.isfile( fname ), 'File "%s" does not exist.' % fname

  dirname = path.dirname( fname )

  jsondict = json.load( open(fname) )

  fname   = jsondict['FNAME']
  shape   = jsondict['DIMS']
  spacing = jsondict['SIZE']

  if not jsondict.has_key('FORMAT'):
    dtype = '<i2' #Two byte integer little endian
  else:
    dtype = jsondict['FORMAT']

  fname = path.join( dirname, fname )

  assert path.isfile( fname ), 'File "%s" does not exist.' % fname

  data = numpy.fromfile( file=open( fname, 'rb' ), dtype=dtype )[:numpy.prod( shape )].reshape( shape )

  #Slice the data
  if jsondict.has_key('SLICE'):
    raise DeprecationWarning('SLICE key is no longer supported in the json metadata file')

  #Shift to threshold
  assert jsondict.has_key('THRESHOLD'), 'THRESHOLD VALUE MUST BE PROVIDED'
  data = data - jsondict['THRESHOLD']

  log.info( 'Original data shape: (%s)' % ','.join(map(str,data.shape)) )

  #Construct the domain and geometry
  bb = [[0,sh*sp] for sp,sh in zip(spacing,data.shape)]

  return VoxelData( data, bb )

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
