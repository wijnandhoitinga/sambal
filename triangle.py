import numpy, ctypes
loader = ctypes.LibraryLoader( ctypes.CDLL )


def myproperty( f ):
  name = f.__name__
  def property_getter( self ):
    try:
      value = self.__dict__[name]
    except KeyError:
      value = f( self )
    return value
  def property_setter( self, value ):
    self.__dict__[name] = value
  return property( fget=property_getter, fset=property_setter )


class Triangulate( ctypes.Structure ):

  c_double = ctypes.c_double
  c_int = ctypes.c_int
  p_double = ctypes.POINTER( c_double )
  p_int = ctypes.POINTER( c_int )

  _fields_ = [
    ( 'pointlist', p_double ),
    ( 'pointattributelist', p_double ),
    ( 'pointmarkerlist', p_int ),
    ( 'numberofpoints', c_int ),
    ( 'numberofpointattributes', c_int ),
    ( 'trianglelist', p_int ),
    ( 'triangleattributelist', p_double ),
    ( 'trianglearealist', p_double ),
    ( 'neighborlist', p_int ),
    ( 'numberoftriangles', c_int ),
    ( 'numberofcorners', c_int ),
    ( 'numberoftriangleattributes', c_int ),
    ( 'segmentlist', p_int ),
    ( 'segmentmarkerlist', p_int ),
    ( 'numberofsegments', c_int ),
    ( 'holelist', p_double ),
    ( 'numberofholes', c_int ),
    ( 'regionlist', p_double ),
    ( 'numberofregions', c_int ),
    ( 'edgelist', p_int ),
    ( 'edgemarkerlist', p_int ),
    ( 'normlist', p_double ),
    ( 'numberofedges', c_int ),
  ]

  @myproperty
  def points( self ):
    return numpy.ctypeslib.as_array( self.pointlist, (self.numberofpoints,2) )

  @myproperty
  def pointmarkers( self ):
    return numpy.ctypeslib.as_array( self.pointmarkerlist, (self.numberofpoints,) )

  @myproperty
  def pointattributes( self ):
    return numpy.ctypeslib.as_array( self.pointattributelist, (self.numberofpoints,self.numberofpointattributes) )

  @myproperty
  def triangles( self ):
    return numpy.ctypeslib.as_array( self.trianglelist, (self.numberoftriangles,self.numberofcorners) )

  @myproperty
  def triangleattributes( self ):
    return numpy.ctypeslib.as_array( self.triangleattributelist, (self.numberoftriangles,self.numberoftriangleattributes) )

  @myproperty
  def triangleareas( self ):
    return numpy.ctypeslib.as_array( self.trianglearealist, (self.numberoftriangles,) )

  @myproperty
  def neighbors( self ):
    return numpy.ctypeslib.as_array( self.neighborlist, (self.numberoftriangles,3) )

  @myproperty
  def segments( self ):
    return numpy.ctypeslib.as_array( self.segmentlist, (self.numberofsegments,2) )

  @myproperty
  def segmentmarkers( self ):
    return numpy.ctypeslib.as_array( self.segmentmarkerlist, (self.numberofsegments,) )

  @myproperty
  def holes( self ):
    return numpy.ctypeslib.as_array( self.holelist, (self.numberofholes,2) )

  @myproperty
  def regions( self ):
    return numpy.ctypeslib.as_array( self.regionlist, (self.numberofregions,4) )

  @myproperty
  def edges( self ):
    return numpy.ctypeslib.as_array( self.edgelist, (self.numberofedges,2) )

  @myproperty
  def ergemarkers( self ):
    return numpy.ctypeslib.as_array( self.edgemarkerlist, (self.numberofedges,) )

  @myproperty
  def normals( self ):
    return numpy.ctypeslib.as_array( self.normlist, (self.numberofedges,2) )

  def triangulate( self, fortran=False, poly=False, minangle=28.6, chull=False, area=False, incremental=False, quiet=False, verbose=0 ):
    switches = ''
    if not fortran:
      switches += 'z'
    if poly:
      switches += 'p'
    if minangle:
      switches += 'q{:f}'.format( minangle )
    if chull:
      switches += 'c'
    if area:
      switches += 'a{:f}'.format( area )
    if incremental:
      switches += 'i'
    if quiet:
      switches += 'Q'
    if verbose:
      switches += 'V' * verbose
    out = Triangulate()
    loader['libtriangle.so'].triangulate( ctypes.c_char_p(switches), ctypes.pointer(self), ctypes.pointer(out) )
    return out


class TriangulateInput( Triangulate ):

  def __init__( self, points, segments, pointmarkers=None, segmentmarkers=None, holes=None ):

    self.points = numpy.asarray( points, numpy.float64 )
    self.numberofpoints = len(points)
    self.pointlist = self.points.ctypes.data_as(self.p_double)

    if pointmarkers is not None:
      assert len(pointmarkers) == len(points)
      self.pointmarkers = numpy.asarray( pointmarkers, numpy.int32 )
      self.pointmarkerlist = self.pointmarkers.ctypes.data_as(self.p_int)

    self.segments = numpy.asarray( segments, numpy.int32 )
    self.numberofsegments = len(segments)
    self.segmentlist = self.segments.ctypes.data_as(self.p_int)

    if segmentmarkers is not None:
      assert len(segmentmarkers) == len(segments)
      self.segmentmarkers = numpy.asarray( segmentmarkers, numpy.int32 )
      self.segmentmarkerlist = self.segmentmarkers.ctypes.data_as(self.p_int)

    if holes is not None:
      self.holes = numpy.asarray( holes, numpy.int32 )
      self.numberofholes = len(holes)
      self.holelist = self.holes.ctypes.data_as(self.p_double)

    self.numberofpointattributes = 0
    self.numberofregions = 0


if __name__ == '__main__':

  tri = TriangulateInput( 
    points = [[0,0],[0,1],[1,1],[1,0]],
    segments = [[0,1],[1,2],[2,3],[3,0]]
  ).triangulate( area=.1 )

  tri = TriangulateInput( 
    points = [[0,0],[0,3],[3,0],[3,3],[1,1],[1,2],[2,1],[2,2]],
    pointmarkers = [0,0,0,33,0,0,0,0],
    segments = [[1,2],[5,7],[7,8],[8,6],[6,5]],
    segmentmarkers = [5,0,0,10,0],
    holes = [[1.5,1.5]],
  ).triangulate( fortran=True )
