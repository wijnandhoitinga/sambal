import numpy, ctypes
loader = ctypes.LibraryLoader( ctypes.CDLL )


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

  def setprops( self ):
    if self.pointlist and not hasattr( self, 'points' ):
      self.points = numpy.ctypeslib.as_array( self.pointlist, (self.numberofpoints,2) )
    if self.pointmarkerlist and not hasattr( self, 'pointmarkers' ):
      self.pointmarkers = numpy.ctypeslib.as_array( self.pointmarkerlist, (self.numberofpoints,) )
    if self.pointattributelist and not hasattr( self, 'pointattributes' ):
      self.pointattributes = numpy.ctypeslib.as_array( self.pointattributelist, (self.numberofpoints,self.numberofpointattributes) )
    if self.trianglelist and not hasattr( self, 'triangles' ):
      self.triangles = numpy.ctypeslib.as_array( self.trianglelist, (self.numberoftriangles,self.numberofcorners) )
    if self.triangleattributelist and not hasattr( self, 'triangleattributes' ):
      self.triangleattributes = numpy.ctypeslib.as_array( self.triangleattributelist, (self.numberoftriangles,self.numberoftriangleattributes) )
    if self.trianglearealist and not hasattr( self, 'triangleareas' ):
      self.triangleareas = numpy.ctypeslib.as_array( self.trianglearealist, (self.numberoftriangles,) )
    if self.neighborlist and not hasattr( self, 'neighbors' ):
      self.neighbors = numpy.ctypeslib.as_array( self.neighborlist, (self.numberoftriangles,3) )
    if self.segmentlist and not hasattr( self, 'segments' ):
      self.segments = numpy.ctypeslib.as_array( self.segmentlist, (self.numberofsegments,2) )
    if self.segmentmarkerlist and not hasattr( self, 'segmentmarkers' ):
      self.segmentmarkers = numpy.ctypeslib.as_array( self.segmentmarkerlist, (self.numberofsegments,) )
    if self.holelist and not hasattr( self, 'holes' ):
      self.holes = numpy.ctypeslib.as_array( self.holelist, (self.numberofholes,2) )
    if self.regionlist and not hasattr( self, 'regions' ):
      self.regions = numpy.ctypeslib.as_array( self.regionlist, (self.numberofregions,4) )
    if self.edgelist and not hasattr( self, 'edges' ):
      self.edges = numpy.ctypeslib.as_array( self.edgelist, (self.numberofedges,2) )
    if self.edgemarkerlist and not hasattr( self, 'edgemarkers' ):
      self.ergemarkers = numpy.ctypeslib.as_array( self.edgemarkerlist, (self.numberofedges,) )
    if self.normlist and not hasattr( self, 'normals' ):
      self.normals = numpy.ctypeslib.as_array( self.normlist, (self.numberofedges,2) )

  def triangulate( self, fortran=False, poly=False, quality=False, chull=False, area=False, quiet=False, verbose=0 ):
    switches = ''
    if not fortran:
      switches += 'z'
    if poly:
      switches += 'p'
    if quality:
      switches += 'q'
    if chull:
      switches += 'c'
    if area:
      switches += 'a{:f}'.format( area )
    if quiet:
      switches += 'Q'
    if verbose:
      switches += 'V' * verbose
    out = Triangulate()
    loader['libtriangle.so'].triangulate( ctypes.c_char_p(switches), ctypes.pointer(self), ctypes.pointer(out) )
    out.setprops()
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
