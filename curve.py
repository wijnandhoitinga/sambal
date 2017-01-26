from nutils import function, numeric, _, debug
import numpy

_sqr = lambda x: numeric.contract_fast( x, x, 1 ) # _sqr(x) = norm2(x)**2
pi   = numpy.pi


class Curve( object ):

  def __init__( self, orig=[0,0], angle=0, segments=[] ):
    
    self.current = numpy.array( orig, dtype=float )
    assert self.current.shape == (2,)
    self.angle = angle
    self.segments = list(segments)
    self.cumlen = [0]
    for segment in segments:
      self.cumlen.append( self.cumlen[-1] + segment.length )
      
  @property
  def rot( self ):
    return rotmat( self.angle )

  def __add__( self, other ):
    assert isinstance(other,Curve), 'Cannot add non curve objects'
    segments = list( self.segments )
    for segment in other.segments:
      newsegment = segment.transform( shift=self.current, angle=self.angle )
      segments.append( newsegment )
     
    return Curve( orig=self.current+numpy.dot( self.rot, other.current ), angle=self.angle+other.angle, segments=segments )
    

  @property
  def length( self ):
    return self.cumlen[-1]

  
  def grow( self, length=None, angle=None, curvature=0, flag=False ):

    if length is None:
      assert angle is not None
      assert curvature != 0
      length = abs(angle / curvature)
    else:
      assert angle is None

    if curvature == 0:
      tang = numpy.array([ numpy.cos(self.angle), numpy.sin(self.angle) ])
      xy0 = self.current
      xy1 = xy0 + length * tang
      segment = Line( xy0=xy0, xy1=xy1 )
    else:
      radius = 1./curvature
      norm = numpy.array([ numpy.sin(self.angle), -numpy.cos(self.angle) ])
      origin = self.current + norm / curvature
      phi0 = self.angle + .5 * numpy.pi
      phi1 = phi0 - length * curvature
      self.angle = phi1 - .5 * numpy.pi
      segment = Arc( origin=origin, radius=radius, phi0=phi0, phi1=phi1 )
    
    segment.flag = flag
    self.current = segment.xy1
    self.cumlen.append( self.cumlen[-1] + segment.length )
    self.segments.append( segment )

  def move( self, length=None, absolute=None, angle=None, absangle=None ):
    if absolute is not None:
      assert length==None, 'length is not None, you cannot specify a length AND absolute coordinate'
      self.current = numpy.array( absolute )
    elif length is not None:
      assert absolute==None, 'absolute is not None, you cannot specify a length AND absolute coordinate'
      self.current = self.current + length*numpy.array( [numpy.cos(self.angle), numpy.sin(self.angle)] )

    if angle is not None:
      assert absangle==None, 'absangle is not None, you cannot specify a relative AND absolute angle'
      self.angle += angle
    elif absangle is not None:
      assert angle==None, 'angle is not None, you cannot specify a absolute AND relative angle'
      self.angle = absangle

  def scale( self, factor ):
    segments = [segment.scale( factor ) for segment in self.segments ]
    return Curve( orig=self.current*factor, angle=self.angle, segments=segments )

  def sample( self, spacing ):
    return numpy.concatenate( [ segment.getcoords( numpy.linspace(0,1,segment.length/spacing, endpoint=False) ) for segment in self.segments ], axis=0 )

  def findclosest( self, xy ):
    assert self.segments
    for i, segment in enumerate( self.segments ):
      alpha = numpy.minimum( 1-1e-14, numpy.maximum( 0, segment.findclosest( xy ) ) )
      dist2 = _sqr( segment.getcoords(alpha) - xy )
      if i == 0:
        dist2s = dist2
        alphas = alpha
      else:
        smaller = dist2<dist2s#function.less(dist2, dist2s)
        dist2s[smaller] = dist2[smaller]
        alphas[smaller] = alpha[smaller] + i
    return alphas

# @staticmethod
  def lump_integers( self, cumalpha ):#integers ):
    integers = numpy.minimum( cumalpha.astype(int), len(self.segments)-1 )
    enum_integers = numpy.array([ numpy.arange( len(integers) ), integers ]).T
    while enum_integers.size:
      i = enum_integers[0,1]
      selected = ( enum_integers[:,1] == i )
      yield i, enum_integers[selected,0]
      enum_integers = enum_integers[~selected]
    
  def getcoords( self, cumalpha ):
    coords = numpy.empty( [ len(cumalpha), 2 ] )
    for i, index in self.lump_integers( cumalpha ):
      coords[index] = self.segments[i].getcoords( cumalpha[index]-i )
    return coords

  def tangent( self, cumalpha ):
    tangent = numpy.empty( [ len(cumalpha), 2 ] )
    for i, index in self.lump_integers( cumalpha ):
      tangent[index] = self.segments[i].tangent( cumalpha[index]-i )
    return tangent

  def pathlen( self, cumalpha ):
    lenghts = numpy.empty( len(cumalpha) )
    for i, index in self.lump_integers( cumalpha ):
      lenghts[index] = self.cumlen[i] + (cumalpha[index]-i) * self.segments[i].length
    return lenghts
  
  def pathflag( self, cumalpha ):
    flags = numpy.empty( len(cumalpha) )
    for i, index in self.lump_integers( cumalpha ):
      flags[index] = self.segments[i].flag*numpy.ones_like(self.getcoords(cumalpha))
    return flags

class Segment( object ):

  def __init__( self, xy0, xy1, length, flag=False ):
    self.xy0 = xy0
    self.xy1 = xy1
    self.length = length
    self.flag = flag
  

def rotmat( angle ):
  sin = numpy.sin( angle )
  cos = numpy.cos( angle )
  return numpy.array([[ cos,sin ],[-sin,cos]])

def rotate( x, angle ):
  return numpy.dot( rotmat(angle), x )

class Line( Segment ):
  def __init__( self, xy0, xy1, flag=False ):
    Segment.__init__( self, xy0, xy1, numpy.linalg.norm(xy0-xy1), flag=flag )
    self.type = 'Line'

  def getcoords( self, alpha ):
    return self.xy0 + alpha[...,_] * ( self.xy1 - self.xy0 )
   
  def findclosest( self, xy ):
    R1sqr = _sqr( xy - self.xy0 )
    R2sqr = _sqr( xy - self.xy1 )
    return .5 - ( R2sqr - R1sqr ) / ( 2 * self.length**2 )

  def tangent( self, alpha ):
    return ( ( self.xy1 - self.xy0 )/ numpy.linalg.norm( self.xy1-self.xy0 ) )[_,:]

 
  def transform( self, shift, angle ):
    xy0 = rotate( self.xy0, angle ) + shift
    xy1 = rotate( self.xy1, angle ) + shift
    return Line( xy0=xy0, xy1=xy1 )

  def scale( self, factor ):
    return Line( xy0=self.xy0*factor, xy1=self.xy1*factor, flag=self.flag )
    

class Arc( Segment ):
  def __init__( self, origin, radius, phi0, phi1, flag=False ):

    xy0 = origin + radius * numpy.array([ -numpy.sin(phi0-.5*numpy.pi), numpy.cos(phi0-.5*numpy.pi) ])
    xy1 = origin + radius * numpy.array([ -numpy.sin(phi1-.5*numpy.pi), numpy.cos(phi1-.5*numpy.pi) ])
    length = abs( (phi1-phi0) * radius )
    Segment.__init__( self, xy0, xy1, length, flag=flag )

    self.origin = origin
    self.radius = radius
    self.phi0 = phi0
    self.phi1 = phi1
    self.type = 'Arc'

  def getcoords( self, alpha ):
    a = self.phi0 + alpha * ( self.phi1 - self.phi0 )
    return self.origin + self.radius*numpy.array( [numpy.cos(a), numpy.sin(a)] ).T

  def findclosest( self, xy ):
    phi = numpy.arctan2( xy[...,1]-self.origin[1], xy[...,0]-self.origin[0])
    if self.radius < 0:
      phi += numpy.pi

    meanphi = .5 * ( self.phi0 + self.phi1 )
    beta = numpy.remainder( phi - (meanphi-numpy.pi), 2*numpy.pi ) - numpy.pi #=> phi === meanphi + beta
    dphi = self.phi1 - self.phi0 #=> phi = meanphi + (alpha-.5) * dphi

    return beta / dphi + .5

  def tangent( self, alpha ):
    angle = self.phi0 + alpha * ( self.phi1 - self.phi0 )
    return (numpy.array( [numpy.sin(angle), -numpy.cos(angle)] )).T# * numpy.sign( self.radius )).T
  
  def transform( self, shift, angle ):
    origin = rotate( self.origin, angle ) + shift
    phi0 = self.phi0 - angle
    phi1 = self.phi1 - angle 
    return Arc( origin=origin, radius=self.radius, phi0=phi0, phi1=phi1 ) 

  def scale( self, factor ):
    return Arc( origin=self.origin*factor, radius=self.radius*factor, phi0=self.phi0, phi1=self.phi1, flag=self.flag )




class FindClosest( function.Array ):
  def __init__( self, curve, coords ):
    self.curve = curve
    function.Array.__init__( self, args=[coords], shape=(), dtype=float )
  def evalf( self, coords ):
    return self.curve.findclosest(coords)

class PathCoords( function.Array ):
  def __init__( self, curve, alpha ):
    self.curve = curve
    function.Array.__init__( self, args=[alpha], shape=(2,), dtype=float )
  def evalf( self, alpha ):
    return self.curve.getcoords(alpha)

class PathTangent( function.Array ):
  def __init__( self, curve, alpha ):
    self.curve = curve
    function.Array.__init__( self, args=[alpha], shape=(2,), dtype=float )
  def evalf( self, alpha ):
    return self.curve.tangent(alpha)

class PathLength( function.Array ):
  def __init__( self, curve, alpha ):
    self.curve = curve
    function.Array.__init__( self, args=[alpha], shape=(), dtype=float )
  def evalf( self, alpha ):
    return self.curve.pathlen(alpha)

class PathFlag( function.Array ):
  def __init__( self, curve, alpha ):
    self.curve = curve
    function.Array.__init__( self, args=[alpha], shape=(), dtype=float )
  def evalf( self, alpha ):
    return self.curve.pathflag(alpha)

class Font( object ):
  def __init__( self ):
    
    self.A = Curve( orig=[0,0], angle=numpy.arctan(2) )
    self.A.grow( length=numpy.sqrt(1.25) )
    self.A.move( angle=-2*numpy.arctan(2) )
    self.A.grow( length=numpy.sqrt(1.25) )
    self.A.move( absolute=[0.23,0.4], absangle=0 )
    self.A.grow( length=0.53)
    self.A.move( absolute=[1.2,0.0], absangle=0 )

    self.C = Curve( orig=0.5+numpy.array([0.25*numpy.sqrt(2),0.25*numpy.sqrt(2)]), angle=3*pi/4 )
    self.C.grow( angle=3*pi/2, curvature=-2. )
    self.C.move( absolute=[1.1,0.0], absangle=0 )

    self.D = Curve( orig=[0.0,0.0], angle=pi/2 )
    self.D.grow( length=1.0 )
    self.D.move( angle=-pi/2 )
    self.D.grow( length=0.5 )
    self.D.grow( angle=pi/2, curvature=4. )
    self.D.grow( length=0.5 )
    self.D.grow( angle=pi/2, curvature=4. )
    self.D.grow( length=0.5 )
    self.D.move( absolute=[1.2,0.0], absangle=0 )
    
    self.E = Curve( orig=[0.8,0.0], angle=-pi )
    self.E.grow( length=0.8 )
    self.E.grow( angle=pi/2, curvature=1000 )
    self.E.grow( length=1.0 )
    self.E.grow( angle=pi/2, curvature=1000 )
    self.E.grow( length=0.8 )
    self.E.move( absolute=[0,0.5] )
    self.E.grow( length=0.5 )
    self.E.move( absolute=[1.0,0.0], absangle=0 )

    self.G = Curve( orig=0.5+numpy.array([0.25*numpy.sqrt(2),0.25*numpy.sqrt(2)]), angle=3*pi/4 )
    self.G.grow( angle=7*pi/4, curvature=-2. )
    self.G.move( absolute=[0.5,0.5], absangle=0 )
    self.G.grow( length=0.5 )
    self.G.move( absolute=[1.2,0.0], absangle=0 )

    self.H = Curve( orig=[0,0], angle=pi/2 )
    self.H.grow( length=1.0 )
    self.H.move( length=-0.5, absangle=0 )
    self.H.grow( length=1.0 )
    self.H.move( absolute=[1.0,0.0], absangle=pi/2 )
    self.H.grow( length=1.0 )
    self.H.move( absolute=[1.2,0.0], absangle=0 )
 
    self.I = Curve( orig=[0.1,0], angle=pi/2 )
    self.I.grow( length=1.0 )
    self.I.move( absolute=[0.5,0], absangle=0 )

    self.L = Curve( orig=[0.0,1.0], angle=-pi/2 )
    self.L.grow( length=1.0 )
    self.L.move( absangle=0 )
    self.L.grow( length=1.0 )
    self.L.move( absolute=[1.2,0], absangle=0 )

    self.M = Curve( orig=[0.0,0.0], angle=pi/2 )
    self.M.grow( length=1.0 )
    self.M.move( angle=-2*pi/3 )
    self.M.grow( length=0.6 )
    self.M.move( angle=pi/3 )
    self.M.grow( length=0.6 )
    self.M.move( angle=-2*pi/3 )
    self.M.grow( length=1.0 )
    self.M.move( absolute=[1.2,0.0], absangle=0 )

    self.N = Curve( orig=[0,0], angle=pi/2 )
    self.N.grow( length=1.0 )
    self.N.grow( angle=3*pi/4, curvature=1000 )
    self.N.grow( length=numpy.sqrt( 2 ) )
    self.N.grow( angle=3*pi/4, curvature=-1000 )
    self.N.grow( length=1.0 )
    self.N.move( absolute=[1.2,0.0], absangle=0 )

    self.O = Curve( orig=[0.5, 0], angle=0 )
    self.O.grow( angle=2*pi, curvature=-2.0 )
    self.O.move( absolute=[1.2,0.0], absangle=0 )

    self.R = Curve( orig=[0,0], angle=pi/2 )
    self.R.grow( length=1.0 )
    self.R.move( angle=-pi/2 )
    self.R.grow( length=0.65 )
    self.R.grow( angle=pi, curvature=4 )
    self.R.grow( length=0.65 )
    self.R.move( absolute=[0.65,0.5], absangle=0 )
    self.R.grow( angle=pi/2, curvature=4 )
    self.R.grow( length= 0.25 )
    self.R.move( absolute=[1.2,0.0], absangle=0 )

    self.S = Curve( orig=[1.0,1.0], angle=-pi )
    self.S.grow( length=0.65 )
    self.S.grow( angle=pi, curvature=-4.0 )
    self.S.grow( length=0.5 )
    self.S.grow( angle=pi, curvature=4.0 )
    self.S.grow( length=0.65 )
    self.S.move( absolute=[1.2,0.0], absangle=0 )

    self.T = Curve( orig=[0.5,0], angle=pi/2 )
    self.T.grow( length=1.0 )
    self.T.move( absolute=[0.0,1.0], absangle=0 )
    self.T.grow( length=1.0 )
    self.T.move( absolute=[1.2,0.0], absangle=0 )

    self.U = Curve( orig=[0.0,1.0], angle=-pi/2 )
    self.U.grow( length=0.5 )
    self.U.grow( angle=pi, curvature=-2.0 )
    self.U.grow( length=0.5 )
    self.U.move( absolute=[1.2,0.0], absangle=0 )

    self.v = Curve( orig=[0.25,0.5], angle=-1*pi/3 )
    self.v.grow( length=numpy.sqrt(.25**2+.25) )
    self.v.grow( angle=4*pi/3, curvature=1000 )
    self.v.grow( length=numpy.sqrt(.25**2+.25) )
    self.v.move( absolute=[1.2,0.0], absangle=0 )

    self.Z = Curve( orig=[0.0,1.0], angle=0 )
    self.Z.grow( length=1.0 )
    self.Z.grow( angle=3.*pi/4., curvature=1000 )
    self.Z.grow( length=numpy.sqrt( 2 ) )
    self.Z.grow( angle=3.*pi/4., curvature=-1000 )
    self.Z.grow( length=1.0 )
    self.Z.move( absolute=[1.2,0.0], absangle=0 )
     

  def text( self, letters, spacing=0.0, fontsize=1.0, orig=None ):
    text = Curve() 
    if orig is not None:
      text.move( absolute=orig )
    for iletter, letter in enumerate( letters ):
      text += getattr( self, letter ).scale( factor=fontsize )
      text.move( length=spacing )

    self.line = text
    return text
  def addtext( self, letters, spacing=0.0, fontsize=1.0, orig=None ):
    text=self.line
    if orig is not None:
      text.move( absolute=orig )
    for iletter, letter in enumerate( letters ):
      text += getattr( self, letter.upper() ).scale( factor=fontsize )
      text.move( length=spacing )

    self.line = text
    return text

