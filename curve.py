from nutils import function, numeric, _
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

  
  def grow( self, length=None, angle=None, curvature=0 ):

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
    return numpy.concatenate( [ segment.getcoords( numpy.linspace(0,1,segment.length/spacing) ) for segment in self.segments ], axis=0 )

  def findclosest( self, xy ):
    assert self.segments
    for i, segment in enumerate( self.segments ):
      alpha = numpy.minimum( 1-1e-14, numpy.maximum( 0, segment.findclosest( xy ) ) )
      dist2 = _sqr( segment.getcoords(alpha) - xy )
      if i == 0:
        dist2s = dist2
        alphas = alpha
      else:
        smaller = dist2 < dist2s
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

class Segment( object ):

  def __init__( self, xy0, xy1, length ):
    self.xy0 = xy0
    self.xy1 = xy1
    self.length = length


def rotmat( angle ):
  sin = numpy.sin( angle )
  cos = numpy.cos( angle )
  return numpy.array([[ cos,sin ],[-sin,cos]])

def rotate( x, angle ):
  return numpy.dot( rotmat(angle), x )

class Line( Segment ):
  def __init__( self, xy0, xy1 ):
    Segment.__init__( self, xy0, xy1, numeric.norm2(xy0-xy1) )
    self.type = 'Line'

  def getcoords( self, alpha ):
    return self.xy0 + alpha[...,_] * ( self.xy1 - self.xy0 )
   
  def findclosest( self, xy ):
    R1sqr = _sqr( xy - self.xy0 )
    R2sqr = _sqr( xy - self.xy1 )
    return .5 - ( R2sqr - R1sqr ) / ( 2 * self.length**2 )

  def tangent( self, alpha ):
    return ( ( self.xy1 - self.xy0 )/ numeric.norm2( self.xy1-self.xy0 ) )[_,:]

 
  def transform( self, shift, angle ):
    xy0 = rotate( self.xy0, angle ) + shift
    xy1 = rotate( self.xy1, angle ) + shift
    return Line( xy0=xy0, xy1=xy1 )

  def scale( self, factor ):
    return Line( xy0=self.xy0*factor, xy1=self.xy1*factor )
    

class Arc( Segment ):
  def __init__( self, origin, radius, phi0, phi1 ):

    xy0 = origin + radius * numpy.array([ -numpy.sin(phi0-.5*numpy.pi), numpy.cos(phi0-.5*numpy.pi) ])
    xy1 = origin + radius * numpy.array([ -numpy.sin(phi1-.5*numpy.pi), numpy.cos(phi1-.5*numpy.pi) ])
    length = abs( (phi1-phi0) * radius )
    Segment.__init__( self, xy0, xy1, length )

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
    return Arc( origin=self.origin*factor, radius=self.radius*factor, phi0=self.phi0, phi1=self.phi1 )

class FindClosest( function.ArrayFunc ):
  def __init__( self, curve, coords ):
    function.ArrayFunc.__init__( self, evalf=curve.findclosest, args=[coords], shape=() )

class PathCoords( function.ArrayFunc ):
  def __init__( self, curve, alpha ):
    function.ArrayFunc.__init__( self, evalf=curve.getcoords, args=[alpha], shape=(2,) )

class PathTangent( function.ArrayFunc ):
  def __init__( self, curve, alpha ):
    function.ArrayFunc.__init__( self, evalf=curve.tangent, args=[alpha], shape=(2,) )

class PathLength( function.ArrayFunc ):
  def __init__( self, curve, alpha ):
    function.ArrayFunc.__init__( self, evalf=curve.pathlen, args=[alpha], shape=() )



class Font( object ):
  def __init__( self ):
    self.scale=1.
    self.C = Curve( orig=0.5+numpy.array([0.25*numpy.sqrt(2),0.25*numpy.sqrt(2)]), angle=3*pi/4 )
    self.C.grow( angle=3*pi/2, curvature=-2/self.scale )
    self.C.move( absolute=[1.2,0.0], absangle=0 )
    
    self.E = Curve( orig=[1.0,0], angle=-pi )
    self.E.grow( length=self.scale )
    self.E.grow( angle=pi/2, curvature=1000/self.scale )
    self.E.grow( length=self.scale )
    self.E.grow( angle=pi/2, curvature=1000/self.scale )
    self.E.grow( length=self.scale )
    self.E.move( absolute=[0,0.5*self.scale] )
    self.E.grow( length=0.5*self.scale )
    self.E.move( absolute=[1.2,0.0], absangle=0 )

    self.G = Curve( orig=0.5+numpy.array([0.25*numpy.sqrt(2),0.25*numpy.sqrt(2)]), angle=3*pi/4 )
    self.G.grow( angle=7*pi/4, curvature=-2/self.scale )
    self.G.move( absolute=[0.5*self.scale,0.5*self.scale], absangle=0 )
    self.G.grow( length=0.5*self.scale )
    self.G.move( absolute=[1.2,0.0], absangle=0 )

    self.H = Curve( orig=[0,0], angle=pi/2 )
    self.H.grow( length=self.scale )
    self.H.move( length=-0.5*self.scale, absangle=0 )
    self.H.grow( length=self.scale )
    self.H.move( absolute=[self.scale,0], absangle=pi/2 )
    self.H.grow( length=self.scale )
    self.H.move( absolute=[1.2,0.0], absangle=0 )
 
    self.I = Curve( orig=[0.1,0], angle=pi/2 )
    self.I.grow( length=self.scale )
    self.I.move( absolute=[0.5,0], absangle=0 )

    self.N = Curve( orig=[0,0], angle=pi/2 )
    self.N.grow( length=self.scale )
    self.N.grow( angle=3*pi/4, curvature=1000/self.scale )
    self.N.grow( length=numpy.sqrt( 2*self.scale**2 ) )
    self.N.grow( angle=3*pi/4, curvature=-1000/self.scale )
    self.N.grow( length=self.scale )
    self.N.move( absolute=[1.2,0.0], absangle=0 )

    self.O = Curve( orig=[0.5*self.scale, 0], angle=0 )
    self.O.grow( angle=2*pi, curvature=-2/self.scale )
    self.O.move( absolute=[1.2,0.0], absangle=0 )

    self.R = Curve( orig=[0,0], angle=pi/2 )
    self.R.grow( length=self.scale )
    self.R.grow( angle=pi/2, curvature=1e3/self.scale )
    self.R.grow( length=0.75*self.scale )
    self.R.grow( angle=pi, curvature=4/self.scale )
    self.R.grow( length=0.75*self.scale )
    self.R.move( absolute=[0.75*self.scale,0.5*self.scale], absangle=0 )
    self.R.grow( angle=pi/2, curvature=4/self.scale )
    self.R.grow( length= 0.25*self.scale )
    self.R.move( absolute=[1.2,0.0], absangle=0 )

    self.S = Curve( orig=[1.0,1.0], angle=-pi )
    self.S.grow( length=0.75*self.scale )
    self.S.grow( angle=pi, curvature=-4/self.scale )
    self.S.grow( length=0.5*self.scale )
    self.S.grow( angle=pi, curvature=4/self.scale )
    self.S.grow( length=0.75*self.scale )
    self.S.move( absolute=[1.2,0.0], absangle=0 )

    self.T = Curve( orig=[0.5,0], angle=pi/2 )
    self.T.grow( length=self.scale )
    self.T.move( absolute=[0,self.scale], absangle=0 )
    self.T.grow( length=self.scale )
    self.T.move( absolute=[1.2,0.0], absangle=0 )

    self.U = Curve( orig=[0.0,1.0], angle=-pi/2 )
    self.U.grow( length=0.5*self.scale )
    self.U.grow( angle=pi, curvature=-2/self.scale )
    self.U.grow( length=0.5*self.scale )
    self.U.move( absolute=[1.2,0.0], absangle=0 )

    self.Z = Curve( orig=[0.0,1.0], angle=0 )
    self.Z.grow( length=self.scale )
    self.Z.grow( angle=3.*pi/4., curvature=1000/self.scale )
    self.Z.grow( length=numpy.sqrt( 2*self.scale**2 ) )
    self.Z.grow( angle=3.*pi/4., curvature=-1000/self.scale )
    self.Z.grow( length=self.scale )
    self.Z.move( absolute=[1.2,0.0], absangle=0 )
     

  def text( self, letters, spacing=0.0, fontsize=1.0 ):
    text = Curve() 
    for iletter, letter in enumerate( letters ):
      text += getattr( self, letter.upper() ).scale( factor=fontsize )
      text.move( length=spacing )

    return text
       
    

class Text( object ):

  def __init__( self, cursor, spacing, scale ):
    self.cursor = numpy.array( cursor )
    self.spacing = spacing
    self.scale = scale
    self.curve = Curve( orig=self.cursor, angle=0 )

  def addline( self, text, cursor=None, spacing=None  ):
    cursor = self.cursor if cursor is None else cursor
    spacing = self.spacing if spacing is None else spacing

    for letter in text:
      if letter in ['C', 'c']:
        self.setC()
      if letter in ['E', 'e']:
        self.setE()
      if letter in ['G', 'g']:
        self.setG()
      if letter in ['H','h']:
        self.setH()
      if letter in ['I', 'i']:
        self.setI()
      if letter in ['N', 'n']:
        self.setN()
      if letter in ['O', 'o']:
        self.setO()
      if letter in ['R', 'r']:
        self.setR()
      if letter in ['S', 's']:
        self.setS()
      if letter in ['T', 't']:
        self.setT()
      if letter in ['U', 'u']:
        self.setU()
      if letter in ['Z', 'z']:
        self.setZ()

  def setC( self ):
    self.curve.move( absolute=self.cursor+self.scale*(0.5+numpy.array([0.25*numpy.sqrt(2),0.25*numpy.sqrt(2)])), absangle=3*pi/4 )
    self.curve.grow( angle=3*pi/2, curvature=-2/self.scale )
    self.curve.move( absolute=self.cursor+[self.scale+self.spacing,0], absangle=0 )
    self.cursor = self.curve.current

  def setE( self ):
    self.curve.move( length=self.scale, absangle=-pi )
    self.curve.grow( length=self.scale )
    self.curve.grow( angle=pi/2, curvature=1000/self.scale )
    self.curve.grow( length=self.scale )
    self.curve.grow( angle=pi/2, curvature=1000/self.scale )
    self.curve.grow( length=self.scale )
    self.curve.move( absolute=self.cursor+[0,0.5*self.scale] )
    self.curve.grow( length=0.5*self.scale )
    self.curve.move( absolute=self.cursor+[self.scale+self.spacing, 0] )
    self.cursor = self.curve.current

  def setG( self ):
    self.curve.move( absolute=self.cursor+self.scale*(0.5+numpy.array([0.25*numpy.sqrt(2),0.25*numpy.sqrt(2)])), absangle=3*pi/4 )
    self.curve.grow( angle=7*pi/4, curvature=-2/self.scale )
    self.curve.move( absolute=self.cursor+[0.5*self.scale,0.5*self.scale], absangle=0 )
    self.curve.grow( length=0.5*self.scale )
    self.curve.move( absolute=self.cursor+[self.scale+self.spacing,0], absangle=0 )
    self.cursor = self.curve.current

  def setH( self ):
    self.curve.move( angle=pi/2 )
    self.curve.grow( length=self.scale )
    self.curve.move( length=-0.5*self.scale, absangle=0 )
    self.curve.grow( length=self.scale )
    self.curve.move( absolute=[self.cursor[0]+self.scale,self.cursor[1]], absangle=pi/2 )
    self.curve.grow( length=self.scale )
    self.curve.move( absolute=self.cursor+numpy.array([self.scale+self.spacing, 0] ), absangle=0 ) 
    self.cursor = self.curve.current
  
  def setI( self ):
    self.curve.move( absolute=self.cursor+[0.25*self.scale,0], absangle=pi/2 )
    self.curve.grow( length=self.scale )
    self.curve.move( absolute=self.cursor+[0.5*self.scale+self.spacing,0], absangle=0 )
    self.cursor = self.curve.current

  def setN( self ):
    self.curve.move( angle=pi/2 )
    self.curve.grow( length=self.scale )
    self.curve.grow( angle=3*pi/4, curvature=1000/self.scale )
    self.curve.grow( length=numpy.sqrt( 2*self.scale**2 ) )
    self.curve.grow( angle=3*pi/4, curvature=-1000/self.scale )
    self.curve.grow( length=self.scale )
    self.curve.move( absolute=self.cursor+numpy.array( [self.scale+self.spacing, 0 ] ), absangle=0 )
    self.cursor = self.curve.current

  def setO( self ):
    self.curve.move( absolute=self.cursor+[0.5*self.scale, 0] )
    self.curve.grow( angle=2*pi, curvature=-2/self.scale )
    self.curve.move( absolute=self.cursor+[self.scale+self.spacing,0] )
    self.cursor = self.curve.current

  def setR( self ):
    self.curve.move( absangle=pi/2 )
    self.curve.grow( length=self.scale )
    self.curve.grow( angle=pi/2, curvature=1e3/self.scale )
    self.curve.grow( length=0.75*self.scale )
    self.curve.grow( angle=pi, curvature=4/self.scale )
    self.curve.grow( length=0.75*self.scale )
    self.curve.move( absolute=self.cursor+[0.75*self.scale,0.5*self.scale], absangle=0 )
    self.curve.grow( angle=pi/2, curvature=4/self.scale )
    self.curve.grow( length= 0.25*self.scale )
    self.curve.move( absolute=self.cursor+[self.scale+self.spacing, 0], absangle=0 )
    self.cursor = self.curve.current

  def setS( self ):
    self.curve.move( absolute=self.cursor+[self.scale,self.scale], absangle=-pi )
    self.curve.grow( length=0.75*self.scale )
    self.curve.grow( angle=pi, curvature=-4/self.scale )
    self.curve.grow( length=0.5*self.scale )
    self.curve.grow( angle=pi, curvature=4/self.scale )
    self.curve.grow( length=0.75*self.scale )
    self.curve.move( absolute=self.cursor+[self.scale+self.spacing, 0], absangle=0 )
    self.cursor = self.curve.current
    
  def setT( self ):
    self.curve.move( length=0.5*self.scale, absangle=pi/2 )
    self.curve.grow( length=self.scale )
    self.curve.move( absolute=self.cursor+[0,self.scale], absangle=0 )
    self.curve.grow( length=self.scale )
    self.curve.move( absolute=self.cursor+[self.scale+self.spacing,0] )
    self.cursor = self.curve.current

  def setU( self ):
    self.curve.move( absolute=self.cursor+[0,self.scale], absangle=-pi/2 )
    self.curve.grow( length=0.5*self.scale )
    self.curve.grow( angle=pi, curvature=-2/self.scale )
    self.curve.grow( length=0.5*self.scale )
    self.curve.move( absolute=self.cursor+[self.scale+self.spacing,0], absangle=0 )
    self.cursor = self.curve.current

  def setZ( self ):
     self.curve.move( absolute=self.cursor + numpy.array( [0,self.scale] ) )
     self.curve.grow( length=self.scale )
     self.curve.grow( angle=3.*pi/4., curvature=1000/self.scale )
     self.curve.grow( length=numpy.sqrt( 2*self.scale**2 ) )
     self.curve.grow( angle=3.*pi/4., curvature=-1000/self.scale )
     self.curve.grow( length=self.scale )
     self.curve.move( length=self.spacing )
     self.cursor = self.curve.current
     
