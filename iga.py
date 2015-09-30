import dxfgrabber
import scipy.io

from nutils import *

class bspline ( object ):

  def __init__ ( self, order, knotvalues, knotmultiplicities, controlpoints ):
    self.order = order
    self.knotvalues = knotvalues
    self.knotmultiplicities = knotmultiplicities
    self.controlpoints = controlpoints

  def __str__ ( self ):
    return '\n'.join( ['pu = {0}, pv = {1}'.format( *self.order ),
                       'ku = {0}, kv = {1}'.format( *map( len, self.knotvalues ) ),
                       'n_cps = {0}'.format( self.controlpoints.shape[0] )  ] )

  @property
  def topology( self ):
    domain, coords = mesh.rectilinear( self.knotvalues )
    return domain

  @property
  def basis ( self ):
    return self.topology.basis( 'bspline', degree=self.order, knotvalues=self.knotvalues, knotmultiplicities=self.knotmultiplicities )

  def mesh ( self ): 
    geom =self.basis.vector(2).dot( self.controlpoints.T.ravel() )
    return self.topology, geom

def read_from_autocad ( fname, must_be_uniform=False ):

  dxf = dxfgrabber.readfile( fname )

  assert dxf.dxfversion=="AC1024", "Please save as AutoCAD R2010"
  assert len(dxf.entities)==1, "This script presumes a single entity"

  acis_data   = dxf.entities[0].acis

  line = acis_data.pop(0)
  while len(acis_data) > 0:

      if line.startswith("spline-surface"):

          props_list = line.split()

          k_v = int(props_list[-1]) #Number of knots in v-direction
          k_u = int(props_list[-2]) #Number of knots in u-direction

          assert props_list[-5]=="open" and props_list[-6]=="open", "Spline surface must be open"

          p_v = int(props_list[-7]) #Order in v-direction
          p_u = int(props_list[-8]) #Order in u-direction

          assert props_list[-9]=="nubs" and props_list[-10]=="full"

          u_knots_list = []
          while len(u_knots_list) < 2*k_u:
            add = acis_data.pop(0).split()
            assert len(add)>0, 'Expected additional u-knot data'
            u_knots_list += add

          u_knots = map(float,u_knots_list[0::2])

          if must_be_uniform==True:
            delta = numpy.array(u_knots[1:])-numpy.array(u_knots[:-1])
            assert numpy.std( delta ) < 1e-6 * numpy.mean( delta ), 'Non-uniform u-knot vector not permitted'

          u_mults = map(int,u_knots_list[1::2])
          u_mults[0]=p_u+1
          u_mults[-1]=p_u+1

          v_knots_list = []
          while len(v_knots_list) < 2*k_v:
            add = acis_data.pop(0).split()
            assert len(add)>0, 'Expected additional v-knot data'
            v_knots_list += add

          v_knots = map(float,v_knots_list[0::2])

          if must_be_uniform==True:
            delta = numpy.array(v_knots[1:])-numpy.array(v_knots[:-1])
            assert numpy.std( delta ) < 1e-6 * numpy.mean( delta ), 'Non-uniform v-knot vector not permitted'

          v_mults = map(int,v_knots_list[1::2])
          v_mults[0]=p_v+1
          v_mults[-1]=p_v+1

          n_v = sum(v_mults)-p_v-1
          n_u = sum(u_mults)-p_u-1

          cps = numpy.array([map(float,acis_data.pop(0).split())[:2] for i in range(n_u*n_v)])
          cps = cps.reshape( n_v, n_u, 2 ).transpose((1,0,2)).reshape( n_u*n_v, 2 )

      line = acis_data.pop(0)

  return bspline( (p_u,p_v), (u_knots,v_knots), (u_mults,v_mults), cps )
