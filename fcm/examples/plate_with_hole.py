#!/usr/bin/env python3

from nutils import *
from sambal.fcm import voxel
import json

def main ( fname        = 'plate_with_hole.json',
           ncoarsegrain = 2                ):

  #Load the voxel data file
  voxeldata = voxel.jsonread( fname )

  for icg in range(ncoarsegrain+1):

    #Print an overview of the voxel data
    log.info( voxeldata )

    #Unpack the topology and the geometry function
    topo, geom = voxeldata.mesh

    #Plot the voxel data
    points, values = topo.elem_eval( [geom,voxeldata.func], ischeme='bezier2', separate=True )
    with plot.PyPlot( 'voxeldata' ) as plt:
      plt.mesh( points, values )
      plt.colorbar()

    #Coarsegrain the data  
    voxeldata = voxeldata.coarsegrain()
  

def create_data( R     = 1.     ,
                 L     = 4.     , 
                 nvox  = [50,50],
                 nint  = 8      ,
                 dtype = '<i2' ,
                 fbase = 'plate_with_hole' ):

  #Construct the voxel grid
  topo, geom = mesh.rectilinear( [numpy.linspace(0,L,nv+1) for nv in nvox] )

  #Construct the levelset function
  lvl = (geom**2).sum(-1) - R**2

  #Compute and save the volume fractions
  data = topo.elem_mean( function.heaviside( lvl ), geometry=geom, ischeme='uniform%s'%nint   )
  nf   = numpy.iinfo(dtype)
  data = (nf.max-nf.min)*data+nf.min

  data.astype( dtype ).tofile( fbase + '.raw' )

  #Write the json file
  with open( fbase + '.json', 'w' ) as fout:
    json.dump( { 'FNAME'     : fbase + '.raw'                 ,
                 'THRESHOLD' : 0                              ,
                 'DIMS'      : nvox                           ,
                 'SIZE'      : [L/float(nv) for nv in nvox]  }, fout )

util.run( main, create_data )
