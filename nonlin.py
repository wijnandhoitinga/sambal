from nutils import *

##########################################
# Newton-Raphson solver                  #
##########################################

class NewtonSolver( object ):

  def __init__ ( self, system, state=None ):
    self.system = system

  def solve ( self, rtol, maxiter=25, atol=1e-12, **linearsolverargs ):

    cons = self.system.get_total_cons()-self.system.state

    residual, tangent = self.system.get_residual_and_tangent()

    for iiter in log.range( 'Newton iterations', maxiter ):

      self.system.state += tangent.solve( -residual, constrain=cons, **linearsolverargs )
      residual, tangent = self.system.get_residual_and_tangent()
      
      rnorm = numpy.linalg.norm(residual[~cons.where])
      if iiter == 0: #Predictor step
        rscale = rnorm
        log.info( 'Predictor residual: %5.4e' % rscale )
        rnorm  = 1. if rscale > atol else 0.
        cons&=0
      else: #Corrector steps
        rnorm = (1./rscale)*rnorm
        log.info( 'Iteration %d: %5.4e' % (iiter,rnorm) )

      if rnorm < rtol:
        break

    else:
      log.error('Newton solver did not converge in %d iterations' % maxiter )

    return 0

##########################################
# Partitioned solver                     #
##########################################

class PartitionedSolver( object ):

  def __init__ ( self, system, doforder, state=None ):
    self.system = system
    self.doforder = doforder
    assert all(field in self.system.dofgroups for field in doforder)

  def solve ( self, **linearsolverargs ):

    for ifield in self.doforder:

      #Select the subproblem
      log.info( 'Solving subproblem "%s" with %d DOFs' % (ifield,len(self.system.dofgroups[ifield])) )

      #Get the constraints for the current subproblem
      icons = getattr( self.system, 'get_total_cons_' + ifield )()

      #Constrain all the dofs that are not in the current subproblem
      for jfield in self.doforder:
        if ifield == jfield:
          continue
        jdofs = self.system.dofgroups[jfield]
        icons[list(jdofs)] = self.system.state[list(jdofs)]

      #Get the lhs and rhs for the current subproblem
      ilhs, irhs = getattr( self.system, 'get_residual_and_tangent_' + ifield)()

      #Solve the subproblem
      self.system.state = ilhs.solve( rhs=irhs, constrain=icons, **linearsolverargs )

    return True

##########################################
# System base class                      #
##########################################

class System ( object ):

  def __init__ ( self, size, state=None ):
    if state is not None:
      assert state.ndim==1 and state.size==size
    else:
      state = numpy.zeros( size )
    self.state = state  

  def solve ( self, solver=NewtonSolver, **kwargs ):
    return solver( self ).solve( **kwargs )

if __name__ == "__main__":
  
  #######################################
  # Nonlinear beam                      #
  #######################################

  class NonlinearBeamSystem ( System ):
  
    def __init__ ( self, topo, geom, funcsp, E, I, A, q, size ):
      super(NonlinearBeamSystem, self).__init__( size )
      self.topo = topo
      self.geom = geom
      self.funcsp = funcsp
      self.EI = E*I
      self.EA = E*A
      self.q  = q

    def rfunc ( self ):
      w = funcsp.dot( self.state )
      return self.EI*w.laplace(geom)*self.funcsp.laplace(geom) + 3./2.*self.EA*(w.grad(geom)[0]**3)*self.funcsp.grad(geom)[:,0]-self.q*self.funcsp

    def tanfunc ( self ):
      w = funcsp.dot( self.state )
      return self.EI*function.outer(self.funcsp.laplace(geom)) + 9./2.*self.EA*(w.grad(geom)[0]**2)*function.outer(self.funcsp.grad(geom)).sum(-1)
      
    def get_residual ( self ):
      return self.topo.integrate( self.rfunc(self.state), geometry=self.geom, ischeme='gauss3' )

    def get_tangent ( self ):
      return self.topo.integrate( self.tanfunc(self.state), geometry=self.geom, ischeme='gauss3' )

    def get_residual_and_tangent ( self ):
      return self.topo.integrate( [self.rfunc(),self.tanfunc()], geometry=self.geom, ischeme='gauss3' )

    def get_total_cons ( self ):
      cons  = self.topo.boundary['left'].project( 0, onto=self.funcsp, geometry=self.geom, ischeme='gauss1' )
      cons |= self.topo.boundary['left'].project( 0, onto=self.funcsp.grad(geom)[:,0], geometry=self.geom, ischeme='gauss1' )
      return cons

  L = 1.
  q = 4.
  E = 1.
  A = 1.
  I = 1.
  n = 10

  topo, geom = mesh.rectilinear([numpy.linspace(0,L,n+1)])
  funcsp = topo.splinefunc( degree=2 )   

  system = NonlinearBeamSystem( topo=topo, geom=geom, funcsp=funcsp, E=E, I=I, A=A, q=q, size=funcsp.size )

  system.solve( rtol=1e-6 )

  log.info( 'Tip deflection: %4.3f (%4.3f)' % (system.state[-1],q*L**4/(8.*E*I)) )
