#! /usr/bin/env python

#---------------------------
# Kirchhoff-Love shell class
# Author: Timo van Opstal
#---------------------------

from finity import *
import numpy, time

class KLshell:
  'Kirchhoff-Love shell class'

  def __init__( self, domain, disp, X, params, form='local', theta=None ):
    '''Initialize shell object.
       domain, topology,
       disp,   function space,
       X,      reference configuration,
       params, dict containing E, h and nu, (and rho for inertia)
       form,   formulation to be used ('classic', 'parfree', 'local'),
       theta,  parametrization of domain, in case of 'classic' formulation.

       The formulations available are:
       1. local:   parametrization is based on the element coordinate system, as is common in FEM
       2. classic: parametrization is based on the auxiliary field theta, as is common in theory
       3. parfree: parametrization is bypassed, leaving no ambiguities'''

    # Tests on input
    assert domain.ndims == 2, 'Expected surface (domain.ndim = %i != 2'%domain.ndims
    assert form in ('classic', 'parfree', 'local'), 'Unknown formulation %s'%form

    # Store attributes
    self.domain = domain
    self.disp = disp
    self.X = X
    self.params = params
    self.form = form
    if form == 'classic': self.theta = theta

    # Precomputations
    self.params.setdefault( 'ischeme', 'gauss8' )
    self.c0 = .5 * params['E'] * params['h'] / (1-params['nu']**2)
    self.c1 = self.c0*params['h']**2 / 12.

  @property
  @core.cache
  def det( self ):
    J = function.localgradient( self.X, 2 )
    return function.norm2( function.cross( J[:,0], J[:,1], 0 ) )

  @property
  @core.cache
  def constit( self ):
    'Constitutive tensor.'
    nu = self.params['nu']

    if self.form in ('local','classic'):
      if self.form=='local':
        J = function.localgradient( self.X, 2 )
      else:
        J = function.grad( self.X, self.theta )
      a = function.inverse( (J[:,:,_] * J[:,_,:]).sum( 0 ), (0, 1) ) # contravariant metric tensor of ref config.
    elif self.form=='parfree':
      a = function.eye( 3 )
    else:
      raise ValueError( 'Unknown formulation %s'%self.form )

    return a[:,:,_,_] * a[_,_,:,:] * nu \
         + a[:,_,:,_] * a[_,:,_,:] * ( .5 * (1-nu) ) \
         + a[:,_,_,:] * a[_,:,:,_] * ( .5 * (1-nu) ) # constitutive tensor (upto constants c0, c1)

  def assemble( self, func, **kwargs ):
    'Integrate over parametric domain.'
    kwargs.setdefault( 'ischeme', self.params['ischeme'] )
    kwargs['iweights'] = function.IWeights() if self.form=='local' else None
    kwargs['coords'] = None if self.form=='local' else self.X
    if self.form=='local':
      if not isinstance( func, list ):
        func *= self.det
      else:
        func = [f*self.det for f in func]
    return self.domain.integrate( func, **kwargs )

  def energy_local( self, x ):
    X = self.X
    grad = function.localgradient( x, 2 )
    Grad = function.localgradient( X, 2 )
    hess = function.localgradient( grad, 2 )
    Hess = function.localgradient( Grad, 2 )
    norm = x.normal( 2 )
    Norm = X.normal( 2 )

    epsilon = 0.5 * (grad[:,:,_] * grad[:,_,:]).sum( 0 ) - \
              0.5 * (Grad[:,:,_] * Grad[:,_,:]).sum( 0 )
    kappa = (hess * norm[:,_,_]).sum( 0 ) - \
            (Hess * Norm[:,_,_]).sum( 0 )

    return self.c0 * ( (self.constit * epsilon[:,:,_,_]).sum([0,1]) * epsilon ).sum([0,1]) \
         + self.c1 * ( (self.constit * kappa[:,:,_,_]).sum([0,1]) * kappa ).sum([0,1])

  def energy_classic( self, x ):
    X = self.X
    grad = function.grad( x, self.theta )
    Grad = function.grad( X, self.theta )
    hess = function.grad( grad, self.theta )
    Hess = function.grad( Grad, self.theta )
    norm = x.normal( 2 )
    Norm = X.normal( 2 )

    epsilon = 0.5 * (grad[:,:,_] * grad[:,_,:]).sum( 0 ) - \
              0.5 * (Grad[:,:,_] * Grad[:,_,:]).sum( 0 )
    kappa = (hess * norm[:,_,_]).sum( 0 ) - \
            (Hess * Norm[:,_,_]).sum( 0 )

    return self.c0 * ( (self.constit * epsilon[:,:,_,_]).sum([0,1]) * epsilon ).sum([0,1]) \
         + self.c1 * ( (self.constit * kappa[:,:,_,_]).sum([0,1]) * kappa ).sum([0,1])

  def energy_parfree( self, x ):
    X = self.X
    grad = x.grad( X, ndims=-1 )
    hess = grad.grad( X, ndims=-1 )
    norm = x.normal( ndims=2 )
    Norm = X.normal( ndims=2 )
    Proj = function.eye( 3 ) - Norm[:,_]*Norm[_,:] # Projection onto tangent plane
    Projgrad = Proj.grad( X, ndims=-1 ) # Contains dNorm

    epsilon = 0.5*(Proj - grad[:,:,_] * grad[:,_,:]).sum(0)
    kappa = (hess * norm[:,_,_]).sum(0) - (Projgrad * Norm[:,_,_]).sum(0)

    return self.c0 * ( (self.constit * epsilon[:,:,_,_]).sum([0,1]) * epsilon ).sum([0,1]) \
         + self.c1 * ( (self.constit * kappa[:,:,_,_]).sum([0,1]) * kappa ).sum([0,1])

  def energy( self, x ):
    func = self.energy_local if self.form=='local' else \
           self.energy_classic if self.form=='classic' else \
           self.energy_parfree
    return func( x )

  def stiffness_local( self, x ):
    'Variations of internal energy in local form.'
    X = self.X
    # precompuations
    grad = function.localgradient( x, 2 )
    Grad = function.localgradient( X, 2 )
    hess = function.localgradient( grad, 2 )
    Hess = function.localgradient( Grad, 2 )
    norm = x.normal( 2 )
    Norm = X.normal( 2 )
    z = lambda df, dg: function.cross( df[...,0], dg[...,1], axis=-1 )

    epsilon = 0.5 * (grad[:,:,_] * grad[:,_,:]).sum( 0 ) - \
              0.5 * (Grad[:,:,_] * Grad[:,_,:]).sum( 0 )
    kappa = (hess * norm[:,_,_]).sum( 0 ) - \
            (Hess * Norm[:,_,_]).sum( 0 )

    wgrad = function.localgradient( self.disp, 2 )
    depsilon = (wgrad[:,:,:,_] * function.localgradient( x, 2 )[_,:,_,:] ).sum( 1 )
    depsilon = 0.5 * (depsilon + depsilon.swapaxes( 1, 2 ))
    d2epsilon = (wgrad[:,_,:,:,_] * wgrad[_,:,:,_,:]).sum( 2 )
    d2epsilon = 0.5 * (d2epsilon + d2epsilon.swapaxes( 2, 3 ))

    Var1g3 = z( wgrad, grad ) + z( grad, wgrad )
    Jac = function.norm2(z(grad, grad))
    Var1D = (Var1g3 - (norm*Var1g3).sum()[:,_]*norm[_,:])/Jac

    Var2g3 = z( wgrad[:,_], wgrad[_,:] )
    term0 = -(Var1g3[:,_,_,:] * norm[_,_,_,:]).sum( 3 ) * Var1D[_,:,:]/Jac
    Var2D = term0 + term0.swapaxes(0,1) \
          + Var2g3[:,:,:]/Jac \
          - norm[_,_,:] * (Var2g3[:,:,_,:] * norm[_,_,_,:]).sum( 3 )/Jac \
          - norm[_,_,:] * (Var1D[:,_,_,:] * Var1D[_,:,_,:]).sum( 3 )

    wdgrad = function.localgradient( function.localgradient( self.disp, 2 ), 2 )
    dkappa = (wdgrad[:,:,:,:] * norm[_,:,_,_]).sum( 1 ) \
           + (Var1D[:,:,_,_] * hess[_,:,:,:]).sum( 1 )
    d2kappa = (wdgrad[:,_,:,:,:] * Var1D[_,:,:,_,_]).sum( 2 ) \
            + (wdgrad[_,:,:,:,:] * Var1D[:,_,:,_,_]).sum( 2 ) \
            + (Var2D[:,:,:,_,_] * hess[_,_,:,:,:]).sum( 2 )

    res = 2. * self.c0 * ( (self.constit * epsilon[_,:,:,_,_]).sum([1,2]) * depsilon ).sum([1,2]) \
        + 2. * self.c1 * ( (self.constit * kappa[_,:,:,_,_]).sum([1,2]) * dkappa ).sum([1,2])
    jac = 2. * self.c0 * ( (self.constit * depsilon[:,_,:,:,_,_]).sum([2,3]) * depsilon ).sum([2,3]) \
        + 2. * self.c0 * ( (self.constit * epsilon[_,_,:,:,_,_]).sum([2,3]) * d2epsilon ).sum([2,3]) \
        + 2. * self.c1 * ( (self.constit * dkappa[:,_,:,:,_,_]).sum([2,3]) * dkappa ).sum([2,3]) \
        + 2. * self.c1 * ( (self.constit * kappa[_,_,:,:,_,_]).sum([2,3]) * d2kappa ).sum([2,3])

    return res, jac

  def stiffness_classic( self, x ):
    'Variations of internal energy in parametric (i.e. classic) form.'
    X = self.X
    # precompuations
    grad = function.grad( x, self.theta )
    Grad = function.grad( X, self.theta )
    hess = function.grad( grad, self.theta )
    Hess = function.grad( Grad, self.theta )
    norm = x.normal( 2 )
    Norm = X.normal( 2 )
    z = lambda df, dg: function.cross( df[...,0], dg[...,1], axis=-1 )

    epsilon = 0.5 * (grad[:,:,_] * grad[:,_,:]).sum( 0 ) - \
              0.5 * (Grad[:,:,_] * Grad[:,_,:]).sum( 0 )
    kappa = (hess * norm[:,_,_]).sum( 0 ) - \
            (Hess * Norm[:,_,_]).sum( 0 )

    wgrad = function.grad( self.disp, self.theta )
    depsilon = (wgrad[:,:,:,_] * grad[_,:,_,:] ).sum( 1 )
    depsilon = 0.5 * (depsilon + depsilon.swapaxes( 1, 2 ))
    d2epsilon = (wgrad[:,_,:,:,_] * wgrad[_,:,:,_,:]).sum( 2 )
    d2epsilon = 0.5 * (d2epsilon + d2epsilon.swapaxes( 2, 3 ))

    Var1g3 = z( wgrad, grad ) + z( grad, wgrad )
    Jac = function.norm2(z(grad, grad))
    Var1D = (Var1g3 - (norm*Var1g3).sum()[:,_]*norm[_,:])/Jac

    Var2g3 = z( wgrad[:,_], wgrad[_,:] )
    term0 = -(Var1g3[:,_,_,:] * norm[_,_,_,:]).sum( 3 ) * Var1D[_,:,:]/Jac
    Var2D = term0 + term0.swapaxes(0,1) \
          + Var2g3[:,:,:]/Jac \
          - norm[_,_,:] * (Var2g3[:,:,_,:] * norm[_,_,_,:]).sum( 3 )/Jac \
          - norm[_,_,:] * (Var1D[:,_,_,:] * Var1D[_,:,_,:]).sum( 3 )

    wdgrad = function.grad( wgrad, self.theta )
    dkappa = (wdgrad[:,:,:,:] * norm[_,:,_,_]).sum( 1 ) \
           + (Var1D[:,:,_,_] * hess[_,:,:,:]).sum( 1 )
    d2kappa = (wdgrad[:,_,:,:,:] * Var1D[_,:,:,_,_]).sum( 2 ) \
            + (wdgrad[_,:,:,:,:] * Var1D[:,_,:,_,_]).sum( 2 ) \
            + (Var2D[:,:,:,_,_] * hess[_,_,:,:,:]).sum( 2 )

    res = 2. * self.c0 * ( (self.constit * epsilon[_,:,:,_,_]).sum([1,2]) * depsilon ).sum([1,2]) \
        + 2. * self.c1 * ( (self.constit * kappa[_,:,:,_,_]).sum([1,2]) * dkappa ).sum([1,2])
    jac = 2. * self.c0 * ( (self.constit * depsilon[:,_,:,:,_,_]).sum([2,3]) * depsilon ).sum([2,3]) \
        + 2. * self.c0 * ( (self.constit * epsilon[_,_,:,:,_,_]).sum([2,3]) * d2epsilon ).sum([2,3]) \
        + 2. * self.c1 * ( (self.constit * dkappa[:,_,:,:,_,_]).sum([2,3]) * dkappa ).sum([2,3]) \
        + 2. * self.c1 * ( (self.constit * kappa[_,_,:,:,_,_]).sum([2,3]) * d2kappa ).sum([2,3])

    return res, jac

  def stiffness_parfree( self, x ):
    'Variations of internal energy with tangential differential calculus (i.e. parametrization-free form).'
    X = self.X
    w = self.disp

    # precompuations
    grad = x.grad( X, ndims=-1 )
    hess = grad.grad( X, ndims=-1 )
    norm = x.normal( ndims=2 )
    Norm = X.normal( ndims=2 )
    Proj = function.eye( 3 ) - Norm[:,_]*Norm[_,:] # Projection onto tangent plane
    Projgrad = Proj.grad( X, ndims=-1 ) # Contains dNorm
    wgrad = w.grad( X, ndims=-1 )
    whess = wgrad.grad( X, ndims=-1 )

    # strain and variations
    epsilon = 0.5*(Proj - grad[:,:,_] * grad[:,_,:]).sum(0)
    depsilon = (wgrad[:,:,:,_] * grad[_,:,_,:] ).sum(1)
    depsilon = 0.5 * (depsilon + depsilon.swapaxes( 1, 2 ))
    d2epsilon = (wgrad[:,_,:,:,_] * wgrad[_,:,:,_,:]).sum(2)

    # precompuations
    def zfunc( df, dg ):
      temp = (function.cross( df[...,:,_,:], dg[...,_,:,:], -1 )*Norm).sum(-1)
      levicivita = numpy.zeros( 3*(3,) ) # TODO: special functionality for function.cross?
      levicivita[0,1,2] = levicivita[1,2,0] = levicivita[2,0,1] =  1
      levicivita[2,1,0] = levicivita[1,0,2] = levicivita[0,2,1] = -1
      return (levicivita[...,:,:,:]*temp[...,_,:,:]).sum([-1,-2])
    dz = zfunc( grad, wgrad )
    d2z = zfunc( wgrad[:,_], wgrad[_,:] )
    dnorm = dz - norm[_,:]*(norm*dz).sum(-1)[:,_]
    temp = (norm*dz[:,_]).sum(-1)[:,:,_]*dnorm[_,:,:]
    d2norm = d2z \
           - norm[_,_,:]*(norm*d2z).sum(-1)[:,:,_] \
           - (dnorm[:,_]*dnorm[_,:]).sum(-1)[:,:,_]*norm[_,_,:] \
           - temp - temp.swapaxes( 0, 1 )

    # curvature and variations
    kappa = (hess * norm[:,_,_]).sum(0) - (Projgrad * Norm[:,_,_]).sum(0)
    dkappa = (whess[:,:,:,:] * norm[_,:,_,_]).sum(1) \
           + (hess[_,:,:,:] * dnorm[:,:,_,_]).sum(1)
    d2kappa = (whess[:,_,:,:,:] * dnorm[_,:,:,_,_]).sum(2) \
            + (whess[_,:,:,:,:] * dnorm[:,_,:,_,_]).sum(2) \
            + (hess[_,_,:,:,:] * d2norm[:,:,:,_,_]).sum(2)

    # residual and jacobian
    res = 2. * self.c0 * ( (self.constit * epsilon[_,:,:,_,_]).sum([1,2]) * depsilon ).sum([1,2]) \
        + 2. * self.c1 * ( (self.constit * kappa[_,:,:,_,_]).sum([1,2]) * dkappa ).sum([1,2])
    jac = 2. * self.c0 * ( (self.constit * depsilon[:,_,:,:,_,_]).sum([2,3]) * depsilon ).sum([2,3]) \
        + 2. * self.c0 * ( (self.constit * epsilon[_,_,:,:,_,_]).sum([2,3]) * d2epsilon ).sum([2,3]) \
        + 2. * self.c1 * ( (self.constit * dkappa[:,_,:,:,_,_]).sum([2,3]) * dkappa ).sum([2,3]) \
        + 2. * self.c1 * ( (self.constit * kappa[_,_,:,:,_,_]).sum([2,3]) * d2kappa ).sum([2,3])

    return res, jac

  def stiffness( self, x ):
    func = self.stiffness_local if self.form=='local' else \
           self.stiffness_classic if self.form=='classic' else \
           self.stiffness_parfree
    return func( x )

  def inertia( self, x0, x1, x2, tau ):
    'Inertia term given previous levels x0, x1 and current level x2 (Implicit Euler).'
    assert self.params.has_key( 'rho' ), 'density of shell object required in self.params'
    res = (self.disp*x2).sum() * self.params['rho'] * tau**-2 \
        - (self.disp*x1).sum() * self.params['rho'] * tau**-2 * 2 \
        + (self.disp*x0).sum() * self.params['rho'] * tau**-2
    # res = (self.disp*(x2 - 2*x1 + x0)).sum()*self.params['rho']*tau**-2 # TODO: what's wrong here?
    jac = (self.disp[:,_]*self.disp[_,:]).sum()*self.params['rho']*tau**-2
    return res, jac

  def load( self, load, live=False ):
    if callable( load ):
      raise NotImplementedError
    elif live:
      raise NotImplementedError
    else:
      res = -(self.disp * numpy.asarray( load )).sum()
      return res
    
  def solve( self, constrain, x0=None, relax=1, TOL=1e-12, phys={} ):
    '''Newton solve generator given
    constrain, essential bcs,
    x0,        initial guess,
    relax,     relaxation of Newton iteration,
    TOL,       linear solve tolerance (0 = direct),
    phys,      physics dictionary: phys['load']=load,
                                   phys['inertia']=(x0, x1, tau)'''
    log.context( 'kl' )
    # TODO: check for invalid keys in phys... .pop()?
    x = x0 if x0 else self.X
    upd = 0

    while True:
      # Assemble
      res, jac = self.stiffness( x )
      if phys.has_key( 'load' ):
        res += self.load( phys['load'] )
      if phys.has_key( 'inertia' ):
        x0, x1, tau = phys['inertia']
        temp = self.inertia( x0, x1, x, tau )
        res += temp[0]
        jac += temp[1]
      vec, mat = self.assemble( [res, jac], force_dense=not bool(TOL), title='assemble[%i]'%self.disp.shape[0] )

      # Print and yield
      err = numpy.linalg.norm( constrain | vec )
      if locals().has_key('t0'): elapsed( t0 )
      yield x, err, upd

      # Solve
      t0 = time.time()
      upd = mat.solve( vec, constrain=constrain, tol=TOL, precon='spilu' )
      elapsed( t0 )

      # New iterate
      t0 = time.time()
      x -= relax*self.disp.dot( constrain | upd )

def elapsed( t0 ):
  dt = time.time() - t0
  hours   = int( dt // 3600 )
  minutes = int( dt // 60 - 60 * hours )
  seconds = int( dt // 1 - 60 * minutes - 3600 * hours )
  log.info( 'elapsed %d:%02d:%02d'%(hours, minutes, seconds) )

def slroof( verify=True, nelems=16, jmax=2**4, form='parfree', transform=False ):
  '''Scordelis-Lo roof benchmark
  I: verify,    perform some verification tests,
     nelems,    element count in 1 direction,
     jmax,      max # nonlinear iterations,
     form,      formulation: 'classic' or 'parfree',
     transform, if 'classic' skew parametrization,
  O: None.'''
  xmidref = numpy.array( [1.6069807987149836e+01, -3.3306690738754696e-16, -5.8489885841615301e+00] )

  # Discretization
  grid = numpy.linspace( -.5, .5, nelems+1 )
  domain, coords = mesh.rectilinear( 2*(grid,) )
  disp = domain.splinefunc( degree=2*(3,) ).vector( 3 )
  assert not numpy.mod( nelems, 2 ), 'need even nelems, got %i'%nelems
  midpoint = domain[nelems/2:,nelems/2:].boundary['bottom'].boundary['right']
  midpointdisp = lambda func: midpoint.integrate( func, coords=X, ischeme='none' )

  # Geometry
  xi, eta = coords
  length=50; radius=25; angle=40
  phi = angle * xi * numpy.pi / 90.
  slroof = function.stack( [ radius * function.sin(phi), length * eta, radius * (function.cos(phi)-1) ] )
  if transform: coords = function.stack( (xi, .5*function.sin( numpy.pi*eta )) )
  X = slroof # domain.projection( slroof, onto=disp, coords=coords, ischeme='gauss3' ) # This projection is not strictly necessary, but makes 'parfree' code run much faster.

  # Physics
  clamps = domain.boundary['top'] + domain.boundary['bottom'] 
  cons = clamps.project( 0., onto=disp[:,0], coords=coords, title='proj[bcs x]' ) \
       | clamps.project( 0., onto=disp[:,2], coords=coords, title='proj[bcs z]' )
  params = {'E':1., 'nu':0., 'h':.25, 'ischeme':'gauss7'}
  phys = {'load':[0.,0.,5.e-6/3.]}
  # params = {'E':4.32e8, 'nu':0., 'h':.25, 'ischeme':'gauss7'}
  # phys = {'load':[0.,0.,90.]}
  roof = KLshell( domain, disp, X, params, form=form, theta=coords )

  # Pre processing
  if verify:
    TOL = 1.e-2 if nelems==2 else 1.e-5
    assert numpy.linalg.norm( midpointdisp( X ) - xmidref )/numpy.linalg.norm( xmidref ) < TOL, 'midpoint verification failed'

  # Solve
  TOL = 1e-5 # doesn't converge if set lower!
  j, errs = 0, []
  for x, err, upd in roof.solve( constrain=cons, phys=phys, TOL=TOL ):
    errs.append( err )
    d = midpointdisp(x-X)[2]
    log.info( 'j:{0:d}, d(.5):{1:6f}, err:{2:1e}'.format( j, d, err ) )
    # plot.writevtu( './slroof{0:02d}.vtu'.format( j ), domain.refine( 3 ), x )
    if (verify and j==1 ) or (err<TOL or j>jmax):
      xdofs = domain.project( x, onto=disp, coords=X, ischeme=roof.params['ischeme'] )
      pickle.dump( {'xdofs':xdofs, 'd':d},
                   open( 'slroof-%sT%s-ne%i.pck' % (form, transform, nelems) ) )
      break
    j += 1

    # Finite difference approximation
    if verify and False:
      'Preparatory calculations'
      dofs = domain.project( x, onto=disp, coords=X ) | 0.
      dres = domain.integrate( 
             roof.stiffness( x )[1],
             iweights=function.IWeights(), ischeme='gauss8', title='jacobian' )
      resf = lambda xlin: roof.stiffness( xlin )[0]
      title = 'fd jac %% 3i/% 3i'%len( dofs )

      'Do FD approx for series of deltas'
      deltarange = 10.**-numpy.arange( 3, 7 )
      diffnorm = []
      for delta in deltarange:
        dres_fd = fdapprox( resf, disp, dofs, delta=delta )
        temp = numpy.array( [domain.integrate( fi, iweights=function.IWeights(), ischeme='gauss8', title=title%i ) for i, fi in enumerate( dres_fd )] )
        diff = dres.toarray() - temp
        diffnorm.append( numpy.linalg.norm( diff ) )
        with plot.PyPlot( 'rel_err_jac' ) as fig:
          fig.cspy( diff/temp )
      with plot.PyPlot( 'fd-conv' ) as fig:
        fig.xlabel( r'$\delta$' )
        fig.ylabel( r'$\| \partial K - \partial K_\delta \|_\mathrm{HS}$' )
        fig.title( 'Finite difference convergence stiffness $\partial K$' )
        fig.loglog( deltarange, diffnorm, 'k.-' )
        fig.slope_triangle( deltarange, diffnorm )
        fig.grid( True )
      log.debug( 'Finished FD convergence of stiffness' )

  # Post processing
  if verify:
    zdisp = 0.025734 if nelems==2 else 0.29615383 # use literature value unless lowest resolution
    assert numpy.abs( zdisp + midpointdisp( x-X )[2] ) < 1.e-4, 'scordelis-lo roof benchmark failed'
    log.debug( 'scordelis-lo roof benchmark passed' )

if __name__ == '__main__':
  util.run( slroof )

# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
