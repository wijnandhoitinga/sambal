#! /usr/bin/env python

#---------------------------
# Kirchhoff-Love shell class
# Author: Timo van Opstal
#---------------------------

from nutils import *
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

    # Function definitions
    self.energy, self.stiffness = {
        'local':(self.energy_local,   self.stiffness_local),
      'classic':(self.energy_classic, self.stiffness_classic),
      'parfree':(self.energy_parfree, self.stiffness_parfree)}[self.form]

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
    kwargs['iweights'] = self.det * function.IWeights() if self.form=='local' else None
    kwargs['coords'] = None if self.form=='local' else self.X
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

    return 0.5 * self.c0 * ( (self.constit * epsilon[:,:,_,_]).sum([0,1]) * epsilon ).sum([0,1]) \
         + 0.5 * self.c1 * ( (self.constit * kappa[:,:,_,_]).sum([0,1]) * kappa ).sum([0,1])

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

    return 0.5 * self.c0 * ( (self.constit * epsilon[:,:,_,_]).sum([0,1]) * epsilon ).sum([0,1]) \
         + 0.5 * self.c1 * ( (self.constit * kappa[:,:,_,_]).sum([0,1]) * kappa ).sum([0,1])

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

    return 0.5 * self.c0 * ( (self.constit * epsilon[:,:,_,_]).sum([0,1]) * epsilon ).sum([0,1]) \
         + 0.5 * self.c1 * ( (self.constit * kappa[:,:,_,_]).sum([0,1]) * kappa ).sum([0,1])

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
    grad = x.grad( X, 2 )
    hess = grad.grad( X, 2 )
    norm = x.normal( 2 )
    Norm = X.normal( 2 )
    Proj = function.eye( 3 ) - Norm[:,_]*Norm[_,:] # Projection onto tangent plane
    Projgrad = Proj.grad( X, 2 )
    wgrad = w.grad( X, 2 )
    whess = wgrad.grad( X, 2 )

    # strain and variations
    epsilon = 0.5*(Proj - (grad[:,:,_] * grad[:,_,:]).sum(0))
    depsilon = (wgrad[:,:,:,_] * grad[_,:,_,:]).sum(1)
    depsilon = -0.5 * (depsilon + depsilon.swapaxes(1,2))
    d2epsilon = -(wgrad[:,_,:,:,_] * wgrad[_,:,:,_,:]).sum(2)

    # precompuations
    def zfunc( df, dg ):
      temp = (function.cross( df[...,:,_,:], dg[...,_,:,:], -1 )*Norm).sum(-1)
      levicivita = numpy.zeros( 3*(3,) ) # TODO: special functionality for function.cross?
      levicivita[0,1,2] = levicivita[1,2,0] = levicivita[2,0,1] =  1
      levicivita[2,1,0] = levicivita[1,0,2] = levicivita[0,2,1] = -1
      return (levicivita[...,:,:,:]*temp[...,_,:,:]).sum([-1,-2])
    dz = zfunc( grad, wgrad )
    d2z = zfunc( wgrad[:,_], wgrad[_,:] )
    dnorm = dz - norm[_,:]*(norm[_,:]*dz).sum(1)[:,_]
    temp = (norm[_,:]*dz[:,:]).sum(1)[:,_,_]*dnorm[_,:,:]
    d2norm = d2z - norm[_,_,:]*(norm[_,_,:]*d2z).sum(2)[:,:,_] \
           - (dnorm[:,_,:]*dnorm[_,:,:]).sum(-1)[:,:,_]*norm[_,_,:] \
           - temp - temp.swapaxes( 0, 1 )

    # curvature and variations
    kappa = (hess[:,:,:] * norm[:,_,_]).sum(0) - (Projgrad * Norm[:,_,_]).sum(0)
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
    t0 = time.time()
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
      log.info( 'func obj compiled in %.1f seconds' % (time.time()-t0) )
      t0 = time.time()
      vec, mat = self.assemble( [res, jac], title='assemble[%i]'%self.disp.shape[0] )

      # Print and yield
      err = numpy.linalg.norm( constrain | vec )
      log.info( 'assembled in %.1f seconds' % (time.time()-t0) )
      yield x, err, upd

      # Solve
      upd = mat.solve( vec, constrain=constrain, tol=TOL, precon='spilu' )

      # New iterate
      t0 = time.time()
      x -= relax*self.disp.dot( constrain | upd )

def scordelislo( domain, coords, case='lo' ):
  '''Scordelis-Lo roof benchmark,
  I: domain,  StructuredTopology instance,
     coords,  parametric coordinate function,
     case,    formulation in ('lo', 'tr', 'pf'),
  O: x,       displaced position,
     d,       midpoint displacement,
     W,       total internal energy.'''
  log.context( 'scordelis-lo' )

  # Geometry
  xi, eta = coords
  length, radius, angle = 50, 25, 40
  phi = angle * xi * numpy.pi / 90.
  x0 = function.stack( [radius * function.sin(phi), length * eta, radius * (function.cos(phi)-1)] )

  # Basis
  nelems, temp = domain.structure.shape # Assume bivariate StructuredTopology
  assert nelems == temp, 'Assumes equal element count in both directions.'
  assert not numpy.mod( nelems, 2 ), 'need even nelems, got %i'%nelems
  disp = domain.splinefunc( degree=2 ).vector(3)
  midpoint = domain[nelems/2:,nelems/2:].boundary['bottom'].boundary['right']
  midpointdisp = lambda func: midpoint.integrate( func, coords=coords, ischeme='none', title='midpointdisp' )
  ischeme = 'gauss%i'%7 # {'pf':4, 'tr':3, 'lo':2}[case] # TODO: choose optimal orders

  # Physics
  clamps = domain.boundary['top'] + domain.boundary['bottom'] 
  cons = clamps.project( 0., onto=disp[:,0], coords=coords, ischeme=ischeme, title='proj[bcs x]' ) \
       | clamps.project( 0., onto=disp[:,2], coords=coords, ischeme=ischeme, title='proj[bcs z]' )
  params = {'E':4.32e8, 'nu':0., 'h':.25, 'ischeme':ischeme} # dimensionless: {'E':1., 'nu':0., 'h':.25, 'ischeme':ischeme}
  phys = {'load':[0.,0.,-90.]} #                                              {'load':[0.,0.,-5.e-6/3.]}
  roof = KLshell( domain, disp, x0, params,
                  form={'pf':'parfree', 'tr':'classic', 'lo':'local'}[case],
                  theta=function.stack( [xi, .5*function.sin(numpy.pi*eta)] ) if case=='tr' else None )

  # Solve
  for x, err, upd in roof.solve( constrain=cons, phys=phys, TOL=1.e-7 ):
    j = 0 if not locals().has_key('j') else j+1
    d = midpointdisp(x-x0)[2]
    W = float( roof.assemble( roof.energy( x ), title='W' ) )
    log.info( '#el/j:%i/%i, d(.5):%.6f, W:%.4e' % (nelems, j, d, W) )
    if j: return x, d, W

def conv( case='lo', levels=3, test=scordelislo ):
  'Convergence test for KL shell with b-splines.'
  # Coarse grid, reference geometry
  domain0, coords = mesh.rectilinear( 2*(numpy.linspace(-.5,.5,3),) )

  # Compute errors
  errh2, d, W = [], [], []
  nrange = numpy.arange( levels, -1, -1 )
  for n in nrange:
    if n == levels: # Reference configuration
      domain = domain0.refine(n)
      xref, dref, Wref = test( domain, coords, case=case )
      h2norm = lambda x: numpy.sqrt( domain.integrate(
               (x**2).sum(0) +
               (x.grad(coords)**2).sum([0,1]) +
               (x.grad(coords).grad(coords)**2).sum([0,1,2]),
               coords=coords, ischeme='gauss7', title='h2norm' ) )
    else:
      xi, di, Wi = test( domain0.refine(n), coords, case=case )
      errh2.append( h2norm( xi-xref ) )
      d.append( di )
      W.append( Wi )

  # Plot
  x = 50 * 2.**-nrange[:-1] # h, based on scordelis-lo benchmark
  def convplot( y, title ):
   with plot.PyPlot( 'conv' ) as fig:
     fig.loglog( x, y, 'k.-' )
     slope = fig.slope_triangle( x[:2], y[:2] )
     fig.xlabel( '$h$' )
     fig.title( title + ' (case:%s, d:%.6f)'%(case, dref) )
   return slope
  slope_d = convplot( numpy.abs(numpy.asarray(d)/dref-1.), 'rel. err. midpoint displ.' )
  slope_W = convplot( numpy.abs(numpy.asarray(W)/Wref-1.), 'rel. err. potential energy' )
  slope_n = convplot( errh2/h2norm(xref), 'rel. err. in $H^2$-norm' )

  # Verify results
  if levels > 4 and test is scordelislo:
    assert slope_d > 3.7 and slope_W > 3.7 and slope_n > 1.1, 'Convergence rate insufficient.'
    numpy.testing.assert_almost_equal( dref, -0.3005511, decimal=4,
        err_msg='Midpoint displacement (=%.4f) incorrect.'%dref )
  else:
    log.info( 'No code verification performed.' )

def verify( case='lo' ):
  'Quick code verification.'
  domain, coords = mesh.rectilinear( 2*(numpy.linspace(-.5,.5,9),) )
  dref = scordelislo( domain, coords, case=case )[1]
  numpy.testing.assert_almost_equal( dref, -0.242468, decimal=4,
        err_msg='Midpoint displacement (=%.4f) incorrect.'%dref )

if __name__ == '__main__':
  util.run( verify, conv )
# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
