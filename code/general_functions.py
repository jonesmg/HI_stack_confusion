import numpy, scipy.special, scipy.integrate

def HIMF(logMHI):
    '''
    ALFALFA 100% HI mass function Schechter model.
    
    Inputs:
    
    logMHI = log(M_HI/M_sol) 
    
    Outputs:
    
    n = Number density at log(M_HI/M_sol)
    '''

    alpha = -1.25
    phi_s = 4.5E-3
    M_s = 10.**9.94

    MHI = 10.**logMHI

    return numpy.log(10.)*phi_s*((MHI/M_s)**(alpha+1))*numpy.exp(-MHI/M_s)


#Cosmology
#WMAP9
H0 = 69.7 #km/s/Mpc
O_m = 0.2814
O_L = 1.-O_m
c = 299792.458 #km/s
O_HI = 3.9E-4
G = 6.67E-11

rho_c_0 = 3.*((H0/3.086E19)**2.)/(8.*numpy.pi*G)
rho_HI_0 = O_HI*rho_c_0*(((3.086E22)**3.)/2.E30) #M_sol/Mpc^3


def E(z):

    return numpy.sqrt((O_m*((1.+z)**3.)) + O_L)

def co_dist(z):
    
    d_H = c/H0
    
    return d_H * scipy.integrate.quad(lambda x: 1./E(x),0.,z)[0]

def phys_size(theta,z):
    '''
    Returns the physical size of the beam in Mpc at a given redshift.
    
    Inputs:
    
    theta = Angular size (radians) at z=0
    z = Redshift
    
    Outputs:
    
    l = Physical size (Mpc)
    '''
    
    theta_eff = theta*(1.+z)
    
    return co_dist(z)*theta_eff/(1.+z)


def m_av_mod(y,x):
    '''
    Integral over the 2PCF in velocity space (y-direction) and angular space (x-direction). 
    Uses the fit made in Jones et al. 2016. Both x and y should be in Mpc.
    
    Inputs:
    
    y = Integration limit in velocity space (Mpc)
    x = Integration limit in the angular direcction (Mpc)
    
    Output:
    
    m = Effective volume (Mpc^3) due to clustering in 2PCF.
    '''
    
    A, alpha, a = 12.104434  ,   -1.13210593,   0.64058959
    b = 1./a
    
    r = A**(-1./alpha)
    
    return a*((2.*numpy.pi)/((alpha+2.)*(alpha+3.)*r**alpha)) * ( (y/b)*((x/a)**2.)*((r)**alpha)*(alpha+2.)*(alpha+3.) + 2.*(y/b)*((x/a)**(2.+alpha))*(alpha+3.)*scipy.special.hyp2f1(0.5,-1.-(alpha/2.),1.5,-(y*a/(x*b))**2.) - 2.*(y/b)**(alpha+3.) )

