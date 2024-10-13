"""
This module provides a basic numerical toolkit
"""

import numpy as np


# ROOT FINDING
# ============

def root_bisection(f, a, b, roots, eps=1e-3):
    """
    recursive bisection algorithm for finding multiple roots
    of a function f(x)

    args: f - function f(x)
          a - left endpoint of start interval
          b - right endpoint of start interval
          roots - numpy array of roots
          eps - tolerance

    returns: estimate of x for which f(x) = 0
    """
    # midpoint
    x = 0.5*(a+b)

    # break recursion if x is an exact solution
    if f(x) == 0:
        roots = np.append(roots, x)
        print("found {:d}. solution (exact)".\
              format(len(roots)))
    # break recursion if tolerance is reached
    elif abs(b-a) <= eps:
        roots = np.append(roots, x)
        print("found {:d}. solution,".format(len(roots)),
              "deviation f(x) = {:6e}".format(f(x)))
    # continue recursion if function crosses zero
    # in any subinterval
    else:
        if f(a)*f(x) <= 0:
            roots = root_bisection(f, a, x, roots, eps)
        if f(x)*f(b) <= 0:
            roots = root_bisection(f, x, b, roots, eps)

    return roots


def root_newton(f, df, x0, eps=1e-3, imax=100):
    """
    Newton–Raphson algorithm for finding the root of a function f(x)

    args: f - function f(x)
          df - derivative df/dx
          x0 - start point of iteration
          eps - tolerance
          imax - maximal number of iterations
          verbose - print additiontal information if true

    returns: estimate of x for which f(x) = 0
    """

    for i in range(imax):
        x = x0 - f(x0)/df(x0)

        if abs(x - x0) < eps:
            print("tolerance reached after {:d} iterations".format(i+1))
            print("deviation: f(x) = {:.3e}".format(f(x)))
            return x

        x0 = x

    print("exceeded {:d} iterations".format(i+1), "without reaching tolerance")
    return x


# FINITE DIFFERENCES
# ==================

def derv_center2(f, x, h):
    """
    approximates derivative of a function
    by second-order centered differences

    args: f - function f(x)
          x - points for which df/dx is computed
          h - backward/forward difference

    returns: approximation of df/dx
    """
    return (f(x+h) - f(x-h))/(2*h)


def dderv_center2(f, x, h):
    """
    approximates second derivative of a function
    by second-order centered differences

    args: f - function f(x)
          x - points for which df/dx is computed
          h - backward/forward difference

    returns: approximation of d^2 f/dx^2
    """
    return (f(x+h) - 2*f(x) + f(x-h))/h**2


# NUMERICAL INTEGRATION
# =====================

def integr_trapez(f, a, b, n):
    """
    numerical integration of a function f(x)
    using the trapezoidal rule

    args: f - function f(x)
          a - left endpoint of interval
          b - right endpoint of interval
          n - number of subintervals

    returns: approximate integral
    """
    n = int(n)

    # integration step with exception handling
    try:
        if n > 0:
            h = (b - a)/n
        else:
            raise ValueError
    except ValueError:
        print("Invalid argument: n must be positive")
        return None

    # endpoints of subintervals between a+h and b-h
    x = np.linspace(a+h, b-h, n-1)

    return 0.5*h*(f(a) + 2*np.sum(f(x)) + f(b))


def integr_simpson(f, a, b, n):
    """
    numerical integration of a function f(x)
    using Simpson's rule

    args: f - function f(x)
          a - left endpoint of interval
          b - right endpoint of interval
          n - number of subintervals (positive even integer)

    returns: approximate integral
    """

    # need even number of subintervals
    n = max(2, 2*int(n/2))

    # integration step
    h = (b - a)/n

    # endpoints of subintervals (even and odd multiples of h)
    x_even = np.linspace(a+2*h, b-2*h, n/2-1)
    x_odd  = np.linspace(a+h, b-h, n/2)

    return (h/3)*(f(a) + 2*np.sum(f(x_even)) + 4*np.sum(f(x_odd)) + f(b))


# first ORDER INITIAL VALUE PROBLEMS
# ===================================

def rk4_step(f, t, x, h, *args):
    """
    fourth-order Runge-Kutta- step for function x(t)
    given by first order differential equation

    args: f - function determining second derivative
          t - value of independent variable t
          x - value of x(t)
          h - time step
          args - parameters

    returns: iterated value for t + h
    """
    k1 = h * f(t, x, *args)
    k2 = h * f(t + 0.5*h, x + 0.5*k1, *args)
    k3 = h * f(t + 0.5*h, x + 0.5*k2, *args)
    k4 = h * f(t + h, x + k3, *args)

    return x + (k1 + 2*(k2 + k3) + k4)/6

# SECOND ORDER INITIAL VALUE PROBLEMS
# ===================================

def euler_step(f, t, x, xdot, h, *args):
    """
    semi-implicit Euler step for function x(t)
    given by second order differential equation

    args: f - function determining second derivative
          t - value of independent variable t
          x - value of x(t)
          xdot - value of first derivative dx/dt
          h - time step
          args - parameters

    returns: iterated values for t + h
    """
    v = xdot + h*f(t, x, xdot, *args)

    return ( x + h*v, v )

def rkn3_step(f, t, x, xd, h, *args):
    """
    third-order Runge-Kutta-Nystroem step for function x(t)
    given by second order differential equation

    args: f - function determining second derivative
          t - value of independent variable t
          x - value of x(t)
          xdot - value of first derivative dx/dt
          h - time step
          args - parameters

    returns: iterated values for t + h
    """
    hsq = h*h
    a = np.array( [0, 2/7, 7/15, 35/38, 1] )
    c = np.array( [229/1470, 0, 600/1813, 361/27195] )
    cd = np.array( [79/490, 0, 2175/3626, 2166/9065] )
    g = np.array( [[0, 0, 0, 0],
                   [2/49, 0, 0, 0],
                   [2009/40500, 2401/40500, 0, 0],
                   [15925/109744, 0,30625/109744, 0],
                   [229/1470, 0,600/1813, 361/27195]] )
    b = np.array( [[0, 0, 0, 0],
                   [2/7, 0, 0, 0],
                   [77/900, 343/900, 0, 0],
                   [805/1444, -77175/54872, 97125/54872, 0],
                   [79/400, 0, 2175/3626, 2166/9065]] )

    fi = np.zeros(4)
    fi[0] = f(t, x, xd0, *args)
    cifisum = fi[0] * c[0]
    cdifisum = fi[0] * cd[0]

    for i in range(1,4):
        ti = t + a[i]*h
        gijfj = 0
        bijfj = 0
        for j in range(0,i):
            gijfj += g[i][j] * fi[j]
            bijfj += b[i][j] * fi[j]
        xi = x + xd*a[i]*h + hsq*gijfj
        xdi = xd + h * bijfj
        fi[i] = f(ti, xi, xdi, *args)
        cifisum += fi[i] * c[i]
        cdifisum += fi[i] * cd[i]

    return ( x + h*xd + hsq*cifisum, xd + h*cdifisum )


def rkn4_step(f, t, x, xd, h, *args):
    """
    fourth-order Runge-Kutta-Nystroem step for function x(t)
    given by second order differential equation

    args: f - function determining second derivative
          t - value of independent variable t
          x - value of x(t)
          xdot - value of first derivative dx/dt
          h - time step
          args - parameters

    returns: iterated values for t + h
    """
    hsq = h*h
    a = np.array( [0, 0.25, 0.375, 12/13, 1.] )
    c = np.array( [253/2160, 0, 4352/12825, 2197/41040, -0.01] )
    cd = np.array( [25/216, 0, 1408/2565, 2197/4104, -0.2] )
    g = np.array( [[0, 0, 0, 0, 0],
                   [1/32, 0, 0, 0, 0],
                   [9/256,9/256, 0, 0, 0],
                   [27342/142805, -49266/142805, 82764/142805, 0, 0],
                   [5/18, -2/3, 8/9, 0, 0]] )
    b = np.array( [[0, 0, 0, 0, 0],
                   [0.25, 0, 0, 0, 0],
                   [3/32, 9/32, 0, 0, 0],
                   [1932/2197, -7200/2197, 7296/2197, 0, 0],
                   [439/216, -8.,3680/513, -845/4104, 0]] )

    fi = np.zeros(5)
    fi[0] = f(t, x, xd, *args)
    cifisum = fi[0] * c[0]
    cdifisum = fi[0] * cd[0]

    for i in range(1,5):
        ti = t + a[i]*h
        gijfj = 0
        bijfj = 0
        for j in range(0,i):
            gijfj += g[i][j] * fi[j]
            bijfj += b[i][j] * fi[j]
        xi = x + xd*a[i]*h + hsq*gijfj
        xdi = xd + h * bijfj
        fi[i] = f(ti, xi, xdi, *args)
        cifisum += fi[i] * c[i]
        cdifisum += fi[i] * cd[i]

    return ( x + h*xd + hsq*cifisum, xd + h*cdifisum )
