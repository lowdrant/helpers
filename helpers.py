"""
Boilerplate Python. Mostly plotting and some numerics.
"""
import matplotlib.pyplot as plt
from numpy import array, asarray, cos, isscalar, mean, sin, sqrt, sum, zeros

__all__ = ['r2d', 'newplot', 'ipychk', 'SE2', 'rms']

# ============================================================================
# Rigid Body Funcs


def r2d(th):
    """Vectorized 2d rotation matrix calculation
        INPUTS:
            th -- Nx... angle, can be ndarray
        OUTPUTS:
            R -- 2x2xN... rotation matrices
    """
    return array([[cos(t), -sin(t)], [sin(t), cos(t)]])


def SE2(x, y, th):
    """Generate SE(2) RBT from R^3 parameterization
        INPUTS:
            x -- N... --  x coord of transform
            y -- N... -- y coord of transform
            th -- angle of transform
        OUTPUTS:
            g -- 3x3xN... -- SE(2) RBT
    """
    N = 1
    if not isscalar(x):
        N = len(x)
    out = zeros((3, 3, N), dtype=float).squeeze()
    out[:2, :2] = r2d(th)
    out[0, 2] = x
    out[1, 2] = y
    out[2, 2] = 1
    return out

# ============================================================================
# Plotting


def newplot(num=None):
    """Generate fresh figure with single axis with grid.
        INPUTS:
            num -- (optional) num argument for `plt.figure`
        OUTPUTS:
            ax -- axis for plotting
    """
    if num is None:
        f = plt.figure()
    else:
        f = plt.figure(num)
    f.clf()
    ax = f.add_subplot(111)
    ax.grid()
    ax.set_title(str(num))
    return ax


def ipychk():
    """If ipython session, turn on interactive plots; else `plt.show()`."""
    try:
        get_ipython()
        plt.ion()
    except NameError:
        plt.ioff()
        plt.show()

# ============================================================================
# Numerics


def rms(x, axis=None):
    """Compute RMS of x
        INPUTS:
            x -- Nx... array of data
            axis -- (optional) axis along which to compute mean
        OUTPUTS:
            rms -- sqrt((x**2).mean(axis))
    """
    x = asarray(x)
    return sqrt((x**2).mean(axis=axis))
