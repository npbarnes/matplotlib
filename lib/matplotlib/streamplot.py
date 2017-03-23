"""
Streamline plotting for 2D vector fields.

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six.moves import xrange

import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
import matplotlib.patches as patches
import bisect


__all__ = ['streamplot']


def streamplot(axes, x, y, u, v, density=1, linewidth=None, color=None,
               cmap=None, norm=None, arrowsize=1, arrowstyle='-|>',
               minlength=0.1, transform=None, zorder=None, start_points=None,
               maxlength=4.0, integration_direction='both'):
    """Draws streamlines of a vector flow.

    *x*, *y* : 1d arrays
        an *evenly spaced* grid.
    *u*, *v* : 2d arrays
        x and y-velocities. Number of rows should match length of y, and
        the number of columns should match x.
    *density* : float or 2-tuple
        Controls the closeness of streamlines. When `density = 1`, the domain
        is divided into a 30x30 grid---*density* linearly scales this grid.
        Each cell in the grid can have, at most, one traversing streamline.
        For different densities in each direction, use [density_x, density_y].
    *linewidth* : numeric or 2d array
        vary linewidth when given a 2d array with the same shape as velocities.
    *color* : matplotlib color code, or 2d array
        Streamline color. When given an array with the same shape as
        velocities, *color* values are converted to colors using *cmap*.
    *cmap* : :class:`~matplotlib.colors.Colormap`
        Colormap used to plot streamlines and arrows. Only necessary when using
        an array input for *color*.
    *norm* : :class:`~matplotlib.colors.Normalize`
        Normalize object used to scale luminance data to 0, 1. If None, stretch
        (min, max) to (0, 1). Only necessary when *color* is an array.
    *arrowsize* : float
        Factor scale arrow size.
    *arrowstyle* : str
        Arrow style specification.
        See :class:`~matplotlib.patches.FancyArrowPatch`.
    *minlength* : float
        Minimum length of streamline in axes coordinates.
    *start_points*: Nx2 array
        Coordinates of starting points for the streamlines.
        In data coordinates, the same as the ``x`` and ``y`` arrays.
    *zorder* : int
        any number
    *maxlength* : float
        Maximum length of streamline in axes coordinates.
    *integration_direction* : ['forward', 'backward', 'both']
        Integrate the streamline in forward, backward or both directions.

    Returns:

        *stream_container* : StreamplotSet
            Container object with attributes

                - lines: `matplotlib.collections.LineCollection` of streamlines

                - arrows: collection of `matplotlib.patches.FancyArrowPatch`
                  objects representing arrows half-way along stream
                  lines.

            This container will probably change in the future to allow changes
            to the colormap, alpha, etc. for both lines and arrows, but these
            changes should be backward compatible.

    """
    grid = Grid(x, y)
    mask = StreamMask(density)
    dmap = DomainMap(grid, mask)

    if zorder is None:
        zorder = mlines.Line2D.zorder

    # default to data coordinates
    if transform is None:
        transform = axes.transData

    if color is None:
        color = axes._get_lines.get_next_color()

    if linewidth is None:
        linewidth = matplotlib.rcParams['lines.linewidth']

    line_kw = {}
    arrow_kw = dict(arrowstyle=arrowstyle, mutation_scale=10 * arrowsize)

    if integration_direction not in ['both', 'forward', 'backward']:
        errstr = ("Integration direction '%s' not recognised. "
                  "Expected 'both', 'forward' or 'backward'." %
                  integration_direction)
        raise ValueError(errstr)

    if integration_direction == 'both':
        maxlength /= 2.

    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        if color.shape != grid.shape:
            msg = "If 'color' is given, must have the shape of 'Grid(x,y)'"
            raise ValueError(msg)
        line_colors = []
        color = np.ma.masked_invalid(color)
    else:
        line_kw['color'] = color
        arrow_kw['color'] = color

    if isinstance(linewidth, np.ndarray):
        if linewidth.shape != grid.shape:
            msg = "If 'linewidth' is given, must have the shape of 'Grid(x,y)'"
            raise ValueError(msg)
        line_kw['linewidth'] = []
    else:
        line_kw['linewidth'] = linewidth
        arrow_kw['linewidth'] = linewidth

    line_kw['zorder'] = zorder
    arrow_kw['zorder'] = zorder

    ## Sanity checks.
    if (u.shape != grid.shape) or (v.shape != grid.shape):
        msg = "'u' and 'v' must be of shape 'Grid(x,y)'"
        raise ValueError(msg)

    u = np.ma.masked_invalid(u)
    v = np.ma.masked_invalid(v)

    integrate = get_integrator(u, v, dmap, minlength, maxlength,
                               integration_direction)

    trajectories = []
    if start_points is None:
        start_points = np.asanyarray(list(_gen_starting_points(dmap)))
    else:
        start_points = np.asanyarray(start_points, dtype=float).copy()

        # Check if start_points are outside the data boundaries
        for xs, ys in sp2:
            if not (grid.x_origin <= xs <= grid.x_origin + grid.width
                    and grid.y_origin <= ys <= grid.y_origin + grid.height):
                raise ValueError("Starting point ({}, {}) outside of data "
                                 "boundaries".format(xs, ys))

    for x0, y0 in start_points:
        t = integrate(x0, y0)
        if t is not None:
            trajectories.append(t)

    if use_multicolor_lines:
        if norm is None:
            norm = mcolors.Normalize(color.min(), color.max())
        if cmap is None:
            cmap = cm.get_cmap(matplotlib.rcParams['image.cmap'])
        else:
            cmap = cm.get_cmap(cmap)

    streamlines = []
    arrows = []
    for t in trajectories:
        tx = np.array(t[0])
        ty = np.array(t[1])

        points = np.transpose([tx, ty]).reshape(-1, 1, 2)
        streamlines.extend(np.hstack([points[:-1], points[1:]]))

        # Add arrows half way along each trajectory.
        s = np.cumsum(np.sqrt(np.diff(tx) ** 2 + np.diff(ty) ** 2))
        n = np.searchsorted(s, s[-1] / 2.)
        arrow_tail = (tx[n], ty[n])
        arrow_head = (np.mean(tx[n:n + 2]), np.mean(ty[n:n + 2]))

        if isinstance(linewidth, np.ndarray):
            raise NotImplemented
            line_widths = interpgrid(linewidth, tgx, tgy)[:-1]
            line_kw['linewidth'].extend(line_widths)
            arrow_kw['linewidth'] = line_widths[n]

        if use_multicolor_lines:
            raise NotImplemented
            color_values = interpgrid(color, tgx, tgy)[:-1]
            line_colors.append(color_values)
            arrow_kw['color'] = cmap(norm(color_values[n]))

        p = patches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform, **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)

    lc = mcollections.LineCollection(
        streamlines, transform=transform, **line_kw)
    #lc.sticky_edges.x[:] = [grid.x_origin, grid.x_origin + grid.width]
    #lc.sticky_edges.y[:] = [grid.y_origin, grid.y_origin + grid.height]
    if use_multicolor_lines:
        lc.set_array(np.ma.hstack(line_colors))
        lc.set_cmap(cmap)
        lc.set_norm(norm)
    axes.add_collection(lc)
    axes.update_datalim(((x.min(),y.min()), (x.max(), y.max())))
    axes.autoscale_view()

    ac = matplotlib.collections.PatchCollection(arrows)
    stream_container = StreamplotSet(lc, ac)
    return stream_container


class StreamplotSet(object):

    def __init__(self, lines, arrows, **kwargs):
        self.lines = lines
        self.arrows = arrows


# Coordinate definitions
# ========================

class DomainMap(object):
    """
    This class has methods for adding trajectories to the StreamMask.
    Before adding a trajectory, run `start_trajectory` to keep track of regions
    crossed by a given trajectory. Later, if you decide the trajectory is bad
    (e.g., if the trajectory is very short) just call `undo_trajectory`.

    Coordinate definitions:

    * data-coordinates are specified by the input x-y coordinates. The same
      they are used elsewhere in matplotlib.
    * mask-coordinates are integers that go from 0 to N and 0 to M for an 
      N x M mask, where N and M are user-specified to control the density of
      streamlines.
    """

    def __init__(self, grid, mask):
        self.grid = grid
        self.mask = mask
        # Constants for conversion between data- and mask-coordinates
        self.x_data2mask = float(mask.nx - 1) / grid.width
        self.y_data2mask = float(mask.ny - 1) / grid.height

        self.x_mask2data = 1. / self.x_data2mask
        self.y_mask2data = 1. / self.y_data2mask


    # Note that data2mask and mask2data are NOT inverses of each other unless the
    # specific data coordinates in question exactly fall on a space in the mask.
    def data2mask(self, xi, yi):
        """Find which mask cell the given data-coords fall in."""
        return int((xi - self.grid.x_origin) * self.x_data2mask + 0.5), \
            int((yi - self.grid.y_origin)  * self.y_data2mask + 0.5)

    def mask2data(self, xm, ym):
        """Get the data coordinates of the center of the given mask cell."""
        return self.grid.x_origin + xm * self.x_mask2data, \
            self.grid.y_origin + ym * self.y_mask2data

    def start_trajectory(self, x, y):
        xm, ym = self.data2mask(x, y)
        self.mask._start_trajectory(xm, ym)

    def reset_start_point(self, x, y):
        xm, ym = self.data2mask(x, y)
        self.mask._current_xy = (xm, ym)

    def update_trajectory(self, x, y):
        if not self.grid.within_grid(x, y):
            raise InvalidIndexError
        xm, ym = self.data2mask(x, y)
        self.mask._update_trajectory(xm, ym)

    def undo_trajectory(self):
        self.mask._undo_trajectory()


class Grid(object):
    """Grid of data."""
    def __init__(self, x, y):

        if x.ndim == 1:
            pass
        elif x.ndim == 2:
            x_row = x[0, :]
            if not np.allclose(x_row, x):
                raise ValueError("The rows of 'x' must be equal")
            x = x_row
        else:
            raise ValueError("'x' can have at maximum 2 dimensions")

        if y.ndim == 1:
            pass
        elif y.ndim == 2:
            y_col = y[:, 0]
            if not np.allclose(y_col, y.T):
                raise ValueError("The columns of 'y' must be equal")
            y = y_col
        else:
            raise ValueError("'y' can have at maximum 2 dimensions")

        self.nx = len(x)
        self.ny = len(y)

        self.x = x
        self.y = y

        self.dxs = x[1:] - x[:-1]
        self.dys = y[1:] - y[:-1]

        self.x_origin = x[0]
        self.y_origin = y[0]

        self.width = x[-1] - x[0]
        self.height = y[-1] - y[0]

    @property
    def shape(self):
        return self.ny, self.nx

    def within_grid(self, x, y):
        """Return True if point is inside the grid boundaries."""
        return self.x_origin <= x < self.x_origin + self.width and \
                self.y_origin <= y < self.y_origin + self.height


class StreamMask(object):
    """Mask to keep track of discrete regions crossed by streamlines.

    The resolution of this grid determines the approximate spacing between
    trajectories. Streamlines are only allowed to pass through zeroed cells:
    When a streamline enters a cell, that cell is set to 1, and no new
    streamlines are allowed to enter.
    """

    def __init__(self, density):
        if np.isscalar(density):
            if density <= 0:
                raise ValueError("If a scalar, 'density' must be positive")
            self.nx = self.ny = int(30 * density)
        else:
            if len(density) != 2:
                raise ValueError("'density' can have at maximum 2 dimensions")
            self.nx = int(30 * density[0])
            self.ny = int(30 * density[1])
        self._mask = np.zeros((self.ny, self.nx))
        self.shape = self._mask.shape

        self._current_xy = None

    def __getitem__(self, *args):
        return self._mask.__getitem__(*args)

    def _start_trajectory(self, xm, ym):
        """Start recording streamline trajectory"""
        self._traj = []
        self._update_trajectory(xm, ym)

    def _undo_trajectory(self):
        """Remove current trajectory from mask"""
        for t in self._traj:
            self._mask.__setitem__(t, 0)

    def _update_trajectory(self, xm, ym):
        """Update current trajectory position in mask.

        If the new position has already been filled, raise `InvalidIndexError`.
        """
        if self._current_xy != (xm, ym):
            if self[ym, xm] == 0:
                self._traj.append((ym, xm))
                self._mask[ym, xm] = 1
                self._current_xy = (xm, ym)
            else:
                raise InvalidIndexError


class InvalidIndexError(Exception):
    pass


class TerminateTrajectory(Exception):
    pass


# Integrator definitions
#========================
def index_frac(x, x0):
    """return the index of x0 and the fraction of the way through the cell it is"""
    index = bisect.bisect(x, x0) - 1
    if index < 0: raise IndexError
    if index > len(x)-2: raise IndexError
    frac = (x0 - x[index]) / (x[index+1] - x[index])
    return index, frac

def get_integrator(u, v, dmap, minlength, maxlength, integration_direction):
    # speed (path length) will be in axes-coordinates
    u_ax = u / dmap.grid.width
    v_ax = v / dmap.grid.height
    speed = np.ma.sqrt(u_ax ** 2 + v_ax ** 2)

    def forward_time(x, y):
        """x and y are scalar positions"""

        ds_dt = interpgrid(dmap.grid, speed, x, y)
        if ds_dt == 0:
            raise TerminateTrajectory()
        dt_ds = 1. / ds_dt
        ui = interpgrid(dmap.grid, u, x, y)
        vi = interpgrid(dmap.grid, v, x, y)
        return ui * dt_ds, vi * dt_ds

    def backward_time(x, y):
        """x and y are scalar positions"""
        dx, dy = forward_time(x, y)
        return -dx, -dy

    def integrate(x0, y0):
        """Return grid-coordinates of trajectory based on starting point.

        Integrate both forward and backward in time from starting point in
        grid coordinates.

        Integration is terminated when a trajectory reaches a domain boundary
        or when it crosses into an already occupied cell in the StreamMask. The
        resulting trajectory is None if it is shorter than `minlength`.
        """

        stotal, x_traj, y_traj = 0., [], []

        try:
            dmap.start_trajectory(x0, y0)
        except InvalidIndexError:
            return None
        if integration_direction in ['both', 'backward']:
            s, xt, yt = _integrate_rk12(x0, y0, dmap, backward_time, maxlength)
            stotal += s
            x_traj += xt[::-1]
            y_traj += yt[::-1]

        if integration_direction in ['both', 'forward']:
            dmap.reset_start_point(x0, y0)
            s, xt, yt = _integrate_rk12(x0, y0, dmap, forward_time, maxlength)
            if len(x_traj) > 0:
                xt = xt[1:]
                yt = yt[1:]
            stotal += s
            x_traj += xt
            y_traj += yt

        if stotal > minlength:
            return x_traj, y_traj
        else:  # reject short trajectories
            dmap.undo_trajectory()
            return None

    return integrate


def _integrate_rk12(x0, y0, dmap, f, maxlength):
    """2nd-order Runge-Kutta algorithm with adaptive step size.

    This method is also referred to as the improved Euler's method, or Heun's
    method. This method is favored over higher-order methods because:

    1. To get decent looking trajectories and to sample every mask cell
       on the trajectory we need a small timestep, so a lower order
       solver doesn't hurt us unless the data is *very* high resolution.
       In fact, for cases where the user inputs
       data smaller or of similar grid size to the mask grid, the higher
       order corrections are negligible because of the very fast linear
       interpolation used in `interpgrid`.

    2. For high resolution input data (i.e. beyond the mask
       resolution), we must reduce the timestep. Therefore, an adaptive
       timestep is more suited to the problem as this would be very hard
       to judge automatically otherwise.

    This integrator is about 1.5 - 2x as fast as both the RK4 and RK45
    solvers in most setups on my machine. I would recommend removing the
    other two to keep things simple.
    """
    # This error is below that needed to match the RK4 integrator. It
    # is set for visual reasons -- too low and corners start
    # appearing ugly and jagged. Can be tuned.
    maxerror = 0.003

    # This limit is important (for all integrators) to avoid the
    # trajectory skipping some mask cells. We could relax this
    # condition if we use the code which is commented out below to
    # increment the location gradually. However, due to the efficient
    # nature of the interpolation, this doesn't boost speed by much
    # for quite a bit of complexity.
    maxds = min(1. / dmap.mask.nx, 1. / dmap.mask.ny, 0.1)

    ds = maxds
    stotal = 0
    xi = x0
    yi = y0
    xf_traj = []
    yf_traj = []

    while dmap.grid.within_grid(xi, yi):
        xf_traj.append(xi)
        yf_traj.append(yi)
        try:
            k1x, k1y = f(xi, yi)
            k2x, k2y = f(xi + ds * k1x,
                         yi + ds * k1y)
        except IndexError:
            # Out of the domain on one of the intermediate integration steps.
            # Take an Euler step to the boundary to improve neatness.
            ds, xf_traj, yf_traj = _euler_step(xf_traj, yf_traj, dmap, f)
            stotal += ds
            break
        except TerminateTrajectory:
            break

        dx1 = ds * k1x
        dy1 = ds * k1y
        dx2 = ds * 0.5 * (k1x + k2x)
        dy2 = ds * 0.5 * (k1y + k2y)

        nx, ny = dmap.grid.shape
        # Error is normalized to the axes coordinates
        error = np.sqrt(((dx2 - dx1) / nx) ** 2 + ((dy2 - dy1) / ny) ** 2)

        # Only save step if within error tolerance
        if error < maxerror:
            xi += dx2
            yi += dy2
            try:
                dmap.update_trajectory(xi, yi)
            except InvalidIndexError:
                break
            if (stotal + ds) > maxlength:
                break
            stotal += ds

        # recalculate stepsize based on step error
        if error == 0:
            ds = maxds
        else:
            ds = min(maxds, 0.85 * ds * (maxerror / error) ** 0.5)

    return stotal, xf_traj, yf_traj


def _euler_step(xf_traj, yf_traj, dmap, f):
    """Simple Euler integration step that extends streamline to boundary."""
    ny, nx = dmap.grid.shape
    xi = xf_traj[-1]
    yi = yf_traj[-1]
    cx, cy = f(xi, yi)
    if cx == 0:
        dsx = np.inf
    elif cx < 0:
        dsx = (xi - dmap.grid.x_origin) / -cx
    else:
        dsx = (dmap.grid.x[-1] - xi) / cx
    if cy == 0:
        dsy = np.inf
    elif cy < 0:
        dsy = (yi - dmap.grid.y_origin) / -cy
    else:
        dsy = (dmap.grid.y[-1] - yi) / cy
    ds = min(dsx, dsy)
    xf_traj.append(xi + cx * ds)
    yf_traj.append(yi + cy * ds)
    return ds, xf_traj, yf_traj

# Utility functions
# ========================

def interpgrid(grid, v, x, y):
    """Fast 2D, linear interpolation of the values v on the grid to the point x,y"""

    i_x, dx = index_frac(grid.x, x)
    i_y, dy = index_frac(grid.y, y)

    # get corners
    v00 = v[i_y, i_x]
    v01 = v[i_y, i_x + 1]
    v10 = v[i_y + 1, i_x]
    v11 = v[i_y + 1, i_x + 1]

    # do interpolation
    v0 = v00 * (1 - dx) + v01 * dx
    v1 = v10 * (1 - dx) + v11 * dx
    vi = v0 * (1 - dy) + v1 * dy

    if not isinstance(i_x, np.ndarray):
        if np.ma.is_masked(vi):
            raise TerminateTrajectory

    return vi

def _gen_starting_points(dmap):
    """Yield starting points for streamlines.

    Trying points on the boundary first gives higher quality streamlines.
    This algorithm starts with a point on the mask corner and spirals inward.
    This algorithm is inefficient, but fast compared to rest of streamplot.
    """
    ny, nx = dmap.mask.shape
    xfirst = 0
    yfirst = 1
    xlast = nx - 1
    ylast = ny - 1
    x, y = 0, 0
    i = 0
    direction = 'right'
    for i in xrange(nx * ny):

        if dmap.mask[x,y] == 0:
            yield dmap.mask2data(x,y)

        if direction == 'right':
            x += 1
            if x >= xlast:
                xlast -= 1
                direction = 'up'
        elif direction == 'up':
            y += 1
            if y >= ylast:
                ylast -= 1
                direction = 'left'
        elif direction == 'left':
            x -= 1
            if x <= xfirst:
                xfirst += 1
                direction = 'down'
        elif direction == 'down':
            y -= 1
            if y <= yfirst:
                yfirst += 1
                direction = 'right'
