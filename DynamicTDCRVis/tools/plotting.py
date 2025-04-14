import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

def setupfigure(g0, clearance=0.03):
    """
    Sets up the matplotlib figure of the model along with the visuals.
    """
    fig = plt.figure()
    fig.set_size_inches(1280 / fig.dpi, 1024 / fig.dpi)
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    ax.set_box_aspect([1, 1, 1])

    curvelength = np.sum(np.linalg.norm(g0[1:, 12:15] - g0[:-1, 12:15], axis=1))
    max_val_x = np.max(np.abs(g0[:, 12])) + clearance
    max_val_y = np.max(np.abs(g0[:, 13])) + clearance
    ax.set_xlim(0, max_val_x)
    ax.set_ylim(-1.0, max_val_y)
    ax.set_zlim(0, curvelength + clearance)
    # Set axis limits explicitly to ±0.5 in all directions
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([0, 2])  # Ensure positive z-axis direction
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.grid(True, alpha=0.3)
    ax.view_init(azim=45, elev=30)
    return fig, ax

def plot_tf(ax, g, seg_end, tipframe, segframe, baseframe, projections, baseplate):
    """
    Plots the TDCR model and returns plot elements for updates.
    """
    plot_elements = {}

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_ylim(ax.get_xlim())
    # ax.set_ylim(1.0)
    # Projections
    if projections:
        # print("XLIM:", ax.get_xlim)
        plot_elements['proj_yz'] = ax.plot(np.ones(g.shape[0]) * ax.get_xlim()[0], g[:, 13], g[:, 14], linewidth=2, color='r')[0]
        plot_elements['proj_xz'] = ax.plot(g[:, 12], np.ones(g.shape[0]) * ax.get_ylim()[0], g[:, 14], linewidth=2, color='g')[0]
        plot_elements['proj_xy'] = ax.plot(g[:, 12], g[:, 13], np.zeros(g.shape[0]), linewidth=2, color='b')[0]

    # Base Plate
    if baseplate:
        color = [0.9, 0.9, 0.9]
        squaresize = 0.03
        thickness = 0.001
        x = np.array([-1, 1, 1, -1]) * squaresize
        y = np.array([-1, -1, 1, 1]) * squaresize
        z = np.array([-1, -1, -1, -1]) * thickness
        verts = [list(zip(x, y, z))]
        plot_elements['baseplate'] = ax.add_collection3d(Poly3DCollection(verts, color=color, rasterized=True, zorder=-1), zdir='z')

    # Coordinate Frames
    if tipframe and not segframe:
        plot_elements['tip_x'] = ax.quiver(g[-1, 12], g[-1, 13], g[-1, 14], g[-1, 0], g[-1, 1], g[-1, 2], length=0.01, linewidth=3, color='r')
        plot_elements['tip_y'] = ax.quiver(g[-1, 12], g[-1, 13], g[-1, 14], g[-1, 4], g[-1, 5], g[-1, 6], length=0.01, linewidth=3, color='g')
        plot_elements['tip_z'] = ax.quiver(g[-1, 12], g[-1, 13], g[-1, 14], g[-1, 8], g[-1, 9], g[-1, 10], length=0.01, linewidth=3, color='b')

    if segframe:
        plot_elements['seg_frames'] = []
        for i in range(seg_end.size):
            seg_end_idx = seg_end[i] - 1
            plot_elements['seg_frames'].append(
                ax.quiver(g[seg_end_idx, 12], g[seg_end_idx, 13], g[seg_end_idx, 14], 
                          g[seg_end_idx, 0], g[seg_end_idx, 1], g[seg_end_idx, 2], 
                          length=0.01, linewidth=3, color='r')
            )

    # Base Frame
    if baseframe:
        plot_elements['base_x'] = ax.quiver(0, 0, 0, 1, 0, 0, length=0.01, linewidth=3, color='r')
        plot_elements['base_y'] = ax.quiver(0, 0, 0, 0, 1, 0, length=0.01, linewidth=3, color='g')
        plot_elements['base_z'] = ax.quiver(0, 0, 0, 0, 0, 1, length=0.01, linewidth=3, color='b')

    return plot_elements

def update_plot(plot_elements, g):
    """
    Updates the plot elements based on new data.
    """
    # Update projections
    plot_elements['proj_yz'].set_data(np.ones(g.shape[0]) * plot_elements['proj_yz'].axes.get_xlim()[0], g[:, 13])
    plot_elements['proj_yz'].set_3d_properties(g[:, 14])

    plot_elements['proj_xz'].set_data(g[:, 12], np.ones(g.shape[0]) * plot_elements['proj_xz'].axes.get_ylim()[0])
    plot_elements['proj_xz'].set_3d_properties(g[:, 14])

    plot_elements['proj_xy'].set_data(g[:, 12], g[:, 13])
    plot_elements['proj_xy'].set_3d_properties(np.zeros(g.shape[0]))

    # Example update for other elements (e.g., tip frame, segment frames)
    # This can be expanded as needed


def nullspace(A: np.ndarray[float], atol: float=1e-13, rtol: float=0):
    """
    Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    # Source: https://scipy-cookbook.readthedocs.io/items/RankNullspace.html

    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns



def draw_tdcr(ax, g: np.ndarray[float], seg_end: np.ndarray[int], r_disk: float=0.0127, r_height: float=1.5*1e-3, 
              tipframe: bool=True, segframe: bool=False, baseframe: bool=False, projections: bool=False, 
              baseplate: bool=True):
    '''
    DRAW_TDCR Creates a figure of a tendon-driven continuum robot (tdcr)

    Takes a matrix with nx16 entries, where n is the number
    of points on the backbone curve. For each point on the curve, the 4x4
    transformation matrix is stored columnwise (16 entries). The x- and
    y-axis span the material orientation and the z-axis is tangent to the
    curve.

    Parameters
    ----------
    g: ndarray
        Backbone curve with n 4x4 transformation matrices reshaped into 1x16 vector (columnwise)
    seg_end: ndarray
        Indices of g where tdcr segments terminate
    r_disk: double
        Radius of spacer disks
    r_height: double
        height of spacer disks
    tipframe: bool, default=True
        Shows tip frame
    segframe: bool, default=False
        Shows segment end frames
    baseframe: bool, default=False
        Shows robot base frame
    projections: bool, default=False
        Shows projections of backbone curve onto coordinate axes
    baseplate: bool, default=True
        Shows robot base plate

    Outputs
    -------
    plot_elements: dict
        Dictionary containing references to the plot elements (e.g., lines, surfaces, etc.)
    '''
    # Argument validation
    if g.shape[0] < len(seg_end) or max(seg_end) > g.shape[0]:
        raise ValueError("Dimension mismatch")

    #numseg = seg_end.size
    numseg = 3

    if numseg == 1:
        col = np.array([0.8])
    else:
        col = np.linspace(0.2, 0.8, numseg)

    plot_elements = {}

    # Backbone
    # start = 0
    # plot_elements['backbone'] = []
    # for i in range(numseg):
    #     line, = ax.plot(g[start:seg_end[i], 12], g[start:seg_end[i], 13], g[start:seg_end[i], 14], 
    #                     linewidth=5, color=col[i]*np.ones(3))
    #     plot_elements['backbone'].append(line)
    #     start = seg_end[i]

    # Tendons
    plot_elements['tendon1'] = ax.plot([], [], [], color='k')[0]
    plot_elements['tendon2'] = ax.plot([], [], [], color='k')[0]
    plot_elements['tendon3'] = ax.plot([], [], [], color='k')[0]

    tendon1 = np.zeros((seg_end[numseg - 1], 3))
    tendon2 = np.zeros((seg_end[numseg - 1], 3))
    tendon3 = np.zeros((seg_end[numseg - 1], 3))

    # Tendon locations on disk
    r1 = np.array([0, r_disk, 0])
    r2 = np.array([np.cos(30*np.pi/180)*r_disk, -np.sin(30*np.pi/180)*r_disk, 0])
    r3 = np.array([-np.cos(30*np.pi/180)*r_disk, -np.sin(30*np.pi/180)*r_disk, 0])

    for i in range(seg_end[numseg - 1]):
        RotMat = np.array([g[i, 0:3], g[i, 4:7], g[i, 8:11]]).T
        tendon1[i, 0:3] = RotMat@r1 + g[i, 12:15]
        tendon2[i, 0:3] = RotMat@r2 + g[i, 12:15]
        tendon3[i, 0:3] = RotMat@r3 + g[i, 12:15]

    plot_elements['tendon1'].set_data(tendon1[:, 0], tendon1[:, 1])
    plot_elements['tendon1'].set_3d_properties(tendon1[:, 2])

    plot_elements['tendon2'].set_data(tendon2[:, 0], tendon2[:, 1])
    plot_elements['tendon2'].set_3d_properties(tendon2[:, 2])

    plot_elements['tendon3'].set_data(tendon3[:, 0], tendon3[:, 1])
    plot_elements['tendon3'].set_3d_properties(tendon3[:, 2])

    # Draw spheres to represent tendon location at end disks
    radius = 0.75e-3

    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]  # Get sphere coordinates
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)

    plot_elements['spheres'] = []
    for i in range(numseg):
        sphere1 = ax.plot_surface(x*radius+tendon1[seg_end[i] - 1, 0], y*radius+tendon1[seg_end[i] - 1, 1], 
                                  z*radius+tendon1[seg_end[i] - 1, 2], color='k')
        sphere2 = ax.plot_surface(x*radius+tendon2[seg_end[i] - 1, 0], y*radius+tendon2[seg_end[i] - 1, 1], 
                                  z*radius+tendon2[seg_end[i] - 1, 2], color='k')
        sphere3 = ax.plot_surface(x*radius+tendon3[seg_end[i] - 1, 0], y*radius+tendon3[seg_end[i] - 1, 1], 
                                  z*radius+tendon3[seg_end[i] - 1, 2], color='k')
        plot_elements['spheres'].append(sphere1)
        plot_elements['spheres'].append(sphere2)
        plot_elements['spheres'].append(sphere3)

    # Spacer disks
    plot_elements['disks'] = []
    seg_idx = 0
    theta = np.arange(0, 2 * np.pi, 0.05)
    for i in range(g.shape[0]):
        if seg_end[seg_idx] < i:
            seg_idx += 1

        color = col[seg_idx]*np.ones(3)

        RotMat = np.array([g[i, 0:3], g[i, 4:7], g[i, 8:11]]).T
        normal = RotMat[:3, 2].T
        v = nullspace(normal)
        v_theta = v[:, 0].reshape((-1, 1)) * np.cos(theta) + v[:, 1].reshape((-1, 1)) * np.sin(theta)

        # Draw the lower circular surface of the disk
        pos = g[i, 12:15].T - RotMat @ np.array([0, 0, r_height/2])
        lowercirc = np.tile(pos.reshape((-1, 1)), theta.size) + r_disk * v_theta
        x, y, z = lowercirc[0, :], lowercirc[1, :], lowercirc[2, :]

        verts = [list(zip(x, y, z))]
        disk = ax.add_collection3d(Poly3DCollection(verts, color=color, edgecolor='k', rasterized=True, zorder=10), zdir='z')
        plot_elements['disks'].append(disk)

        # Draw the upper circular surface of the disk
        pos = g[i, 12:15].T + RotMat @ np.array([0, 0, r_height/2])
        uppercirc = np.tile(pos.reshape((-1, 1)), theta.size) + r_disk * v_theta
        x, y, z = uppercirc[0, :], uppercirc[1, :], uppercirc[2, :]

        verts = [list(zip(x, y, z))]
        disk = ax.add_collection3d(Poly3DCollection(verts, color=color, edgecolor='k', rasterized=True, zorder=10), zdir='z')
        plot_elements['disks'].append(disk)

        # Draw the in-between surface of the disk
        x = np.vstack((lowercirc[0, :], uppercirc[0, :]))
        y = np.vstack((lowercirc[1, :], uppercirc[1, :]))
        z = np.vstack((lowercirc[2, :], uppercirc[2, :]))

        disk_surface = ax.plot_surface(x, y, z, color=color, shade=False, zorder=10)
        plot_elements['disks'].append(disk_surface)

    return plot_elements
