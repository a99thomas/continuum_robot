import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from matplotlib.animation import FuncAnimation
from tools.kinematics import lengths_to_q, robotindependentmapping, q_to_lengths
from tools.plotting import setupfigure, plot_tf, update_plot, draw_tdcr
        

current_path = Path(__file__).resolve().parent.parent
print(f"Current script path: {current_path}")

def clear_plot(plot_elements):
    """
    Clears all the plotted elements from the figure without resetting the axis limits.
    """
    for element in plot_elements.values():
        if isinstance(element, list):
            # For segment frames, clear each one
            for e in element:
                e.remove()
        else:
            element.remove()

def animate():

    lengths = np.array([[0.18], [0.26], [0.26]])
    radius = [0.0254, 0.0254]

    kappa, phi, ell = np.array([4,4]), np.array([0, 0]), np.array([0.392,0.392])
    # kappa, phi, ell = np.array([4]), np.array([0]), np.array([0.392])
    # kappa, phi, ell = np.array([4, 4]), np.array([0]), np.array([0.78])
    # kappa, phi, ell = lengths_to_q(type="threesegtdcr", lengths=lengths, radius = radius)
    # print("PHI", phi)

    g0 = robotindependentmapping(np.array(kappa), np.array(phi), np.array(ell), np.array([10]))
    fig, ax = setupfigure(g0=g0)
    seg_end = np.array([11,22])  # Example segment indices
    
    clearance = 0.03
    curvelength = np.sum(np.linalg.norm(g0[1:, 12:15] - g0[:-1, 12:15], axis=1))
    
    def frame_update(frame):
        # Call robotindependentmapping to update `g` based on the current frame
        ax.clear()
        max_val_x = np.max(np.abs(g0[:, 12])) + clearance
        max_val_y = np.max(np.abs(g0[:, 13])) + clearance
        ax.set_xlim(-max_val_x, max_val_x)
        ax.set_ylim(-max_val_y, max_val_y)
        ax.set_zlim(0, curvelength + clearance)

        # kappa, phi, ell = np.array([4]), np.array([0]), np.array([0.78])
        kappa, phi, ell = np.array([4,4]), np.array([np.pi*np.sin(np.pi*frame/100)*0.997, 0.997*np.pi*np.cos(np.pi*frame/100)]), np.array([0.393,0.393])
        # lengths = q_to_lengths(kappa, phi, ell, radius).T
        # print("LENGTHS",lengths)
        
        # kappa, phi, ell = lengths_to_q(lengths= lengths, radius = radius)

        print(kappa,phi,ell)

        
        g = robotindependentmapping(np.array(kappa), np.array(phi), np.array(ell), np.array([10]))
        # print("G",g[-4:-1])
        
        # Plot the updated `g`
        plot_elements = plot_tf(ax, g, seg_end, tipframe=True, segframe=False, baseframe=True, projections=True, baseplate=True)
        update_plot(plot_elements, g)
        plot_elements2 = draw_tdcr(ax,
            g, 
            seg_end, 
            r_disk=2.5*1e-2, 
            r_height=1.5*1e-3, 
            tipframe=True, 
            segframe=False, 
            baseframe=True, 
            projections=False, 
            baseplate=False
        )

        # Redraw the plot for the new frame
        plt.draw()
        # clear_plot(plot_elements)

    ani = FuncAnimation(fig, frame_update, frames=200, interval=10)
    plt.show()

animate()
# length_to_q(lengths = np.array([[5,6],[4,6],[5,6]]))
