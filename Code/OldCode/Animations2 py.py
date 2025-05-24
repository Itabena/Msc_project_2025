import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
from scipy.integrate import odeint
from scipy.optimize import brentq
import random as rnd
import os
from PIL import Image
import glob
import shutil
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.signal import argrelextrema
from PIL import Image, ImageSequence
from fractions import Fraction
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import multiprocessing
from tqdm import tqdm
# from IPython.display import HTML


# Defining the equations of motion for the Schwarzschild metric
# def L(rp,E):
#     return ((((E**2)/(1-(2/rp)))-1)*(rp**2))**(1/2)

save_path=r"C:\Users\itama\Documents\.venv\Scripts\Research scripts\gifspnp"

def gr_r_dot(r, E, l):
    return ((1 - (2 / r)) / E) * np.sqrt((E**2) - (1 - (2 / r)) * (1 + ((l**2) / (r**2))))

def gr_v_eff(r, l):
    return -(2 / r) + ((l**2) / (r**2)) - 2 * ((l**2) / (r**3))

def gr_v_eff_derivative(r, l):
    return 6 * ((l**2) / (r**4)) - (2 / r**3) * (l**2) + (2 / r**2)

def gr_f_dot(r, E, l):
    return (1 - (2 / r)) * (l / (E * (r**2)))

# Solving the equations of motion using scipy's odeint

def equations(y, t, E, l):
    r, f = y
    drdt = -gr_r_dot(r, E, l)
    dfdot = gr_f_dot(r, E, l)
    return [drdt, dfdot]

def equationsr(y, t, E, l):
    r, f = y
    drdt = gr_r_dot(r, E, l)
    dfdot = gr_f_dot(r, E, l)
    return [drdt, dfdot]

def find_rp_ap(l,E):

    def equationrp(r, E, l):
        return (E**2 - 1) * (r**3) + 2 * (r**2) - (l**2) * r + 2 * (l**2)
    roots = np.roots([(E**2 - 1), 2, -(l**2), 2 * (l**2)])
    real_roots = [r.real for r in roots if np.isreal(r) and r > 2]
    real_roots.sort()
    if len(real_roots) == 1:
        return real_roots[0] ,0 , 1
    elif len(real_roots) == 2:
        return real_roots[-1] ,0 , 2
    elif len(real_roots) == 3:
        return real_roots[1], real_roots[2], 3
    else:
        return 0, 0, 0

def find_possible_values(l):
    # Find the local minimum of the effective potential above rp
    rp, ra, num_roots = find_rp_ap(l, 1)
    result = sp.optimize.minimize_scalar(
        lambda r: gr_v_eff(r, l),
        bounds=(rp, 10 * rp),
        method='bounded'
    )
    if result.success:
        v_min = result.fun
        r_min = result.x
        # print('r_min:', r_min, 'r_min:', v_min)
        return r_min, v_min
    else:
        raise ValueError("Failed to find the local minimum of the effective potential.")
def find_possible_values2(l):
    # Use scipy's optimize.brentq to find the local maximum of the effective potential
    rp, ra, num_roots = find_rp_ap(l, 1)
    def derivative(r):
        return gr_v_eff_derivative(r,l)

    # Find the root of the derivative within the bounds (2, rp)
    try:
        r_max = brentq(derivative, 2, rp)
        # print('r_max:', r_max)
        v_max = gr_v_eff(r_max, l)
        # print('v_max:', v_max)
        return r_max, v_max
    except ValueError:
        raise ValueError("Failed to find the local maximum of the effective potential.")


r0 = 100.0  # initial radius
rp = 8  # periapsis radius
f0 = 0  # initial angle
E = 1  # energy per unit mass (adjusted to ensure a valid trajectory)
y0 = [r0, f0]  # initial conditions vector

# Calculate the derivative of the effective potential

def calc_time_angle_gr(l, E):
    # Calculate the time of periapsis passage and angle at periapsis passage
    if l==np.sqrt(12):
        r_min=6
        v_min=np.sqrt(8/9)
        r_max=1000
        v_max=1000
    else:
        r_min,v_min = find_possible_values(l)
        r_max,v_max = find_possible_values2(l)
    # f0=rnd.uniform(0,np.pi/2)
    f0=np.pi/6
    rp,ra, num_roots = find_rp_ap(l,E)
    if rp>100:
        r0=rp*3
    else:
        r0=100
    rp=rp+10**-7
    print('num_roots:', num_roots)
    if E >= 1 and E < np.sqrt(1+v_max):
        tp = sp.integrate.quad(lambda r: -1/gr_r_dot(r, E, l), r0, rp)[0]  # time of periapsis passage
        print('tp:', tp)
        fp = f0 + sp.integrate.quad(lambda r: -gr_f_dot(r, E, l)/gr_r_dot(r, E, l), r0, rp)[0]  # angle at periapsis passage
        t_approach = np.linspace(0, tp, 50000)
        # Solve the equations of motion for the approaching side
        y0 = [r0, f0]
        sol_approach = odeint(equations, y0, t_approach, args=(E, l))

        # Extract r and f for the approaching side
        r_approach = sol_approach[:, 0]
        f_approach = sol_approach[:, 1]

        # Time points for the returning side
        t_return = np.linspace(tp, 2*tp, 50000)

        # Reverse the initial conditions for the returning side
        y0_return = [rp+10**-10, fp]

        # Solve the equations of motion for the returning side
        sol_return = odeint(equationsr, y0_return, t_return, args=(E, l))

        # Extract r and f for the returning side
        r_return = sol_return[:, 0]
        f_return = sol_return[:, 1]

        # Combine the results for the full orbit
        r = np.concatenate((r_approach, r_return))
        f = np.concatenate((f_approach, f_return))

        # Plotting the trajectory in polar coordinates
        x = r * np.cos(f)  # Convert polar to Cartesian x-coordinate
        y = r * np.sin(f)  # Convert polar to Cartesian y-coordinate
        t= np.concatenate((t_approach, t_return))
            
    elif E<1 and E>np.sqrt(1+v_min):
        rp=rp+0.0001
        tp = sp.integrate.quad(lambda r: -1/gr_r_dot(r, E, l), ra, rp)[0]  # time of periapsis passage
        ttmp= sp.integrate.quad(lambda r: -1/gr_r_dot(r, E, l), ra-0.5, rp)[0]  # time of periapsis passage
        fp = f0 + sp.integrate.quad(lambda r: -gr_f_dot(r, E, l)/gr_r_dot(r, E, l), ra, rp)[0]  # angle at periapsis passage
        num_periods = 2 # Number of periods to simulate
        t_approach = np.linspace(0, ttmp, 50000)
        y0 = [ra-0.5,f0]
        # Solve the equations of motion for the approaching side
        sol_approach = odeint(equations, y0, t_approach, args=(E, l))
        y0=[]
        # Extract r and f for the approaching side
        r_approach = sol_approach[:, 0]
        f_approach = sol_approach[:, 1]

        # Initialize arrays to store the full orbit
        r_full = r_approach
        f_full = f_approach
        t_full = t_approach
        
        print( fp , f_full[-1])
        # Loop through multiple periods
        for i in range(num_periods - 1):
            # Time points for the returning side
            t_return = np.linspace(tp+tp*2*i, tp+tp*(2*i+1), 50000)
            # Reverse the initial conditions for the returning side
            y0_return = [rp, f_full[-1]]
            # Solve the equations of motion for the returning side
            sol_return = odeint(equationsr, y0_return, t_return, args=(E, l))
            y0_return =[]
            # Extract r and f for the returning side
            r_return = sol_return[:, 0]
            f_return = sol_return[:, 1]
            
            # Append the results to the full orbit
            r_full = np.concatenate((r_full, r_return))
            f_full = np.concatenate((f_full, f_return))
            t_full = np.concatenate((t_full, t_return))

            # Update the initial conditions for the next period
            t_approach = np.linspace(tp+tp*(2*i+1), tp+tp*(2*i+2), 50000)
            y0 = [ra-10**-9, f_full[-1]]
            sol_approach = odeint(equations, y0, t_approach, args=(E, l))
            y0=[]
            r_approach = sol_approach[:, 0]
            f_approach = sol_approach[:, 1]
            r_full = np.concatenate((r_full, r_approach))
            f_full = np.concatenate((f_full, f_approach))
            t_full = np.concatenate((t_full, t_approach))
   
        
        # Plotting the trajectory in polar coordinates
        x = r_full * np.cos(f_full)  # Convert polar to Cartesian x-coordinate
        y = r_full * np.sin(f_full)  # Convert polar to Cartesian y-coordinate
        t=t_full
    elif E==np.sqrt(1+v_min):
        
        # For a circular orbit, r remains constant at rp, and the angular frequency is given by gr_f_dot(rp, E, rp)
        t = np.linspace(0, 3 * 2 * np.pi / gr_f_dot(r_min, E, l), 50000)  # Time array for three periods
        f = gr_f_dot(r_min, E, l) * t  # Angular position as a function of time

        # Convert polar coordinates to Cartesian coordinates
        x = r_min * np.cos(f)
        y = r_min * np.sin(f)
        t=t

    elif E>=np.sqrt(1+v_max):
        
        # Plot the effective potential and a horizontal line at E^2 - 1

        tp = sp.integrate.quad(lambda r: -1/gr_r_dot(r, E, l), r0, 2)[0]
        print('tp:', tp)
        fp = f0 + sp.integrate.quad(lambda r: -gr_f_dot(r, E, l)/gr_r_dot(r, E, l), r0, 2)[0]  # angle at periapsis passage
        # Time points for the approaching side
        t_approach = np.linspace(0, tp, 50000)

        # Solve the equations of motion for the approaching side
        y0 = [r0, f0]
        sol_approach = odeint(equations, y0, t_approach, args=(E, l))

        # Extract r and f for the approaching side
        r_approach = sol_approach[:, 0]
        f_approach = sol_approach[:, 1]

        # Plotting the trajectory in polar coordinates for the approaching side
        x = r_approach * np.cos(f_approach)  # Convert polar to Cartesian x-coordinate
        y = r_approach * np.sin(f_approach)  # Convert polar to Cartesian y-coordinate
        t= t_approach
    return x, y, t, f0,r0



def calc_time_angle_gr_ISCO():
        l = np.sqrt(12)
        E = np.sqrt(8/9)
        r_start = 2.000001
        r_end = 6-(10**-2)
        f0 = 0

        # Integrate from r=2 to r=6 (one side only)
        tp = -sp.integrate.quad(lambda r: -1 / gr_r_dot(r, E, l), r_start, r_end)[0]
        print('tp:', tp)
        # fp = f0 + sp.integrate.quad(lambda r: -gr_f_dot(r, E, l) / gr_r_dot(r, E, l), r_start, r_end)[0]

        t_approach_1 = np.linspace(0, tp*0.9, 500000)
        t_approach_2 = np.linspace(tp*0.9, tp, 50000)
        t_approach = np.concatenate((t_approach_1, t_approach_2))
        y0 = [r_start, f0]
        sol_approach = odeint(equationsr, y0, t_approach, args=(E, l))

        r_approach = sol_approach[:, 0]
        f_approach = sol_approach[:, 1]
        print('r_approach:', r_approach)
        print('f_approach:', f_approach)
        # Convert to Cartesian coordinates
        x = r_approach * np.cos(f_approach)
        y = r_approach * np.sin(f_approach)
        t = t_approach

        # Flip arrays so the animation starts at r=6 and ends at r=2
        x = x[::-1]
        y = y[::-1]
        t=(tp-t_approach[::-1])

        return x, y, t, f0, r_end








def show_animation(E, l, i):
    # r0 = 100.0  # initial radius
    if l==np.sqrt(12):
        r_min=6
        v_min=np.sqrt(8/9)
        r_max=1000
        v_max=1000
    else:
        r_max, v_max = find_possible_values2(l)
        r_min, v_min = find_possible_values(l)
    x, y, t ,f0, r0 = calc_time_angle_gr(l, E)

    # Clean x, y, and t from NaN or unreal values
    valid_indices = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    x = np.array(x[valid_indices])
    y = np.array(y[valid_indices])
    t = np.array(t[valid_indices])

    # Check for empty arrays
    if len(x) == 0 or len(y) == 0 or len(t) == 0:
        raise ValueError("x, y, or t is empty after cleaning. Ensure valid input data.")

    r = np.sqrt(x**2 + y**2)  # Calculate the radius from x and y coordinates
    if not np.all(np.isfinite(r)):
        raise ValueError("Invalid values in r. Ensure x and y are valid.")
    rp,ra, num_roots = find_rp_ap(l,E)
    # Convert arrays to lists for animation
    x = x.tolist()
    y = y.tolist()
    t = t.tolist()
    r = r.tolist()

    # Font size settings
    font_size = 24  # 3x larger than typical default (default is ~12)
    legend_font_size = 15
    tick_font_size = 20

    # Create a figure with subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    # Add circles for rp and r0 on the left-hand side
    if ra != 0:
        r0 = ra  # Assign ra to r0 if ra is not zero
    else:
        r0 = r0  # Default value for r0 if ra is zero
    if num_roots == 0:
        rp = 2  # Default value for rp if num_roots is zero
    circle_rs = plt.Circle((0, 0), 2, color='k', fill=False, label='Circle (rs)')
    # Left-hand side: Trajectory plot
    axs[0].set_title('Trajectory in Schwarzschild space-time', fontsize=font_size)
    axs[0].set_xlabel('x', fontsize=font_size)
    axs[0].set_ylabel('y', fontsize=font_size)
    axs[0].grid()
    axs[0].axis('equal')  # Ensure equal scaling for x and y axes
    trajectory_line, = axs[0].plot([], [], '-', label='Trajectory')
    current_position, = axs[0].plot([], [], 'o', label='Current Position')
    axs[0].set_xlim(-r0-10, r0+10)
    axs[0].set_ylim(-r0-10, r0+10)
    axs[0].tick_params(axis='both', which='major', labelsize=tick_font_size)
    axs[0].legend(loc='upper right', fontsize=legend_font_size)

    # Right-hand side: Effective potential plot
    r_vals = np.linspace(2, r0+10, 1000)  # Avoid r=2 to prevent division by zero
    v_eff = gr_v_eff(r_vals, l)
    axs[1].set_title('$V_{eff}$', fontsize=font_size)
    axs[1].set_xlabel('r', fontsize=font_size)
    axs[1].set_ylabel('$E^2-1$', fontsize=font_size)
    axs[1].set_xlim(1, r0+5)
    axs[1].set_ylim(v_min*1.1, np.max([v_max*1.1,0.1]))
    axs[1].grid()
    axs[1].plot(r_vals, v_eff, label=f'Effective Potential, L={l}', color='b')
    if E < 1 and E > np.sqrt(1 + v_min):
        axs[1].plot([rp, ra], [E**2 - 1, E**2 - 1], color='r', linestyle='--', label=f'Energy Level $E={E:.4f}$')
    elif E >= 1 and E < np.sqrt(1 + v_max):
        axs[1].plot([rp, r0+10], [E**2 - 1, E**2 - 1], color='r', linestyle='--', label=f'Energy Level $E={E:.4f}$')
    elif E == np.sqrt(1 + v_min):
        axs[1].plot([r_min], [E**2 - 1], 'ro', label=f'Energy Level $E={E:.4f}$')
    elif E > np.sqrt(1 + v_max):
        axs[1].plot([0, r0+10], [E**2 - 1, E**2 - 1], color='r', linestyle='--', label=f'Energy Level $E={E:.4f}$')
    elif E == np.sqrt(1 + v_max):
        axs[1].plot([0], [E**2 - 1], 'ro', label=f'Energy Level $E={E:.4f}$')
    axs[1].legend(fontsize=legend_font_size)
    axs[1].tick_params(axis='both', which='major', labelsize=tick_font_size)
    particle_position, = axs[1].plot([], [], 'ro', label='Particle Position')
    axs[0].add_artist(circle_rs)
    # Add L={l} to the legend on the left subplot
    # Adjust layout to prevent label overlap and cutting
    plt.tight_layout(rect=[0, 0, 1, 1], pad=3.0)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12, wspace=0.25)

    # Initialization function
    def init():
        trajectory_line.set_data([], [])  # Initialize trajectory as empty
        current_position.set_data([], [])  # Initialize current position as empty
        particle_position.set_data([], [])  # Initialize particle position as empty
        return trajectory_line, current_position, particle_position

    # Frame update function
    def update(frame):
        index = min(frame * 1400, len(x) - 1)  # Double the step size for faster animation
        trajectory_line.set_data(x[:index], y[:index])  # Update trajectory up to current frame
        current_position.set_data([x[index]], [y[index]])  # Update current position as a single point
        particle_position.set_data([r[index]], [E**2 - 1])  # Update particle position on the potential plot
        return trajectory_line, current_position, particle_position

    # Build the animation
    frames = len(t) // 1400  # Halve the number of frames for faster animation
    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, repeat=True)  # Enable looping

    # Show the animation
    l_str = str(l).replace('.', '_')
    title = save_path + rf'\Trajectory and Effective Potential for l={l_str} E_index={i}.gif'
    print('Saving animation to:', title)
    fig.patch.set_alpha(0)  # Make the figure frame transparent
    for ax in axs:
        ax.set_facecolor('#FFFDF3')  # Set each subplot background to #FEFAF0

    anim.save(title, writer='imagemagick', fps=30, savefig_kwargs={'facecolor': 'none'})
    print('done')

    # Loop over l and E values

def show_animation_ISCO(i):
        x, y, t, f0, r0 = calc_time_angle_gr_ISCO()

        # Clean x, y, and t from NaN or unreal values
        valid_indices = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
        x = np.array(x[valid_indices])
        y = np.array(y[valid_indices])
        t = np.array(t[valid_indices])

        if len(x) == 0 or len(y) == 0 or len(t) == 0:
            raise ValueError("x, y, or t is empty after cleaning. Ensure valid input data.")

        r = np.sqrt(x**2 + y**2)
        if not np.all(np.isfinite(r)):
            raise ValueError("Invalid values in r. Ensure x and y are valid.")

        x = x.tolist()
        y = y.tolist()
        t = t.tolist()
        r = r.tolist()

        font_size = 24
        legend_font_size = 15
        tick_font_size = 20

        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        circle_rs = plt.Circle((0, 0), 2, color='k', fill=False, label='Circle (rs)')
        axs[0].set_title('GUI Trajectory', fontsize=font_size)
        axs[0].set_xlabel('x', fontsize=font_size)
        axs[0].set_ylabel('y', fontsize=font_size)
        axs[0].grid()
        axs[0].axis('equal')
        trajectory_line, = axs[0].plot([], [], '-', label='Trajectory')
        current_position, = axs[0].plot([], [], 'o', label='Current Position')
        axs[0].set_xlim(-r0-2, r0+2)
        axs[0].set_ylim(-r0-2, r0+2)
        axs[0].tick_params(axis='both', which='major', labelsize=tick_font_size)
        axs[0].legend(loc='upper right', fontsize=legend_font_size)

        r_vals = np.linspace(2, 6, 500000)
        l_ISCO = np.sqrt(12)
        v_eff = gr_v_eff(r_vals, l_ISCO)
        axs[1].set_title('$V_{eff}$ (GUI)', fontsize=font_size)
        axs[1].set_xlabel('r', fontsize=font_size)
        axs[1].set_ylabel('$E^2-1$', fontsize=font_size)
        axs[1].set_xlim(1, 6)
        axs[1].set_ylim(np.min(v_eff)-0.1, np.max(v_eff)+0.1)
        axs[1].grid()
        axs[1].plot(r_vals, v_eff, label='Effective Potential, $L=\sqrt{12}$', color='b')
        axs[1].plot([6], [gr_v_eff(6, l_ISCO)], 'ro', label='ISCO')
        particle_position, = axs[1].plot([], [], 'go', label='Particle Position')
        axs[1].legend(fontsize=legend_font_size)
        axs[1].tick_params(axis='both', which='major', labelsize=tick_font_size)
        
        axs[0].add_artist(circle_rs)
        plt.tight_layout(rect=[0, 0, 1, 1], pad=3.0)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12, wspace=0.25)

        def init():
            trajectory_line.set_data([], [])
            current_position.set_data([], [])
            particle_position.set_data([], [])
            return trajectory_line, current_position, particle_position

        def update(frame):
            index = min(frame * 1400, len(x) - 1)
            trajectory_line.set_data(x[:index], y[:index])
            current_position.set_data([x[index]], [y[index]])
            particle_position.set_data([r[index]], [gr_v_eff(r[index], l_ISCO)])
            return trajectory_line, current_position, particle_position

        frames = len(t) // 1400
        anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, repeat=True)

        title = save_path + rf'\Trajectory_and_Effective_Potential_ISCO_E_index={i}.gif'
        print('Saving ISCO animation to:', title)
        fig.patch.set_alpha(0)
        for ax in axs:
            ax.set_facecolor('#FFFDF3')
        plt.show()
        anim.save(title, writer='imagemagick', fps=20, savefig_kwargs={'facecolor': 'none'})
        print('done ISCO')


# show_animation_ISCO(0)





# # Calculate x, y, t for multiple l values with constant E=1
# l_values= [6,5,4.5,4.2,4.01,4.001,4.0001,4.00001,4.000001,4.0000001,4.00000001,4.000000001,4.0000000001,4.00000000001]
# # l_values = np.logspace(4+(10**-6),4.1, 100)  # Example range of l values

# # Loop through the l values and show animations one after the other
# for i, l in enumerate(l_values):
#     r_min, v_min = find_possible_values(l)
#     # print('E_min:',np.sqrt(1 + v_min))
#     # E=input('Enter E value: ')
#     # if E=='':
#     #     E=1
#     # E=float(E)
#     # if E<1:
#     #     E=1
#     # print('l:', l, 'E:', E)
#     E=1

    
#     print(f"Calculating trajectory for l={l}, E={E}, i={i}")
#     print(f"Animating for l={l}, E={E}")
#     show_animation(E, l, i)

# # Prepare GIF paths for each l value
# gif_paths_l = [
#     save_path + rf'\Trajectory and Effective Potential for l={str(l).replace(".", "_")} E_index={i}.gif'
#     for i, l in enumerate(l_values)
# ]
# print("GIF paths for varying l:", gif_paths_l)

# # Open all GIFs and extract their frames
# all_frames_l = []
# durations_l = []
# for gif_path in gif_paths_l:
#     with Image.open(gif_path) as im:
#         frames = []
#         try:
#             while True:
#                 frames.append(im.copy())
#                 durations_l.append(im.info.get('duration', 50))
#                 im.seek(im.tell() + 1)
#         except EOFError:
#             pass
#         all_frames_l.extend(frames)

# # Save the concatenated GIF for varying l
# output_path_l = save_path + r'\Combined_Trajectory_and_Effective_Potential_varying_l_4_v1.gif'
# if all_frames_l:
#     all_frames_l[0].save(
#         output_path_l,
#         save_all=True,
#         append_images=all_frames_l[1:],
#         duration=durations_l,
#         loop=0,
#         disposal=2
#     )
#     print(f"Combined GIF for varying l saved to {output_path_l}")
# else:
#     print("No frames found to combine for varying l.")
# --- Animation: Effective potential (zoomed near r_max, v_max) and full orbit for varying l ---

def animate_v_eff_and_orbit_zoomed():
    # l values from 4.2 to just above 4, spaced logarithmically for smoothness
    # Use a smooth logarithmic spacing for l values from 4.2 down to just above 4
    # Create a list of l values with increasing density near 4
    l_values_1 = np.concatenate([
        np.linspace(4.5, 4.1, 100, endpoint=False),
        np.linspace(4.1, 4.01, 100, endpoint=False),
        np.linspace(4.01, 4.001, 100, endpoint=False),
        np.linspace(4.001, 4.0001, 100, endpoint=False),
        np.linspace(4.0001, 4.00001, 100, endpoint=False),
        np.linspace(4.00001, 4.000001, 100, endpoint=False),
        np.linspace(4.000001, 4.0000001, 100, endpoint=False),
        np.linspace(4.0000001, 4.000000001, 700)
    ])
    # print('l_values_1:', l_values_1)
    # l_values_2 = np.linspace(4.0001, 4.0000000001, 200)
    # l_values = np.concatenate((l_values_1, l_values_2))
    l_values=l_values_1
    E = 1

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    font_size = 20

    # --- Left plot: Orbit ---
    axs[0].set_title('Full Orbit (E=1)', fontsize=font_size, pad=20)
    axs[0].set_xlabel('x', fontsize=font_size, labelpad=15)
    axs[0].set_ylabel('y', fontsize=font_size, labelpad=15)
    axs[0].grid()
    axs[0].axis('equal')
    orbit_line, = axs[0].plot([], [], '-', color='tab:blue', label='Orbit')
    axs[0].set_xlim(-120, 120)
    axs[0].set_ylim(-120, 120)
    axs[0].tick_params(axis='both', which='major', labelsize=22, direction='out', length=8, width=2, pad=10)
    axs[0].ticklabel_format(style='sci', axis='both', scilimits=(-2, 2), useMathText=True)
    axs[0].xaxis.get_offset_text().set_fontsize(22)
    axs[0].yaxis.get_offset_text().set_fontsize(22)
    axs[0].xaxis.set_tick_params(which='both', top=True, bottom=True)
    axs[0].yaxis.set_tick_params(which='both', left=True, right=True)
    axs[0].legend(fontsize=18, loc='upper right')

    # --- Right plot: Veff ---
    axs[1].axhline(0, color='red', linestyle='-', linewidth=1.5, label='E=1')
    axs[1].set_xlabel('r', fontsize=font_size, labelpad=15)
    axs[1].set_ylabel('$V_{eff}$', fontsize=font_size, labelpad=15)
    axs[1].grid()
    v_eff_line, = axs[1].plot([], [], color='tab:orange', label='$V_{eff}$')
    v_max_marker, = axs[1].plot([], [], 'ro', markersize=8, label='$V_{eff}$ max')
    axs[1].tick_params(axis='both', which='major', labelsize=22, direction='out', length=8, width=2, pad=10)
    axs[1].ticklabel_format(style='sci', axis='both', scilimits=(-2, 2), useMathText=True)
    axs[1].xaxis.get_offset_text().set_fontsize(22)
    axs[1].yaxis.get_offset_text().set_fontsize(22)
    axs[1].xaxis.set_tick_params(which='both', top=True, bottom=True)
    axs[1].yaxis.set_tick_params(which='both', left=True, right=True)
    axs[1].legend(fontsize=18, loc='upper right')

    # --- Title ---
    title = fig.suptitle('', fontsize=font_size+6, y=1.04)

    def init():
        orbit_line.set_data([], [])
        v_eff_line.set_data([], [])
        v_max_marker.set_data([], [])
        title.set_text('')
        return orbit_line, v_eff_line, v_max_marker, title

    def update(frame):
        l = l_values[frame]
        # Calculate orbit
        x, y, *_ = calc_time_angle_gr(l, E)
        x = np.asarray(x)
        y = np.asarray(y)
        valid_indices = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 1e6) & (np.abs(y) < 1e6)
        x = x[valid_indices]
        y = y[valid_indices]
        if len(x) == 0 or len(y) == 0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Invalid orbit data")
        orbit_line.set_data(x, y)

        # Effective potential
        r_max, v_max = find_possible_values2(l)
        if (l - 4) < 1e-9:
            num_points = 2000000
        elif (l - 4) < 1e-8:
            num_points = 160000
        elif (l - 4) < 1e-7:
            num_points = 120000
        elif (l - 4) < 1e-6:
            num_points = 100000
        elif (l - 4) < 1e-5:
            num_points = 8000
        elif (l - 4) < 1e-4:
            num_points = 6000
        elif (l - 4) < 1e-3:
            num_points = 400
        else:
            num_points = 300
        r_zoom = np.linspace(max(2, r_max-1.5), r_max+1.5, num_points)
        v_eff_zoom = gr_v_eff(r_zoom, l)
        v_eff_line.set_data(r_zoom, v_eff_zoom)
        v_max_marker.set_data([r_max], [v_max])

        # --- Zoom levels for axis limits ---
        zoom_levels = [
            {"threshold": 1e-8, "xlim": (3.9995, 4.0005), "ylim": (-5e-9, 5e-9)},
            {"threshold": 1e-7, "xlim": (3.999, 4.001), "ylim": (-5e-8, 5e-8)},
            {"threshold": 1e-6, "xlim": (3.997, 4.003), "ylim": (-5e-7, 5e-7)},
            {"threshold": 1e-5, "xlim": (3.99, 4.01), "ylim": (-5e-6, 5e-6)},
            {"threshold": 1e-4, "xlim": (3.8, 4.2), "ylim": (-5e-5, 5e-5)},
            {"threshold": 1e-3, "xlim": (3.6, 4.4), "ylim": (-5e-4, 5e-4)},
            {"threshold": 1e-2, "xlim": (3.5, 4.5), "ylim": (-5e-3, 5e-3)},
        ]
        default_xlim = (3, 5)
        default_ylim = (-0.2, 0.23)

        zoom_idx = None
        for idx, zl in enumerate(zoom_levels):
            if (l - 4) < zl["threshold"]:
                zoom_idx = idx
                break

        hold_frames = 10
        transition_frames = 15

        if zoom_idx is not None:
            zoom_start_frame = 0
            for i, zl in enumerate(zoom_levels):
                if i == zoom_idx:
                    break
                l_start = 4 + (zoom_levels[i]["threshold"])
                l_end = 4 + (zoom_levels[i-1]["threshold"]) if i > 0 else l_values[0]
                zoom_start_frame += np.sum((l_values < l_end) & (l_values >= l_start))
            rel_frame = frame - zoom_start_frame

            if rel_frame < hold_frames:
                axs[1].set_xlim(*zoom_levels[zoom_idx]["xlim"])
                axs[1].set_ylim(*zoom_levels[zoom_idx]["ylim"])
            elif rel_frame < hold_frames + transition_frames and zoom_idx > 0:
                prev_xlim = zoom_levels[zoom_idx-1]["xlim"]
                prev_ylim = zoom_levels[zoom_idx-1]["ylim"]
                curr_xlim = zoom_levels[zoom_idx]["xlim"]
                curr_ylim = zoom_levels[zoom_idx]["ylim"]
                alpha = (rel_frame - hold_frames) / transition_frames
                xlim = (
                    prev_xlim[0] * (1 - alpha) + curr_xlim[0] * alpha,
                    prev_xlim[1] * (1 - alpha) + curr_xlim[1] * alpha,
                )
                ylim = (
                    prev_ylim[0] * (1 - alpha) + curr_ylim[0] * alpha,
                    prev_ylim[1] * (1 - alpha) + curr_ylim[1] * alpha,
                )
                axs[1].set_xlim(*xlim)
                axs[1].set_ylim(*ylim)
            else:
                axs[1].set_xlim(*zoom_levels[zoom_idx]["xlim"])
                axs[1].set_ylim(*zoom_levels[zoom_idx]["ylim"])
        else:
            axs[1].set_xlim(*default_xlim)
            axs[1].set_ylim(*default_ylim)

        # --- Format scientific notation for ticks and move offset text to axis edge ---
        for ax in axs:
            ax.ticklabel_format(style='sci', axis='both', scilimits=(-2, 2), useMathText=True)
            ax.xaxis.get_offset_text().set_fontsize(22)
            ax.yaxis.get_offset_text().set_fontsize(22)
            # Move offset text to the edge of the axis
            ax.xaxis.get_offset_text().set_x(1.0)
            ax.yaxis.get_offset_text().set_y(1.0)

        # --- Title, placed above both subplots, no overlap ---
        title.set_text(f"$L={l:.12f}$, $E=1$")
        title.set_y(1.04)
        axs[1].set_title(f'Effective Potential $L-4={(l-4):.2e}$', fontsize=font_size, pad=20)  # Ensure axs[1] title is set

        # Ensure legends are shown in both axes
        axs[0].legend(fontsize=18, loc='upper right')
        axs[1].legend(fontsize=18, loc='upper right')

        return orbit_line, v_eff_line, v_max_marker, title

    anim = FuncAnimation(
        fig, update, frames=len(l_values), init_func=init, blit=True, repeat=True
    )

    output_path = save_path + r'\Orbit_and_Veff_zoomed_varying_l_2.gif'
    fig.patch.set_facecolor('#FFFDF3')
    for ax in axs:
        ax.set_facecolor('#FFFDF3')
    plt.subplots_adjust(top=0.88, bottom=0.13, left=0.08, right=0.98, wspace=0.28)
    anim.save(output_path, writer='imagemagick', fps=50, savefig_kwargs={'facecolor': 'none'})
    print(f"Zoomed orbit and Veff animation saved to {output_path}")

# animate_v_eff_and_orbit_zoomed()


def animate_v_eff_and_orbit_precession():
    # l values from 5 to 100, linearly spaced for smooth animation
    l_values = np.linspace(5, 30, 30)
    r_min_values=(l_values**2)/2+(l_values/2)*np.sqrt((l_values**2)-12)
    v_min_values=gr_v_eff(r_min_values,l_values)
    E_values=0.5*v_min_values

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    font_size = 20
    axs[0].set_title('Full Orbit (E=1)', fontsize=font_size)
    axs[0].set_xlabel('x', fontsize=font_size)
    axs[0].set_ylabel('y', fontsize=font_size)
    axs[0].grid()
    axs[0].axis('equal')
    orbit_line, = axs[0].plot([], [], '-', color='tab:blue')
    symmetry_axis_line, = axs[0].plot([], [], '--', color='tab:red', label='Symmetry Axis')
    axs[0].set_xlim(-120, 120)
    axs[0].set_ylim(-120, 120)
    axs[0].legend(fontsize=14)

    axs[1].set_xlabel('r', fontsize=font_size)
    axs[1].set_ylabel('$V_{eff}$', fontsize=font_size)
    axs[1].grid()
    v_eff_line, = axs[1].plot([], [], color='tab:orange')
    axs[1].tick_params(axis='both', which='major', labelsize=16)

    title = fig.suptitle('', fontsize=font_size+2)

    def init():
        orbit_line.set_data([], [])
        symmetry_axis_line.set_data([], [])
        v_eff_line.set_data([], [])
        title.set_text('')
        return orbit_line, symmetry_axis_line, v_eff_line, title

    def update(frame):
        l = l_values[frame]
        E=E_values[frame]
        # Calculate orbit
        x, y, t, f0, r0 = calc_time_angle_gr(l, E)
        x = np.asarray(x)
        y = np.asarray(y)
        f = np.asarray(t)  # Actually, t is time, but f is returned by calc_time_angle_gr as the 4th value
        # Clean x and y from NaN, inf, and unreal values
        valid_indices = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 1e6) & (np.abs(y) < 1e6)
        x = x[valid_indices]
        y = y[valid_indices]
        if len(x) == 0 or len(y) == 0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            orbit_line.set_data([], [])
            symmetry_axis_line.set_data([], [])
        else:
            orbit_line.set_data(x, y)
            # Estimate precession: find periapsis angles (where r is minimal)
            r = np.sqrt(x**2 + y**2)
            min_indices = np.where((r[1:-1] < r[:-2]) & (r[1:-1] < r[2:]))[0] + 1
            if len(min_indices) >= 2:
                # Take the last two periapsis points to define the axis of symmetry
                idx1, idx2 = min_indices[-2], min_indices[-1]
                x1, y1 = x[idx1], y[idx1]
                x2, y2 = x[idx2], y[idx2]
                # The axis is the line through the origin and the latest periapsis
                axis_angle = np.arctan2(y2, x2)
                axis_length = np.max(r)
                axis_x = np.array([0, axis_length * np.cos(axis_angle)])
                axis_y = np.array([0, axis_length * np.sin(axis_angle)])
                symmetry_axis_line.set_data(axis_x, axis_y)
            else:
                symmetry_axis_line.set_data([], [])

        # Effective potential
        r_vals = np.linspace(2.01, r0 + 30, 800)
        v_eff = gr_v_eff(r_vals, l)
        v_eff_line.set_data(r_vals, v_eff)
        axs[1].set_title(f'Effective Potential $L={l:.2e}$', fontsize=font_size)
        axs[1].set_xlim(2, r0 + 30)
        axs[1].set_ylim(np.nanmin(v_eff) - 0.1, np.nanmax(v_eff) + 0.1)
        title.set_text(f"$L={l:.2f}$, $E=1$")

        return orbit_line, symmetry_axis_line, v_eff_line, title

    anim = FuncAnimation(
        fig, update, frames=len(l_values), init_func=init, blit=True, repeat=True
    )

    output_path = save_path + r'\Orbit_and_Veff_precession_varying_l_large.gif'
    fig.patch.set_facecolor('#FFFDF3')
    for ax in axs:
        ax.set_facecolor('#FFFDF3')
    anim.save(output_path, writer='imagemagick', fps=30, savefig_kwargs={'facecolor': 'none'})
    print(f"Precession orbit and Veff animation saved to {output_path}")

# animate_v_eff_and_orbit_precession()
    # --- Section: Varying E, constant l ---
    # Set a constant l value
# l = 4.7
# r_min, v_min = find_possible_values(l)
# r_max, v_max = find_possible_values2(l)
# # Define a range of E values (example: from 0.95 to 1.05)
# E_start = np.sqrt(1 + v_max) - 0.001
# E_end = np.sqrt(1 + v_max) + 0.003
# E_values = np.linspace(E_start, np.sqrt(1 + v_max)-0.00000001, 6)
# # Add 3 more E values between np.sqrt(1 + v_max)-0.00000001 and E_end
# extra_E_values = np.linspace(np.sqrt(1 + v_max) - 0.000000001, E_end, 4, endpoint=False)
# E_values = np.concatenate((E_values, extra_E_values))

# # Loop through the E values and show animations one after the other
# for i, E in enumerate(E_values):
#     print(f"Calculating trajectory for l={l}, E={E}, i={i}")
#     print(f"Animating for l={l}, E={E}")
#     show_animation(E, l, i)

# # Prepare GIF paths for each E value
# gif_paths_E = [
#     save_path + rf'\Trajectory and Effective Potential for l={str(l).replace(".", "_")} E_index={i}.gif'
#     for i, E in enumerate(E_values)
# ]
# print("GIF paths for varying E:", gif_paths_E)

# # Open all GIFs and extract their frames
# all_frames_E = []
# durations_E = []
# for gif_path in gif_paths_E:
#     with Image.open(gif_path) as im:
#         frames = []
#         try:
#             while True:
#                 frames.append(im.copy())
#                 durations_E.append(im.info.get('duration', 50))
#                 im.seek(im.tell() + 1)
#         except EOFError:
#             pass
#         all_frames_E.extend(frames)

# # Save the concatenated GIF for varying E
# output_path_E = save_path + r'\Combined_Trajectory_and_Effective_Potential_varying_4_7_varinig_E_over_max.gif'
# if all_frames_E:
#     all_frames_E[0].save(
#         output_path_E,
#         save_all=True,
#         append_images=all_frames_E[1:],
#         duration=durations_E,
#         loop=0,
#         disposal=2
#     )
#     print(f"Combined GIF for varying E saved to {output_path_E}")
# else:
#     print("No frames found to combine for varying E.")


    # Animation: Effective potential as a function of l
# def animate_v_eff_l():
#     # Avoid l=0 to prevent invalid values in find_possible_values
#     l_values_1 = np.linspace(0.1, np.sqrt(12), 50)
#     l_values_2 = np.linspace(np.sqrt(12), 4, 50)
#     l_values_3 = np.linspace(4, 8, 400)
#     l_values = np.concatenate((l_values_1, l_values_2, l_values_3))

#     fig, ax = plt.subplots(figsize=(8, 6))
#     line, = ax.plot([], [], lw=2)
#     # vzero_line removed
#     title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontsize=18)
#     ax.set_xlabel('r', fontsize=16)
#     ax.set_ylabel('$V_{eff}$', fontsize=16)
#     ax.grid()

#     # Parameters for special points
#     l_isco = np.sqrt(12)
#     r_isco = 6
#     v_isco = gr_v_eff(r_isco, l_isco)  # Compute the actual minimum value
#     l_mbco = 4
#     r_mbco = 4
#     v_mbco = 0

#     # How many frames to hold at ISCO and MBCO
#     hold_frames = 20

#     # Build frame sequence: normal, hold at ISCO, normal, hold at MBCO, normal
#     isco_idx = np.argmin(np.abs(l_values - l_isco))
#     mbco_idx = np.argmin(np.abs(l_values - l_mbco))
#     frame_sequence = []
#     for i in range(len(l_values)):
#         frame_sequence.append(i)
#         if i == isco_idx:
#             frame_sequence.extend([i] * (hold_frames - 1))
#         if i == mbco_idx:
#             frame_sequence.extend([i] * (hold_frames - 1))

#     # Artists for ISCO and MBCO
#     isco_circle = plt.Circle((r_isco, v_isco), 0.01, color='red', fill=False, lw=2)
#     mbco_circle = plt.Circle((r_mbco, v_mbco), 0.01, color='green', fill=False, lw=2)
#     isco_text = ax.text(r_isco, v_isco + 0.1, '', color='red', fontsize=16, ha='center')
#     mbco_text = ax.text(r_mbco, v_mbco + 0.1, '', color='green', fontsize=16, ha='center')

#     def init():
#         line.set_data([], [])
#         title.set_text('')
#         isco_circle.set_visible(False)
#         mbco_circle.set_visible(False)
#         isco_text.set_text('')
#         mbco_text.set_text('')
#         if isco_circle not in ax.patches:
#             ax.add_patch(isco_circle)
#         if mbco_circle not in ax.patches:
#             ax.add_patch(mbco_circle)
#         return line, title, isco_circle, mbco_circle, isco_text, mbco_text

#     def update(frame):
#         idx = frame_sequence[frame]
#         l = l_values[idx]
#         # Choose r range
#         if l > 4:
#             try:
#                 r_min = ((l**2)/2)+(l/2)*np.sqrt((l**2)-12)
#                 r_max = ((l**2)/2)-(l/2)*np.sqrt((l**2)-12)
#                 v_min = gr_v_eff(r_min_formula, l)
#                 v_max = gr_v_eff(r_max_formula, l)
#                 r_vals = np.linspace(2.01, 5 * r_min, 1000)
#             except Exception:
#                 r_vals = np.linspace(2.01, 100, 1000)
#                 r_min = r_max = None
#         else:
#             r_vals = np.linspace(2.01, 100, 1000)
#             r_min = r_max = None
#         v_eff = gr_v_eff(r_vals, l)
#         line.set_data(r_vals, v_eff)
#         title.set_text(f'Effective Potential $V_{{eff}}$ for $L={l:.2f}$')
#         v_min_val, v_max_val = np.nanmin(v_eff), np.nanmax(v_eff)
#         threshold = 0.05 * (v_max_val - v_min_val)
#         indices = np.where((v_eff > v_min_val + threshold) & (v_eff < v_max_val - threshold))[0]

#         # --- Smooth transition for axis limits ---
#         # Compute the "target" limits for l>4 and l<=4
#         if l > 4:
#             try:
#                 r_min_formula = ((l**2)/2)+(l/2)*np.sqrt((l**2)-12)
#                 r_max_formula = ((l**2)/2)-(l/2)*np.sqrt((l**2)-12)
#                 r01=((l**2)/4)-(l/4)*np.sqrt((l**2)-16)
#                 r02=((l**2)/4)+(l/4)*np.sqrt((l**2)-16)
#                 v_min_formula = gr_v_eff(r_min_formula, l)
#                 v_max_formula = gr_v_eff(r_max_formula, l)
#                 x_pad = 0.5 * (r_min_formula - r_max_formula)
#                 y_pad = 0.05 * (v_max_formula - v_min_formula)
#                 xlim_target = (0,  4 * x_pad)
#                 ylim_target = (v_min_formula - y_pad, v_max_formula + 3 * y_pad)
#             except Exception:
#                 if len(indices) > 0:
#                     x_min = max(r_vals[0], r_vals[indices[0]] - 2)
#                     x_max = min(r_vals[-1], r_vals[indices[-1]] + 2)
#                     xlim_target = (x_min, x_max)
#                 else:
#                     xlim_target = (r_vals[0], r_vals[-1])
#                 ylim_target = (np.nanmin(v_eff) - 0.5, np.nanmax(v_eff) + 0.5)
#         else:
#             if len(indices) > 0:
#                 x_min = max(r_vals[0], r_vals[indices[0]] - 2)
#                 x_max = min(r_vals[-1], r_vals[indices[-1]] + 2)
#                 xlim_target = (x_min, x_max)
#             else:
#                 xlim_target = (r_vals[0], r_vals[-1])
#             ylim_target = (np.nanmin(v_eff) - 0.5, np.nanmax(v_eff) + 0.5)

#         # Smooth interpolation for l in [3.5, 4.5]
#         l_transition_min = 3.5
#         l_transition_max = 4.5
#         if l_transition_min < l < l_transition_max:
#             alpha = (l - l_transition_min) / (l_transition_max - l_transition_min)
#             # Compute both sets of limits
#             # For l<=4
#             if len(indices) > 0:
#                 x_min_l4 = max(r_vals[0], r_vals[indices[0]] - 2)
#                 x_max_l4 = min(r_vals[-1], r_vals[indices[-1]] + 2)
#                 xlim_l4 = (x_min_l4, x_max_l4)
#             else:
#                 xlim_l4 = (r_vals[0], r_vals[-1])
#             ylim_l4 = (np.nanmin(v_eff) - 0.5, np.nanmax(v_eff) + 0.5)
#             # For l>4
#             try:
#                 r_min_formula = ((l**2)/2)+(l/2)*np.sqrt((l**2)-12)
#                 r_max_formula = ((l**2)/2)-(l/2)*np.sqrt((l**2)-12)
#                 v_min_formula = gr_v_eff(r_min_formula, l)
#                 v_max_formula = gr_v_eff(r_max_formula, l)
#                 x_pad = 0.5 * (r_min_formula - r_max_formula)
#                 y_pad = 0.05 * (v_max_formula - v_min_formula)
#                 xlim_lg4 = (0, r_min_formula + 2 * x_pad)
#                 ylim_lg4 = (v_min_formula - y_pad, v_max_formula + 3 * y_pad)
#             except Exception:
#                 xlim_lg4 = xlim_l4
#                 ylim_lg4 = ylim_l4
#             # Interpolate
#             xlim_target = (
#                 (1 - alpha) * xlim_l4[0] + alpha * xlim_lg4[0],
#                 (1 - alpha) * xlim_l4[1] + alpha * xlim_lg4[1]
#             )
#             ylim_target = (
#                 (1 - alpha) * ylim_l4[0] + alpha * ylim_lg4[0],
#                 (1 - alpha) * ylim_l4[1] + alpha * ylim_lg4[1]
#             )

#         ax.set_xlim(*xlim_target)
#         ax.set_ylim(*ylim_target)
#         # --- End smooth transition ---

#         # ISCO
#         if idx == isco_idx:
#             isco_circle.set_visible(True)
#             isco_text.set_text('ISCO')
#         else:
#             isco_circle.set_visible(False)
#             isco_text.set_text('')
#         # MBCO
#         if idx == mbco_idx:
#             mbco_circle.set_visible(True)
#             mbco_text.set_text('MBCO')
#         else:
#             mbco_circle.set_visible(False)
#             mbco_text.set_text('')
#         return line, title, isco_circle, mbco_circle, isco_text, mbco_text


#     anim = FuncAnimation(
#         fig, update, frames=len(frame_sequence), init_func=init, blit=True, repeat=True
#     )

#     output_path = save_path + r'\Effective_Potential_Varying_l_gr_2.gif'
#     fig.patch.set_facecolor('#FFFDF3')
#     ax.set_facecolor('#FFFDF3')
#     anim.save(output_path, writer='imagemagick', fps=10, savefig_kwargs={'facecolor': 'none'})
#     print(f"Effective potential animation saved to {output_path}")

# animate_v_eff_agianl()

# Define source andination directories
# import concurrent.futures

# gifs_dir = r"C:\Users\itama\Documents\.venv\Scripts\Research scripts\gifspnp\Finals"
# wmv_dir_2 = os.path.join(gifs_dir, "webems2")
# os.makedirs(wmv_dir_2, exist_ok=True)
# gif_files = glob.glob(os.path.join(gifs_dir, "*.gif"))
# def convert_to_wmv(gif_path):
#     base_name = os.path.splitext(os.path.basename(gif_path))[0]
#     wmv_path = os.path.join(wmv_dir_2, base_name + ".wmv")
#     try:
#         # Load GIF and convert transparent background to #FFFDF3
#         with Image.open(gif_path) as im:
#             frames = []
#             for frame in ImageSequence.Iterator(im):
#                 frame = frame.convert("RGBA")
#                 bg = Image.new("RGBA", frame.size, "#FFFDF3")
#                 bg.paste(frame, mask=frame.split()[3])  # Use alpha channel as mask
#                 frames.append(bg.convert("RGB"))
#             temp_gif = os.path.join(wmv_dir_2, base_name + "_temp.gif")
#             frames[0].save(
#                 temp_gif,
#                 save_all=True,
#                 append_images=frames[1:],
#                 duration=im.info.get('duration', 50),
#                 loop=0,
#                 disposal=2
#             )

#         clip = VideoFileClip(temp_gif)
#         clip = clip.resize(height=max(clip.h, 1080))
#         clip.write_videofile(
#             wmv_path,
#             codec='wmv2',
#             audio=False,
#             fps=clip.fps,
#             bitrate="12000k",
#             threads=4,
#             verbose=False,
#             logger=None
#         )
#         clip.close()
#         os.remove(temp_gif)
#         print(f"Converted {gif_path} to {wmv_path}")
#     except Exception as e:
#         print(f"Failed to convert {gif_path}: {e}")

# with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(gif_files))) as executor:
#     list(tqdm(executor.map(convert_to_wmv, gif_files), total=len(gif_files), desc="Converting GIFs"))

# import matplotlib.pyplot as plt
# while True:
#     # Set parameters
#     l = 20
#     r_min, v_min = find_possible_values(l)
#     # print(np.sqrt(v_min + 1))
#     # E = rnd.uniform(np.sqrt(v_min + 1), 1) 
#     E=0.9999375910501996
#     print('E=',E)

#     # Calculate trajectory for 3 loops
#     x, y, t, f0, r0 = calc_time_angle_gr(l, E)

#     # Convert to numpy arrays for easier manipulation
#     x = np.asarray(x)
#     y = np.asarray(y)
#     r = np.sqrt(x**2 + y**2)
#     # Remove trailing zeros from x and y
#     # Remove trailing points where both x and y are close to zero (up to 0.1)
#     # Remove trailing points where both x and y are close to zero (up to 0.1)
#     while len(x) > 0 and len(y) > 0 and abs(x[-1]) <= 0.1 and abs(y[-1]) <= 0.1:
#         x = x[:-1]
#         y = y[:-1]
#     # Remove NaN values from x and y
#     valid = np.isfinite(x) & np.isfinite(y)
#     x = x[valid]
#     y = y[valid]
#     # Find periapsis and apoapsis indices (local minima and maxima of r)

#     # Find local maxima (apoapsis)
#     ra_indices = argrelextrema(r, np.greater)[0]
#     # Ensure the first point is included if it's an apoapsis
#     if len(ra_indices) == 0 or ra_indices[0] != 0:
#         ra_indices = np.insert(ra_indices, 0, 0)

#     # Plot the trajectory
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.plot(x, y, label='Trajectory')
#     # ax.scatter(x[ra_indices], y[ra_indices], color='red', zorder=5, label='Apoapsis (ra)')
#     ax.scatter(x[0], y[0], color='green', zorder=6, label='Start')
#     print(x)
#     print(y)

#     # Draw arc from start to the point 100000 indices after
#     idx0 = 0
#     idx1 = -1
#     theta0 = np.arctan2(y[idx0], x[idx0])
#     theta1 = np.arctan2(y[idx1], x[idx1])
#     # Ensure theta1 > theta0 for proper arc direction
#     if theta1 < theta0:
#         theta1 += 2 * np.pi
#     arc_theta = np.linspace(theta0, theta1, 200)
#     arc_r = np.sqrt(x[idx0]**2 + y[idx0]**2)
#     arc_x = arc_r * np.cos(arc_theta)
#     arc_y = arc_r * np.sin(arc_theta)
#     ax.plot(arc_x, arc_y, color='purple', lw=3, label='Precession Arc')
#     # Place the annotation above the arc midpoint
#     mid_idx = len(arc_theta) // 2

#     # Draw a second arc: broken orange line, counterclockwise by 6pi/(l**2) radians from the start
#     arc_angle = 6 * np.pi / (l ** 2)
#     arc_theta2 = np.linspace(theta0, theta0 + arc_angle, 200)
#     arc_x2 = arc_r * np.cos(arc_theta2)
#     arc_y2 = arc_r * np.sin(arc_theta2)
#     frac = Fraction(6, l**2).limit_denominator()
#     arc_label = rf"$\frac{{6\pi}}{{L^2}}$ Arc "
#     ax.plot(arc_x2, arc_y2, color='orange', lw=3, ls='--', label=arc_label)
#     # Mark the end of the arc with a red dot (no label)
#     ax.scatter(arc_x[-1], arc_y[-1], color='red', zorder=7)

#     ax.set_xlabel('x', fontsize=26)
#     ax.set_ylabel('y', fontsize=26)
#     ax.set_title(f'Elliptical Trajectory for $L={l}$', fontsize=28)
#     fig.patch.set_alpha(0)
#     ax.set_facecolor('none')
#     ax.axis('equal')
#     ax.legend(fontsize=22)
#     ax.grid(True)

#     # Add an inset axes in the bottom right, zoomed to 500 units around the arc center
#     # Center of the arc (midpoint)
#     center_x = arc_x[mid_idx]
#     center_y = arc_y[mid_idx]
#     # Create inset axes (width, height in fraction of parent axes)
#     axins = inset_axes(ax, width="35%", height="35%", loc='lower right', borderpad=2)
#     # Plot the same trajectory in the inset
#     axins.plot(x, y, label='Trajectory')
#     axins.plot(arc_x, arc_y, color='purple', lw=3)
#     axins.plot(arc_x2, arc_y2, color='orange', lw=3, ls='--')
#     axins.scatter(arc_x[-1], arc_y[-1], color='red', zorder=7)
#     axins.scatter(x[0], y[0], color='green', zorder=6)
#     # Set zoomed limits
#     axins.set_xlim(center_x - 500, center_x + 500)
#     axins.set_ylim(center_y - 500, center_y + 500)
#     axins.set_xticks([])
#     axins.set_yticks([])
#     axins.set_facecolor('none')
#     axins.set_title("Zoomed Arc", fontsize=14)
#     axins.grid(True, alpha=0.3)

#     # Increase tick font size
#     ax.tick_params(axis='both', which='major', labelsize=22)
#     plt.tight_layout()
#     plt.show(block=False)
#     plt.pause(0.1)
#     booli = input("Do you want to continue? (y/n): ")
#     if booli.lower() != 'y':
#         break
#     plt.close(fig)

# Create a GIF of the particle orbit for l=4.7 and E=0.99 without the effective potential plot
# def animate_orbit_only(l=4.7, E=0.99, save_path=save_path):
#     x, y, t, f0, r0 = calc_time_angle_gr(l, E)
#     x = np.asarray(x)
#     y = np.asarray(y)
#     valid_indices = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 1e6) & (np.abs(y) < 1e6)
#     x = x[valid_indices]
#     y = y[valid_indices]
#     t = np.asarray(t)[valid_indices]

#     fig, ax = plt.subplots(figsize=(8, 8))
#     font_size = 24
#     ax.set_title(f'Orbit for $L={l}$, $E={E}$', fontsize=font_size)
#     ax.set_xlabel('x', fontsize=font_size)
#     ax.set_ylabel('y', fontsize=font_size)
#     ax.grid()
#     ax.axis('equal')
#     # Plot the dark hole as a circle with radius 2 around the origin
#     dark_hole = plt.Circle((0, 0), 2, color='k', fill=True, alpha=0.3, label='Black Hole (r=2)')
#     ax.add_artist(dark_hole)
#     orbit_line, = ax.plot([], [], '-', color='tab:blue', label='Orbit')
#     current_position, = ax.plot([], [], 'o', color='red', label='Particle')
#     ax.set_xlim(np.min(x)-10, np.max(x)+10)
#     ax.set_ylim(np.min(y)-10, np.max(y)+10)
#     ax.legend(fontsize=18)

#     def init():
#         orbit_line.set_data([], [])
#         current_position.set_data([], [])
#         return orbit_line, current_position

#     def update(frame):
#         index = min(frame * 1400, len(x) - 1)
#         orbit_line.set_data(x[:index], y[:index])
#         current_position.set_data([x[index]], [y[index]])
#         return orbit_line, current_position

#     frames = len(t) // 1400
#     anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, repeat=True)
#     output_path = os.path.join(save_path, f'Orbit_only_L_{str(l).replace(".", "_")}_E_{str(E).replace(".", "_")}.gif')
#     fig.patch.set_facecolor('#FFFDF3')
#     ax.set_facecolor('#FFFDF3')
#     plt.tight_layout()
#     anim.save(output_path, writer='imagemagick', fps=30, savefig_kwargs={'facecolor': 'none'})
#     print(f"Orbit-only animation saved to {output_path}")

# # Run the animation for the requested parameters
# # animate_orbit_only(l=4.7, E=0.99)

# def cut_and_loop_gif(input_path, output_path, cut_time_sec=13):
#     with Image.open(input_path) as im:
#         frames = []
#         durations = []
#         total_time = 0
#         for frame in ImageSequence.Iterator(im):
#             duration = frame.info.get('duration', 50)
#             if total_time + duration > cut_time_sec * 1000:
#                 break
#             frames.append(frame.copy())
#             durations.append(duration)
#             total_time += duration

#         if not frames:
#             raise ValueError("No frames found before cut time.")

#         # Loop the cut segment
#         frames = frames * 2
#         durations = durations * 2

#         frames[0].save(
#             output_path,
#             save_all=True,
#             append_images=frames[1:],
#             duration=durations,
#             loop=0,
#             disposal=2
#         )
#         print(f"Looped GIF saved to {output_path}")

# # Usage:
# input_gif = r"C:\Users\itama\Documents\.venv\Scripts\Research scripts\gifspnp\Orbit_only_L_4_7_E_0_99.gif"
# output_gif = r"C:\Users\itama\Documents\.venv\Scripts\Research scripts\gifspnp\Orbit_only_L_4_7_E_0_99_looped.gif"
# cut_and_loop_gif(input_gif, output_gif, cut_time_sec=13)

# import matplotlib.pyplot as plt

# l = 4
# r_vals = np.linspace(1, 15, 1000)
# v_eff = gr_v_eff(r_vals, l)
# r_mbco = 4
# v_mbco = 0  # For l=4, E^2-1=0 at r=4

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(r_vals, v_eff, label='$V_{eff}$ for $L=4$', color='b')
# # Mark MBCO with a star marker
# ax.plot(r_mbco, v_mbco, marker='*', color='orange', markersize=18, label='MBCO')
# ax.annotate('MBCO', xy=(r_mbco, v_mbco), xytext=(r_mbco+0.5, v_mbco+0.05),
#             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=14, color='orange')
# ax.set_xlabel('r', fontsize=16)
# ax.set_ylabel('$V_{eff}$', fontsize=16)
# ax.set_xlim(1, 15)
# # Center the MBCO point in y-limits
# ax.set_ylim(v_mbco - 0.2, v_mbco + 0.2)
# ax.legend(fontsize=14)
# ax.grid(True, alpha=0.3)
# fig.patch.set_alpha(0)
# ax.set_facecolor('none')
# plt.tight_layout()
# plt.show()



# Plot Veff for L=4.7 from r=0 to r=20 with large fonts, thick line, transparent background
l = 4.7
r_vals = np.linspace(0.01, 40, 1000)
v_eff = gr_v_eff(r_vals, l)

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(r_vals, v_eff, color='blue', linewidth=3, label=r'$V_{eff}$')
ax.set_xlabel('r', fontsize=24)
ax.set_ylabel(r'$E^2-1$', fontsize=24)
ax.set_title(r'Effective Potential $V_{eff}$ for $L=4.7$', fontsize=28)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.legend(fontsize=20)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 40)
ax.set_ylim(np.nanmin(v_eff[400:]) - 0.01, np.nanmax(v_eff) + 0.01)
fig.patch.set_alpha(0)
ax.set_facecolor('none')
plt.tight_layout()
plt.show()