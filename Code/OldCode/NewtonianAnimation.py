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
# from IPython.display import HTML


# Defining the equations of motion for the Schwarzschild metric
# def L(rp,E):
#     return ((((E**2)/(1-(2/rp)))-1)*(rp**2))**(1/2)

save_path=r"C:\Users\itama\Documents\.venv\Scripts\Research scripts\gifspnp"

def N_r_dot(r, E, l):
    return  np.sqrt(2*E + (2 / r) -((l**2) / (r**2)))

def N_v_eff(r, l):
    return -(1 / r) + ((l**2) /(2*(r**2))) 

def N_v_eff_derivative(r, l):
    return (1 / r**2) - (l**2) / (r**3)

def N_f_dot(r, E, l):
    return  (l / (r**2))

# Solving the equations of motion using scipy's odeint

def equations(y, t, E, l):
    r, f = y
    drdt = -N_r_dot(r, E, l)
    dfdot = N_f_dot(r, E, l)
    return [drdt, dfdot]

def equationsr(y, t, E, l):
    r, f = y
    drdt = N_r_dot(r, E, l)
    dfdot = N_f_dot(r, E, l)
    return [drdt, dfdot]

def find_rp_ap(l,E):

    roots = np.roots([2*E, 2,-l**2])
    real_roots = [r.real for r in roots if np.isreal(r) and r > 2]
    real_roots.sort()
    if len(real_roots) == 1:
        return real_roots[0] ,0 , 1
    elif len(real_roots) == 2:
        return real_roots[0] ,real_roots[1] ,2
    else:
        return 0, 0, 0

def find_possible_values(l):
    # Find the local minimum of the effective potential above rp
    rp, ra, num_roots = find_rp_ap(l, 1)
    result = sp.optimize.minimize_scalar(
        lambda r: N_v_eff(r, l),
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


r0 = 100.0  # initial radius
rp = 8  # periapsis radius
f0 = 0  # initial angle
E = 1  # energy per unit mass (adjusted to ensure a valid trajectory)
y0 = [r0, f0]  # initial conditions vector

# Calculate the derivative of the effective potential

def calc_time_angle_gr(l, E):
    # Calculate the time of periapsis passage and angle at periapsis passage

    r_min,v_min = find_possible_values(l)


    r0=100
    # f0=rnd.uniform(0,np.pi/2)
    f0=np.pi/6
    rp,ra, num_roots = find_rp_ap(l,E)
    rp=rp+10**-7
    print('num_roots:', num_roots)
    if E >= 0:
        tp = sp.integrate.quad(lambda r: -1/N_r_dot(r, E, l), r0, rp)[0]  # time of periapsis passage
        fp = f0 + sp.integrate.quad(lambda r: -N_f_dot(r, E, l)/N_r_dot(r, E, l), r0, rp)[0]  # angle at periapsis passage
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
            
    elif E<0 and E>v_min:
        tp = sp.integrate.quad(lambda r: -1/N_r_dot(r, E, l), ra, rp)[0]  # time of periapsis passage
        ttmp= sp.integrate.quad(lambda r: -1/N_r_dot(r, E, l), ra-0.5, rp)[0]  # time of periapsis passage
        fp = f0 + sp.integrate.quad(lambda r: -N_f_dot(r, E, l)/N_r_dot(r, E, l), ra, rp)[0]  # angle at periapsis passage
        num_periods = 2  # Number of periods to simulate
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
            y0_return = [rp+10**-7, f_full[-1]]
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
    elif E==v_min:
        
        # For a circular orbit, r remains constant at rp, and the angular frequency is given by gr_f_dot(rp, E, rp)
        t = np.linspace(0, 3 * 2 * np.pi / N_f_dot(r_min, E, l), 50000)  # Time array for three periods
        f = N_f_dot(r_min, E, l) * t  # Angular position as a function of time

        # Convert polar coordinates to Cartesian coordinates
        x = r_min * np.cos(f)
        y = r_min * np.sin(f)
        t=t


    return x, y, t, f0












def show_animation(E, l, i):
    r0 = 100.0  # initial radius

    r_min, v_min = find_possible_values(l)
    x, y, t ,f0 = calc_time_angle_gr(l, E)

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
        r0 = 100  # Default value for r0 if ra is zero
    circle_rs = plt.Circle((0, 0), 2, color='k', fill=False, label='Circle (rs)')
    # Left-hand side: Trajectory plot
    axs[0].set_title('Trajectory in flat space-time', fontsize=font_size)
    axs[0].set_xlabel('x', fontsize=font_size)
    axs[0].set_ylabel('y', fontsize=font_size)
    axs[0].grid()
    axs[0].axis('equal')  # Ensure equal scaling for x and y axes
    trajectory_line, = axs[0].plot([], [], '-', label='Trajectory')
    current_position, = axs[0].plot([], [], 'o', label='Current Position')
    axs[0].set_xlim(-100, 100)
    axs[0].set_ylim(-100, 100)
    axs[0].tick_params(axis='both', which='major', labelsize=tick_font_size)
    axs[0].legend(loc='upper right', fontsize=legend_font_size)

    # Right-hand side: Effective potential plot
    r_vals = np.linspace(2, 100, 1000)  # Avoid r=2 to prevent division by zero
    v_eff = N_v_eff(r_vals, l)
    axs[1].set_title('$V_{eff}$', fontsize=font_size)
    axs[1].set_xlabel('r', fontsize=font_size)
    axs[1].set_ylabel('$E$', fontsize=font_size)
    axs[1].set_xlim(1, 100)
    axs[1].set_ylim(-0.1, 0.4)
    axs[1].grid()
    axs[1].plot(r_vals, v_eff, label=f'Effective Potential, $L={l}$', color='b')
    if E >= 0:
        axs[1].plot([rp, r0], [E, E], color='r', linestyle='--', label=f'Energy Level $E= {E}$')
    elif E == v_min:
        axs[1].plot([r_min], [E], 'ro', label=f'Energy Level $E= {E}$')
    elif v_min<E<0:
        axs[1].plot([rp, r0], [E,E], color='r', linestyle='--', label=f'Energy Level $E= {E}$')
    axs[1].legend(fontsize=legend_font_size)
    axs[1].tick_params(axis='both', which='major', labelsize=tick_font_size)
    particle_position, = axs[1].plot([], [], 'ro', label='Particle Position')
    axs[0].add_artist(circle_rs)

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
        index = min(frame * 700, len(x) - 1)  # Increase the step size to make the animation faster
        trajectory_line.set_data(x[:index], y[:index])  # Update trajectory up to current frame
        current_position.set_data([x[index]], [y[index]])  # Update current position as a single point
        particle_position.set_data([r[index]], [E])  # Update particle position on the potential plot
        return trajectory_line, current_position, particle_position

    # Build the animation
    frames = len(t) // 700  # Adjust the number of frames for faster animation
    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, repeat=True)  # Enable looping

    # Show the animation
    l_str = str(l).replace('.', '_')
    title = save_path + rf'\Trajectory and Effective Potential for Newt l={l_str} E_index={i}.gif'
    print('Saving animation to:', title)
    fig.patch.set_alpha(0)  # Make the figure frame transparent
    for ax in axs:
        ax.set_facecolor('#FFFDF3')  # Set each subplot background to #FEFAF0
    anim.save(title, writer='imagemagick', fps=40, savefig_kwargs={'facecolor': 'none'})
    print('done')


    # Loop over l and E values










# # Set l to a fixed value
# l = 4.7

# # Find v_min for this l
# r_min, v_min = find_possible_values(l)

# # Define E values as requested
# E_values = [0.3, 0, v_min + 0.01, v_min]

# # Loop through the E values and show animations one after the other
# for i, E in enumerate(E_values):
#     print(f"Calculating trajectory for l={l}, E={E}, i={i}")
#     print(f"Animating for l={l}, E={E}")
#     show_animation(E, l, i)

# # Prepare GIF paths for each E value
# gif_paths_E = [
#     save_path + rf'\Trajectory and Effective Potential for Newt l={str(l).replace(".", "_")} E_index={i}.gif'
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
# output_path_E = save_path + r'\Combined_Trajectory_and_Effective_Potential_varying_E_Neqtonian.gif'
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
def animate_v_eff_l():
    # Avoid l=0 to prevent invalid values in find_possible_values
    l_values = np.linspace(0.1, 100, 1000)
    

    fig, ax = plt.subplots(figsize=(8, 6))
    line, = ax.plot([], [], lw=2)
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontsize=18)
    ax.set_xlabel('r', fontsize=16)
    ax.set_ylabel('$V_{eff}$', fontsize=16)
    ax.grid()

    def init():
        line.set_data([], [])
        title.set_text('')
        return line, title

    def update(frame):
        l = l_values[frame]
        try:
            r_min=2*(l**2)
            v_min = N_v_eff(r_min, l)
            # print(v_min)
            r_vals = np.linspace(0.01, 3*r_min, 1000)
            v_eff = N_v_eff(r_vals, l)
            line.set_data(r_vals, v_eff)
            title.set_text(f'Effective Potential $V_{{eff}}$ for $L={l:.2f}$')
            ax.set_xlim(0,r_min*2)
            ax.set_ylim(v_min*1.5,np.abs(v_min)*1.5)
        except Exception as e:
            print(f"Skipping frame {frame} for l={l}: {e}")
            line.set_data([], [])
            title.set_text(f'Skipped l={l:.2f}')
        return line, title

    anim = FuncAnimation(
        fig, update, frames=len(l_values), init_func=init, blit=True, repeat=True
    )

    output_path = save_path + r'\Effective_Potential_Varying_l.gif'
    fig.patch.set_facecolor('#FFFDF3')
    ax.set_facecolor('#FFFDF3') 
    anim.save(output_path, writer='imagemagick', fps=10, savefig_kwargs={'facecolor': 'none'})
    print(f"Effective potential animation saved to {output_path}")

# animate_v_eff_l()


def newtonian_example_animation():
    # Parameters
    l = 4.7
    _, v_min = find_possible_values(l)
    E = v_min + 0.01
    x, y, t, _ = calc_time_angle_gr(l, E)
    x = np.array(x)
    y = np.array(y)
    t = np.array(t)
    # # Initial particle position
    # x0, y0 = x[0], y[0]

    # # Set up figure with larger size and LaTeX font
    # plt.rcParams.update({
    #     'font.size': 26,
    #     'font.family': 'serif',
    #     'text.usetex': False  # Disable LaTeX rendering to avoid missing package errors
    # })
    # fig, ax = plt.subplots(figsize=(14, 14))  # Increased figure size for more space
    # ax.set_xlim(-110, 110)  # Widened limits for more margin
    # ax.set_ylim(-110, 125)  # Increased upper limit to avoid cutting text
    # ax.set_aspect('equal')
    # ax.set_facecolor('#FFFDF3')
    # fig.patch.set_facecolor('#FFFDF3')
    # ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7, zorder=1)
    # ax.axis('on')  # Show axes for grid

    # # Add x and y labels
    # ax.set_xlabel('x', fontsize=28)
    # ax.set_ylabel('y', fontsize=28)

    # # Black hole (circle at origin, radius 2)
    # black_hole = plt.Circle((0, 0), 2, color='k', zorder=10)
    # ax.add_patch(black_hole)
    # # Particle (red dot)
    # particle, = ax.plot([], [], 'ro', markersize=6, zorder=20)
    # # Trajectory
    # trajectory, = ax.plot([], [], '-', color='blue', lw=3, zorder=5)

    # # Texts
    # text_bh = ax.text(0, -15, '', ha='center', va='center', fontsize=32, color='k', zorder=30, family='serif')
    # text_particle = ax.text(x0 - 50 , y0 +10, '', ha='left', va='bottom', fontsize=32, color='r', zorder=30, family='serif')
    # text_conserved = ax.text(0, 0, '', ha='center', va='center', fontsize=25, color='purple', zorder=30, family='serif')

    # # Animation steps
    # hold_frames = 40  # ~2 seconds at 20 fps
    # move_frames = len(x) // 700
    # total_frames = hold_frames * 3 + move_frames

    # def init():
    #     black_hole.set_visible(False)
    #     particle.set_data([], [])
    #     trajectory.set_data([], [])
    #     text_bh.set_text('')
    #     text_particle.set_text('')
    #     text_conserved.set_text('')
    #     return black_hole, particle, trajectory, text_bh, text_particle, text_conserved

    # def animate(frame):
    #     # 1. Show black hole and text
    #     if frame < hold_frames:
    #         black_hole.set_visible(True)
    #         text_bh.set_text(r'A Black Hole')
    #         text_bh.set_position((0, -15))
    #         particle.set_data([], [])
    #         text_particle.set_text('')
    #         text_conserved.set_text('')
    #         trajectory.set_data([], [])
    #     # 2. Show particle and text
    #     elif frame < 2 * hold_frames:
    #         black_hole.set_visible(True)
    #         text_bh.set_text('')
    #         particle.set_data([x0], [y0])
    #         text_particle.set_text(r'A test particle')
    #         text_particle.set_position((x0 - 50, y0 + 10))  # Move text further away from particle
    #         text_conserved.set_text('')
    #         trajectory.set_data([], [])
    #     # 3. Show conservation text
    #     elif frame < 3 * hold_frames:
    #         black_hole.set_visible(True)
    #         text_bh.set_text('')
    #         particle.set_data([x0], [y0])
    #         text_particle.set_text('')
    #         # Midpoint between origin and particle
    #         xm, ym = x0 / 2, y0 / 2
    #         text_conserved.set_text(r'$L$ and $E$ are conserved')
    #         text_conserved.set_position((xm, ym))
    #         trajectory.set_data([], [])
    #     # 4. Animate trajectory
    #     else:
    #         black_hole.set_visible(True)
    #         text_bh.set_text('')
    #         text_particle.set_text('')
    #         text_conserved.set_text('')
    #         idx = min((frame - 3 * hold_frames) * 700, len(x) - 1)
    #         trajectory.set_data(x[:idx], y[:idx])
    #         particle.set_data([x[idx]], [y[idx]])
    #     return black_hole, particle, trajectory, text_bh, text_particle, text_conserved

    # anim = FuncAnimation(
    #     fig, animate, frames=total_frames, init_func=init, blit=True, repeat=False
    # )
    # plt.show()
    output_path = os.path.join(save_path, 'newtonian_example_gif.gif')
    # # Use PillowWriter for GIF output
    # from matplotlib.animation import PillowWriter
    # anim.save(output_path, writer=PillowWriter(fps=30), dpi=200)
    # print(f"Newtonian example animation saved to {output_path}")

    # Convert GIF to high quality WMV using moviepy
    from moviepy.video.io.VideoFileClip import VideoFileClip
    gif_clip = VideoFileClip(output_path)
    # Ensure width and height are even
    w, h = gif_clip.size
    new_w = w if w % 2 == 0 else w - 1
    new_h = h if h % 2 == 0 else h - 1
    if (new_w, new_h) != (w, h):
        gif_clip = gif_clip.resized(new_size=(new_w, new_h))
    wmv_path = os.path.join(save_path, 'newtonian_example_high_quality2.wmv')
    gif_clip.write_videofile(wmv_path, codec='wmv2', fps=20, audio=False, bitrate="5000k")
    print(f"WMV file saved to {wmv_path}")

newtonian_example_animation()