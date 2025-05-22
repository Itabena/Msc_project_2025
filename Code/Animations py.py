import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio

#test test test


# Defining the equations of motion for the Schwarzschild metric
def gr_r_dot(r,E,L):
    return ((1-(2/r))/E)*np.sqrt((E**2)-(1-(2/r))*(1+(L**2)/(r**2)))
def gr_v_eff(r,E,L):
    return (1-(2/r))*(1+(L**2)/(r**2)) - 1
def gr_f_dot(r,E,L):
    return (1-(2/r))*(L/(E*(r**2)))

#Solving the equations of motion using scipy's odeint
from scipy.integrate import odeint

def equations(y, t, E, L):
    r, f = y
    drdt = -gr_r_dot(r, E, L)
    dfdot = gr_f_dot(r, E, L)
    return [drdt, dfdot]


def equationsr(y, t, E, L):
    r, f = y
    drdt = gr_r_dot(r, E, L)
    dfdot = gr_f_dot(r, E, L)
    return [drdt, dfdot]

def equationepoapsis(r, E, L):
    return (E**2-1)(r**3)+2*(r**2)-(L**2)*r+2*L**2
def Calac_apoapsis(r0, E, L):
    # Calculate the apoapsis radius
    r = sp.optimize.fsolve(equationepoapsis, r0, args=(E, L))
    return r[0]


#####COntinue from here######

# Initial conditions
r0 = 100.0  # initial radius
rp = 6  # periapsis radius
f0 = 1.0   # initial angle
E = 1   # energy per unit mass
L = ((((E**2)/(1-(2/rp)))-1)*(rp**2))**(1/2)    # angular momentum per unit mass
y0 = [r0, f0]  # initial conditions vector
print('L:', L)

tp = sp.integrate.quad(lambda r: -1/gr_r_dot(r, E, L), r0, rp)[0]  # time of periapsis passage
fp = f0 + sp.integrate.quad(lambda r: -gr_f_dot(r, E, L)/gr_r_dot(r, E, L), r0, rp)[0]  # angle at periapsis passage
# Time points for the approaching side
t_approach = np.linspace(0, tp, 500000)

# Solve the equations of motion for the approaching side
sol_approach = odeint(equations, y0, t_approach, args=(E, L))

# Extract r and f for the approaching side
r_approach = sol_approach[:, 0]
f_approach = sol_approach[:, 1]

# Time points for the returning side
t_return = np.linspace(tp, 2*tp, 500000)

# Reverse the initial conditions for the returning side
y0_return = [rp+10**-10, fp]

# Solve the equations of motion for the returning side
sol_return = odeint(equationsr, y0_return, t_return, args=(E, L))

# Extract r and f for the returning side
r_return = sol_return[:, 0]
f_return = sol_return[:, 1]

# Combine the results for the full orbit
r = np.concatenate((r_approach, r_return))
f = np.concatenate((f_approach, f_return))

# Plotting the trajectory in polar coordinates
x = r * np.cos(f)  # Convert polar to Cartesian x-coordinate
y = r * np.sin(f)  # Convert polar to Cartesian y-coordinate

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# Trajectory plot
axs[0].plot(x, y, label='Trajectory')
circle_r0 = plt.Circle((0, 0), r0, color='r', fill=False, label='Circle (r0)')
circle_rp = plt.Circle((0, 0), rp, color='k', fill=False, label='Circle (rp)')
axs[0].add_artist(circle_r0)
axs[0].add_artist(circle_rp)
axs[0].set_title('Trajectory in Schwarzschild Metric')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].legend()
axs[0].grid()
axs[0].axis('equal')  # Ensure equal scaling for x and y axes

# Effective potential plot
r_vals = np.linspace(2, r0, 1000)  # Avoid r=2 to prevent division by zero
v_eff = gr_v_eff(r_vals, E, L)
axs[1].plot(r_vals, v_eff, label='Effective Potential')
axs[1].plot([rp, r0], [E**2 - 1, E**2 - 1], color='r', linestyle='--', label='Energy Level (E^2-1)')
# axs[1].axvline(rp, color='k', linestyle=':', label='Periapsis (rp)')
axs[1].set_title('Effective Potential')
axs[1].set_xlabel('r')
axs[1].set_ylabel('V_eff')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()











# 1) Create a figure
fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 200)
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1.1, 1.1)

# 2) Initialization
def init():
    line.set_data([], [])
    return line,

# 3) Frame update
def update(frame):
    y = np.sin(x + 2*np.pi * frame / 100)
    line.set_data(x, y)
    return line,

# 4) Build the animation
anim = FuncAnimation(fig, update, frames=100, init_func=init, blit=True)

# 5) Save each frame and compile into GIF
filenames = []
for i in range(100):
    update(i)
    fname = f"_frame_{i:03d}.png"
    fig.savefig(fname)
    filenames.append(fname)
plt.close(fig)

# 6) Use imageio to write GIF
with imageio.get_writer('sine_wave.gif', mode='I', duration=0.05) as writer:
    for fname in filenames:
        image = imageio.imread(fname)
        writer.append_data(image)