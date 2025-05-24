import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import mcint as mc
import sys
from matplotlib import cm, ticker
import json
import time
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
import os
from datetime import datetime
from itertools import combinations
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
import traceback
from scipy.stats import linregress
import numpy as np
from sympy import Rational
from scipy.optimize import fsolve
from matplotlib.lines import Line2D

def Eff_pt_gr(r,L):
    return -2/r+((L**2)/(r**2))-2*((L**2)/(r**3))

def Eff_pt_Newt(r,L):
    return -1/r+((L**2)/(2*r**2))



r = np.linspace(0, 10000, 100000)
v1 = Eff_pt_gr(r, np.sqrt(12))
v2 = Eff_pt_gr(r, 4)
v3 = Eff_pt_gr(r, 5)
vNewt1 = Eff_pt_Newt(r, np.sqrt(12))
vNewt2 = Eff_pt_Newt(r, 4)
vNewt3 = Eff_pt_Newt(r, 5)

# Separate figure for L=5 with additional lines and circles
fig1, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(r, v3, '-.k', label='Effective potential')

ax1.set_xlabel('r', fontsize=20)
ax1.set_ylabel('$E^2-1$', fontsize=20)
ax1.grid()
ax1.legend(fontsize=15)
ax1.set_ylim(-1, 0.5)
ax1.set_xlim(0, 100)
ax1.tick_params(axis='both', which='major', labelsize=15)

# Add horizontal line at y = 0.1
def equation_to_solve(r):
    return Eff_pt_gr(r, 5) - 0.1
rtmp = fsolve(equation_to_solve, x0=10)[0]
ax1.hlines(0.1, rtmp, 100, color='green', linestyle='--', linewidth=1.5, label=r'Hyperbolic orbit')
ax1.scatter(rtmp, Eff_pt_gr(rtmp, 5), color='none', edgecolor='green', s=100, zorder=5)

# Add horizontal line at y = 0
def equation_to_solve1(r):
    return Eff_pt_gr(r, 5) - 0.0
rtmp = fsolve(equation_to_solve1, x0=10)[0]
ax1.hlines(0, rtmp, 100, color='gray', linestyle='--', linewidth=2, label=r'Parabolic orbit')
ax1.scatter(rtmp, Eff_pt_gr(rtmp, 5), color='none', edgecolor='gray', s=100, zorder=5)
# ax1.hlines(0, rtmp, 1000, color='gray',linestyle='--', linewidth=1.5)

# Add horizontal line at y = -0.03
def equation_to_solve2(r):
    return Eff_pt_gr(r, 5) + 0.03
rtmp1 = fsolve(equation_to_solve2, x0=50)[0]
rtmp2 = fsolve(equation_to_solve2, x0=11)[0]
ax1.hlines(-0.03, rtmp2, rtmp1, color='purple', linestyle='--', linewidth=1.5, label=r'Eliptical orbit')
ax1.scatter([rtmp1, rtmp2], [Eff_pt_gr(rtmp1, 5), Eff_pt_gr(rtmp2, 5)], color='none', edgecolor='purple', s=100, zorder=5)

# Circle the minimum and maximum points
m1 = np.max(v3[20:300])
m2 = np.min(v3[40:700])
print(m1,m2)
rm1 = np.argwhere(v3 == m1)[0]
rm2 = np.argwhere(v3 == m2)[0]
ax1.scatter(r[rm1], m1, color='none', edgecolor='red', s=100, zorder=5, label=r'Unstable circular orbit')
ax1.scatter(r[rm2], m2, color='none', edgecolor='blue', s=100, zorder=5, label=r'Stable circular orbit')

# Add horizontal line at y = m1+0.03
ax1.hlines(m1+0.03, 0, 100, color='brown', linestyle='--', linewidth=1.5, label=r'Infall into the black hole')
# ax1.scatter(rtmp3, Eff_pt_gr(rtmp3, 5), color='none', edgecolor='red', s=100, zorder=5)

# Add legend
ax1.legend(fontsize=15)
plt.tight_layout()
plt.show()




# Remaining subplots for L=4 and L=sqrt(12)
fig2, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# First subplot (L=4)
axs[0].plot(r, v2, '--b', label='Effective potential,L=4')
axs[0].hlines(0, 0, 1000, color='gray', lw=1.5)
axs[0].set_xlabel('r', fontsize=20)
axs[0].set_ylabel('$E^2-1$', fontsize=20)
axs[0].grid()
axs[0].legend(fontsize=15)
axs[0].set_ylim(-1, 0.5)
axs[0].set_xlim(0, 100)
axs[0].tick_params(axis='both', which='major', labelsize=15)
# Circle the point (4, 0) and add to legend
axs[0].scatter(4, 0, color='none', edgecolor='orange', s=100, zorder=5, label='MBCO')

# Add legend
axs[0].legend(fontsize=15)

# Second subplot (L=sqrt(12))
axs[1].plot(r, v1, '-r', label='Effective potential,$L=\sqrt{12}$')
axs[1].hlines(0, 0, 1000, color='gray', lw=1.5)
axs[1].set_xlabel('r', fontsize=20)
axs[1].set_ylabel('$E^2-1$', fontsize=20)
axs[1].grid()
axs[1].legend(fontsize=15)
axs[1].set_ylim(-1, 0.5)
axs[1].set_xlim(0, 100)
axs[1].tick_params(axis='both', which='major', labelsize=15)
# Circle the point (6, -1/9) and add to legend
axs[1].scatter(6, -1/9, color='none', edgecolor='green', s=100, zorder=5, label='ISCO')

# Add legend
axs[1].legend(fontsize=15)
plt.tight_layout()
plt.show()





fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.plot(r, vNewt3, '-.k', label='Effective potential')

ax2.set_xlabel('r', fontsize=20)
ax2.set_ylabel('$E$', fontsize=20)
ax2.grid()
ax2.legend(fontsize=15)
ax2.set_ylim(-1, 0.5)
ax2.set_xlim(0, 100)
ax2.tick_params(axis='both', which='major', labelsize=15)

# Add horizontal line at y = 0.1
def equation_to_solve(r):
    return Eff_pt_Newt(r, 5) - 0.1
rtmp = fsolve(equation_to_solve, x0=10)[0]
ax2.hlines(0.1, rtmp, 100, color='green', linestyle='--', linewidth=1.5, label=r'Hyperbolic orbit')
ax2.scatter(rtmp, Eff_pt_Newt(rtmp, 5), color='none', edgecolor='green', s=100, zorder=5)

# Add horizontal line at y = 0
def equation_to_solve1(r):
    return Eff_pt_Newt(r, 5) - 0.0
rtmp = fsolve(equation_to_solve1, x0=10)[0]
ax2.hlines(0, rtmp, 100, color='gray', linestyle='--', linewidth=2, label=r'Parabolic orbit')
ax2.scatter(rtmp, Eff_pt_Newt(rtmp, 5), color='none', edgecolor='gray', s=100, zorder=5)
# ax1.hlines(0, rtmp, 1000, color='gray',linestyle='--', linewidth=1.5)

# Add horizontal line at y = -0.01
def equation_to_solve2(r):
    return Eff_pt_Newt(r, 5) + 0.01
rtmp1 = fsolve(equation_to_solve2, x0=50)[0]
rtmp2 = fsolve(equation_to_solve2, x0=11)[0]
ax2.hlines(-0.01, rtmp2, rtmp1, color='purple', linestyle='--', linewidth=1.5, label=r'Eliptical orbit')
ax2.scatter([rtmp1, rtmp2], [Eff_pt_Newt(rtmp1, 5), Eff_pt_Newt(rtmp2, 5)], color='none', edgecolor='purple', s=100, zorder=5)

# Circle the minimum and maximum points
# m1 = np.max(v3[20:300])
m2 = np.min(vNewt3[40:700])
print(m2)
# rm1 = np.argwhere(v3 == m1)[0]
rm2 = np.argwhere(vNewt3 == m2)[0]
# ax1.scatter(r[rm1], m1, color='none', edgecolor='red', s=100, zorder=5, label=r'Unstable circular orbit')
ax2.scatter(r[rm2], m2, color='none', edgecolor='blue', s=100, zorder=5, label=r'Stable circular orbit')

# Add legend
ax2.legend(fontsize=15)
plt.tight_layout()
plt.show()

rlist=np.linspace(2,6,10000)
rlist=rlist[1:-2]
E_pn_list=[]
L_pn_list=[]
fpn_r_list=[]
xpn_list=[]
ypn_list=[]
rlist_list=[]
fdotpn_list=[]
rdotpn_list=[]

L=np.sqrt(12)
E=np.sqrt(8/9)



rdot_gr=((1-(2/rlist))/(E))*((E**2)-(1-(2/rlist))*(1+((L**2)/(rlist**2))))**(1/2)

fdot_gr=(1-(2/rlist))*(L/rlist**2)/(E)

fgr_r=sp.integrate.cumulative_simpson(fdot_gr/rdot_gr,x=rlist,initial=0)



# 1. compute φ(r) on your original rlist:
phi_of_r = fgr_r

# 2. pick a uniform φ-array from φ_min to φ_max:
Nphi     = 20000
phi_min, phi_max = phi_of_r[ 0 ], phi_of_r[-1]
phi_grid = np.linspace(phi_min, phi_max, Nphi)

# 3. invert r(φ) by simple interpolation:
r_of_phi = np.interp(phi_grid, phi_of_r, rlist)

# 4. get cartesian:
x_spiral = r_of_phi * np.cos(phi_grid)
y_spiral = r_of_phi * np.sin(phi_grid)

# 5. and plot as a solid line:
# plt.plot(x_spiral, y_spiral, '-', linewidth=1)
# ygr=rlist*np.sin(fgr_r)

plt.plot(x_spiral, y_spiral,'-',label='GUI',markersize=0.5,color='blue')
plt.xlabel('x',fontsize=20)
plt.ylabel('y',fontsize=20)
# Add a black circle with radius 2
circle = plt.Circle((0, 0), 2, color='black', fill=False, linewidth=1.5, label='Rs')
plt.gca().add_artist(circle)
circle2 = plt.Circle((0, 0), 6, color='red', fill=False, linewidth=5, label='ISCO')
plt.gca().add_artist(circle2)

# Update legend to use circles
legend_elements = [
    Line2D([0], [0], color='black', lw=1.5, label='Rs', marker='o', markersize=10, linestyle='None', fillstyle='none'),
    Line2D([0], [0], color='red', lw=1.5, label='ISCO', marker='o', markersize=10, linestyle='None', fillstyle='none'),
    Line2D([0], [0], color='blue', lw=1.5, label='GUI', marker='o', markersize=10, linestyle='None', fillstyle='none'),
]
plt.legend(handles=legend_elements, fontsize=15)
# plt.title('Effective potential',fontsize=20)   
plt.grid()
# plt.legend(fontsize=15)
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.gca().set_aspect('equal', adjustable='box')
plt.tick_params(axis='both', which='major', labelsize=15)
plt.show()




for L in np.linspace(4.1, 6, 5):
    E = 1

    def equation(r, L, E):
        return (E**2) - (1 - (2 / r)) * (1 + ((L**2) / (r**2)))

    tmpr = fsolve(equation, x0=2 * L - 0.1, args=(L, E))[0]

    rlist = np.linspace(tmpr, 10000, 1000000)
    rlist = rlist[1:-2]

    rdot_gr = ((1 - (2 / rlist)) / E) * ((E**2) - (1 - (2 / rlist)) * (1 + ((L**2) / (rlist**2))))**(1 / 2)
    fdot_gr = (1 - (2 / rlist)) * (L / rlist**2) / E

    fgr_r = sp.integrate.cumulative_simpson(fdot_gr / rdot_gr, x=rlist, initial=0)
    fgr_back = sp.integrate.cumulative_simpson(-fdot_gr / rdot_gr, x=rlist, initial=0)[::-1]

    rlist_total = np.concatenate((rlist, rlist[::-1]))
    phi_of_r = fgr_r
    phi_of_r_back = fgr_back
    phi_of_r_total = np.concatenate((phi_of_r, phi_of_r_back))

    Nphi = 20000
    phi_min, phi_max = phi_of_r[0], phi_of_r[-1]
    phi_grid = np.linspace(phi_min, phi_max, Nphi)
    phi_min_b, phi_max_b = phi_of_r_back[0], phi_of_r_back[-1]
    phi_grid_b = np.linspace(phi_min_b, phi_max_b, Nphi)
    phi_grid_total = np.concatenate((phi_grid, phi_grid_b))

    r_of_phi = np.interp(phi_grid, phi_of_r, rlist)
    r_of_phi_back = np.interp(phi_grid_b, phi_of_r_back, rlist[::-1])
    r_of_phi_total = np.concatenate((r_of_phi, r_of_phi_back))

    x_spiral = r_of_phi * np.cos(phi_grid)
    x_spiral_back = r_of_phi_back * np.cos(phi_grid_b)
    x_spiral_total = np.concatenate((x_spiral, x_spiral_back))
    y_spiral = r_of_phi * np.sin(phi_grid)
    y_spiral_back = r_of_phi_back * np.sin(phi_grid_b)
    y_spiral_total = np.concatenate((y_spiral, y_spiral_back))

    plt.plot(x_spiral_total, y_spiral_total, '-', label=f'L={L:.2f}', markersize=0.5)

plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)

# Add a black circle with radius 2
circle = plt.Circle((0, 0), 4, color='black', fill=False, linewidth=1.5, label='MBCO')
plt.gca().add_artist(circle)


plt.legend(fontsize=15)
plt.grid()
plt.xlim(-150, 150)
plt.ylim(-100, 100)
plt.gca().set_aspect('equal', adjustable='box')
plt.tick_params(axis='both', which='major', labelsize=15)
plt.show()




def test(x,y,M1,M2):
    return -(M2/(((x-1)**2+(y-1)**2)**(1/2)))-(M1/(((x+1)**2+(y+1)**2)**(1/2)))-(1/10)*(x**2+y**2)

#plot the contur line od test for M1=10, M2 =10000
x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(x, y)
Z = test(X, Y, 1, 10)

plt.contour(X, Y, Z,levels=100, colors='black')
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.title('Contour plot of test function', fontsize=20)
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
plt.tick_params(axis='both', which='major', labelsize=15)
plt.show()
