import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import mcint as mc
import sys
from matplotlib import cm, ticker
from Calculate_Potential import *
from Calculate_precession_parab import *
from datetime import datetime
import os
from sympy import Rational
from scipy.interpolate import interp1d

conditions_text_list=[   
    '1) Covergence to the classical Newtonian potential at large distances',
    '2) Reproducing the ISCO at r=6M',
    '3) Reproducing the same angular momentum of the ISCO L=sqrt(12)M',
    '4) Reproducing the photon sphere at r=3M',
    '5) Reproducing the same precession as in GR for rp>>rs',
    '6) Reproducing max point with value 0 for the effective potential at r=4M',
    '7) Reproducing the same angular momentum for marginally bound orbit as in gr L=4M',
    '8) Reproducing the divergence of the precession of an parabolic orbit for L->4M',
    '9) Equating the 3rd derivative of the effective potential at r=4M to the GR value',
    '10) Equating the 4th derivative of the effective potential at r=4M to the GR value',
    '11) Having a minimal point at r=12M for the effective potential',
    '12) Reproducing the minimal point at r=12M  value as the GR one',
    '13)  Equating the 3rd derivative of the effective potential at r=6M to the GR value',
    '14)  Equating the energy of the Isco to the GR value'
]
dark_colors = [
        "#8A07E8", "#009F9F", "#C00089", "#5D9200", "#660101",
        "#556B2F", "#8B4513", "#2E0854", "#3B3B6D", "#5D3954", "#36454F"
    ]
def new_main(N1,rs,conds,rangelist):
    '''
    N1 - number of alphas , integer between 0 to len(conds)
    rs - value of rs, float between 0 to 2
    conds - list of integers between 0 to 13
    range list - list four values, g1,g2,f1,f2, the range of rp values to calculate the precession ,g1,f1 are floats, g2,f2 are integers

    This function calculates the coefficients of the pn potential for the given , Nq,rs,conds combination,
    then it calculates the precession rate of the Gr,Pn,Pw,Pwegg potentials for the given range of rp values.
    finally it plots the resluts for compering the precession rates of the different potentials.
    '''
    save_pn=False
    save_others=False
    conds_list=''.join([str(i) for i in conds])
    save_json_title_pn = f'N1_{N1}_rs_{rs:.2f}_conds_{"".join(map(str, conds))}_range_{rangelist[0]}_{rangelist[1]:.0e}_{rangelist[2]}_{rangelist[3]:.0e}'.replace('+', '_').replace('.', '')
    save_json_title_others=f'Not_pn_data_for_range_{rangelist[0]}_{rangelist[1]:.0e}_{rangelist[2]}_{rangelist[3]:.0e}'.replace('+', '_').replace('.', '')
    # Set folder paths relative to the current directory
    folder_path_plots = r"C:\Users\itama\Desktop\My Projects\Msc_project_2025\Code\Plots"
    folder_path_jsons = r"C:\Users\itama\Desktop\My Projects\Msc_project_2025\Code\Jsons"
    if not os.path.exists(folder_path_plots):
        os.makedirs(folder_path_plots, exist_ok=True)
    if not os.path.exists(folder_path_jsons):
        os.makedirs(folder_path_jsons, exist_ok=True)

    if not os.path.exists(folder_path_jsons + '\\' + save_json_title_pn + '.json'):
        save_pn = True
    if not os.path.exists(folder_path_jsons + '\\' + save_json_title_others + '.json'):
        save_others = True

    if save_pn:
         print('Calculating PN precession for uncalculated values')
         x=Solve_coeffs(N1,rs,conds)[0]
         print(f'####This run for N1={N1},rs={rs},conds={conds} \n The coefficients are {x}####')
         Pn_L_list,Pn_parb_prec_list=calculate_precession_PN(rs,N1,x,rangelist[0],rangelist[1],rangelist[2],rangelist[3])
         data = {
                'Pn_L_list': Pn_L_list,
                'Pn_parb_prec_list': Pn_parb_prec_list,
                'N1': N1,
                'rs': rs,
                'conds': conds,
                'coeffs': x.tolist()
                }
         with open(folder_path_jsons + '\\' + save_json_title_pn + '.json', 'w') as outfile:
            json.dump(data, outfile)
    else:
        print('loading PN precession data')
        with open(folder_path_jsons + '\\' + save_json_title_pn + '.json', 'r') as outfile:
            data = json.load(outfile)
            Pn_L_list = np.array(data['Pn_L_list'])
            Pn_parb_prec_list = np.array(data['Pn_parb_prec_list'])
            x = np.array(data['coeffs'])
    print(f'####This run for N1={N1},rs={rs},conds={conds} \n The coefficients are {x}####')
    if save_others==True and save_pn==True:
        print('Calculating others precession for uncalculated values')
        Gr_L_list,Pw_L_list,Pwegg_L_list,Gr_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,rp1,rp2,rp_wegg1,rp_wegg2=calculate_precession_others(rangelist[0],rangelist[1],rangelist[2],rangelist[3])
        data = {
            'Gr_L_list': Gr_L_list,
            'Pw_L_list': Pw_L_list,
            'Pwegg_L_list': Pwegg_L_list,
            'Gr_parb_prec_list': Gr_parb_prec_list,
            'Pw_parb_prec_list': Pw_parb_prec_list,
            'Pwegg_parb_prec_list': Pwegg_parb_prec_list,
            'rp1': rp1.tolist(),
            'rp2': rp2.tolist(),
            'rp_wegg1': rp_wegg1.tolist(),
            'rp_wegg2': rp_wegg2.tolist(),
        }
        with open(folder_path_jsons + '\\' + save_json_title_others + '.json', 'w') as outfile:
            json.dump(data, outfile)
    else:
        print('loading others precession data')
        with open(folder_path_jsons + '\\' + save_json_title_others + '.json', 'r') as outfile:
            data = json.load(outfile)
            Gr_L_list = np.array(data['Gr_L_list'])
            Pw_L_list = np.array(data['Pw_L_list'])
            Pwegg_L_list = np.array(data['Pwegg_L_list'])
            Gr_parb_prec_list = np.array(data['Gr_parb_prec_list'])
            Pw_parb_prec_list = np.array(data['Pw_parb_prec_list'])
            Pwegg_parb_prec_list = np.array(data['Pwegg_parb_prec_list'])
            rp1 = np.array(data['rp1'])
            rp2 = np.array(data['rp2'])
            rp_wegg1 = np.array(data['rp_wegg1'])
            rp_wegg2 = np.array(data['rp_wegg2'])

    if  save_pn==False and save_others==True:
        print('loading from old jsons the others data')
        with open(folder_path_jsons + '\\' + save_json_title_pn + '.json', 'r') as outfile:
            data = json.load(outfile)
            Gr_L_list = np.array(data['r_L_list'])
            Pw_L_list = np.array(data['Pw_L_list'])
            Pwegg_L_list = np.array(data['Pwegg_L_list'])
            Gr_parb_prec_list = np.array(data['Gr_parb_prec_list'])
            Pw_parb_prec_list = np.array(data['Pw_parb_prec_list'])
            Pwegg_parb_prec_list = np.array(data['Pwegg_parb_prec_list'])
            rp1 = np.array(data['rp1'])
            rp2 = np.array(data['rp2'])
            rp_wegg1 = np.array(data['rp_wegg1'])
            rp_wegg2 = np.array(data['rp_wegg2'])
        data_new = {
            'Gr_L_list': Gr_L_list,
            'Pw_L_list': Pw_L_list,
            'Pwegg_L_list': Pwegg_L_list,
            'Gr_parb_prec_list': Gr_parb_prec_list,
            'Pw_parb_prec_list': Pw_parb_prec_list,
            'Pwegg_parb_prec_list': Pwegg_parb_prec_list,
            'rp1': rp1.tolist(),
            'rp2': rp2.tolist(),
            'rp_wegg1': rp_wegg1.tolist(),
            'rp_wegg2': rp_wegg2.tolist(),
        }
        with open(folder_path_jsons + '\\' + save_json_title_others + '.json', 'w') as outfile:
            json.dump(data_new, outfile)
            
    data_final = {
            'Gr_L_list': np.array(Gr_L_list),
            'Pw_L_list': np.array(Pw_L_list),
            'Pwegg_L_list': np.array(Pwegg_L_list),
            'Gr_parb_prec_list': np.array(Gr_parb_prec_list),
            'Pw_parb_prec_list': np.array(Pw_parb_prec_list),
            'Pwegg_parb_prec_list': np.array(Pwegg_parb_prec_list),
            'rp1': np.array(rp1),
            'rp2': np.array(rp2),
            'rp_wegg1': np.array(rp_wegg1),
            'rp_wegg2': np.array(rp_wegg2),
            'Pn_L_list': np.array(Pn_L_list),
            'Pn_parb_prec_list': np.array(Pn_parb_prec_list),
            'N1': np.array(N1),
            'rs': np.array(rs),
            'conds': np.array(conds),
            'coeffs': np.array(x)
        }
    return data_final

def plot_effective_potential(ax, rlist, N1_list, coefficient_lists, rs_list, L, marksizq, font_size):
    # dark_colors = [
    #     "#8A07E8", "#009F9F", "#C00089", "#5D9200", "#660101",
    #     "#556B2F", "#8B4513", "#2E0854", "#3B3B6D", "#5D3954", "#36454F"
    # ]
    for i in range(len(N1_list)):
        color = dark_colors[i % len(dark_colors)]
        ax.plot(rlist, 2 * u(rlist, N1_list[i], coefficient_lists[i], rs_list[i]) + (L**2 / rlist**2),
                '-*', label=f'PN-N1={N1_list[i]},{i}_pn', color=color, markersize=marksizq, linewidth=0.7 * marksizq)
    ax.plot(rlist, (1 - (2 / rlist)) * (1 + (L**2 / rlist**2)) - 1, 'k-', label='Gr')
    ax.plot(rlist, 2 * u_pw(rlist) + (L**2 / rlist**2), 'g-.', label='Pw')
    ax.plot(rlist, 2 * u_wegg(rlist) + (L**2 / rlist**2), 'r--', label='Pwegg')
    ax.set_ylim(-0.13, 0.025)
    # ax.set_ylim(-0.3, 0.6)
    ax.set_xlim(2, 13)
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.set_xlabel('r', fontsize=font_size + 4)
    ax.set_ylabel(r'$V_{eff}$', fontsize=font_size + 4)
    ax.text(0.05, 0.95, "(I)", transform=ax.transAxes, fontsize=font_size, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))
    ax.legend(fontsize=font_size - 2)

def plot_precession_near_L4(ax, prcession_L_lists, precession_value_lists, gr_l_list, gr_precession_list, pw_l_list, pw_precession_list, wegg_l_list, wegg_precession_list, rangelist, N1_list, marksizq, font_size):
    # dark_colors = [
    #     "#8A07E8", "#009F9F", "#C00089", "#5D9200", "#660101",
    #     "#556B2F", "#8B4513", "#2E0854", "#3B3B6D", "#5D3954", "#36454F"
    # ]
    print(gr_precession_list[:rangelist[1]])
    print(gr_l_list[:rangelist[1]])
    for i in range(len(N1_list)):
        print(precession_value_lists[i][:rangelist[1]])
        print(prcession_L_lists[i][:rangelist[1]])
    for i in range(len(N1_list)):
        color = dark_colors[i % len(dark_colors)]
        ax.plot(prcession_L_lists[i][:rangelist[1]] - prcession_L_lists[i][0],
                np.array(precession_value_lists[i][:rangelist[1]]) / np.pi,
                '-*', label=f'PN-N1={N1_list[i]},{i}_pn', color=color, markersize=marksizq, linewidth=0.7 * marksizq)
    ax.plot(gr_l_list[:rangelist[1]] - gr_l_list[0], np.array(gr_precession_list[:rangelist[1]]) / np.pi, 'k-', label='Gr')
    ax.plot(pw_l_list[:rangelist[1]] - pw_l_list[0], np.array(pw_precession_list[:rangelist[1]]) / np.pi, 'g-.', label='Pw')
    ax.plot(wegg_l_list[:rangelist[1]] - wegg_l_list[0], np.array(wegg_precession_list[:rangelist[1]]) / np.pi, 'r--', label='Pwegg')
    ax.set_xscale('log')
    ax.set_ylim(1, 5)
    ax.set_xlim(10**(-5), 3*10**(-2))
    
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.set_ylabel(r'$\frac{\Delta\phi}{\pi}$', fontsize=font_size + 4, rotation=0, labelpad=20)
    ax.set_xlabel('L-4', fontsize=font_size + 4)
    ax.text(0.05, 0.95, "(II)", transform=ax.transAxes, fontsize=font_size, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))
    ax.legend(fontsize=font_size - 2)

def plot_precession_far_L4(ax, prcession_L_lists, precession_value_lists, gr_l_list, gr_precession_list, pw_l_list, pw_precession_list, wegg_l_list, wegg_precession_list, rangelist, N1_list, marksizq, font_size):
    # dark_colors = [
    #     "#4B0082", "#2F4F4F", "#483D8B", "#191970", "#8B008B",
    #     "#556B2F", "#8B4513", "#2E0854", "#3B3B6D", "#5D3954", "#36454F"
    # ]
    for i in range(len(N1_list)):
        color = dark_colors[i % len(dark_colors)]
        ax.plot(prcession_L_lists[i][rangelist[1]:], np.array(precession_value_lists[i][rangelist[1]:]) / np.pi,
                '-*', label=f'PN-N1={N1_list[i]},{i}_pn', color=color, markersize=marksizq, linewidth=0.7 * marksizq)
    ax.plot(gr_l_list[rangelist[1]:], np.array(gr_precession_list[rangelist[1]:]) / np.pi, 'k-', label='Gr')
    ax.plot(pw_l_list[rangelist[1]:], np.array(pw_precession_list[rangelist[1]:]) / np.pi, 'g-.', label='Pw')
    ax.plot(wegg_l_list[rangelist[1]:], np.array(wegg_precession_list[rangelist[1]:]) / np.pi, 'r--', label='Pwegg')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(3*10**(-4), 10**-2)
    ax.set_xlim(25, 10**2)
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.tick_params(axis='x', which='minor', labelsize=font_size)  # Ensure minor x ticks have the same font size
    # ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel('L', fontsize=font_size + 4)
    ax.set_ylabel(r'$\frac{\Delta\phi}{\pi}$', fontsize=font_size + 4, rotation=0, labelpad=20)
    ax.text(0.05, 0.95, "(III)", transform=ax.transAxes, fontsize=font_size, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))
    ax.legend(fontsize=font_size - 2)

def plot_isco_rdot(ax, rlist_list, rdotpn_list, rdot_gr, rdot_pw, rdot_wegg, rlist, rlist_wegg, r_wegg_isdo, N1_list, font_size):
    # dark_colors = [
    #     "#4B0082", "#2F4F4F", "#483D8B", "#191970", "#8B008B",
    #     "#556B2F", "#8B4513", "#2E0854", "#3B3B6D", "#5D3954", "#36454F"
    # ]
    for i in range(len(N1_list)):
        color = dark_colors[i % len(dark_colors)]
        ax.plot(rlist_list[i], (np.array(rdotpn_list[i]) - np.array(rdot_gr[:len(rdotpn_list[i])])), '-*', label=f'Pn-N1={N1_list[i]},{i}_pn', markersize=0.3, color=color)
    ax.plot(rlist, (np.array(rdot_pw) - (np.array(rdot_gr[:len(rdot_pw)]))), 'g-.', label='Pw')
    ax.plot(rlist_wegg[:len(rdot_gr)] + (6 - r_wegg_isdo), (np.array(rdot_wegg[:len(rdot_gr)]) - (np.array(rdot_gr))), 'r--', label='Pwegg(shifted)')

    ax.legend(markerscale=10, loc='upper right', fontsize=font_size - 6)
    ax.grid()
    ax.set_ylim(-0.01, 0.11)
    ax.set_xlim(4, 6)
    print(rlist)
    # Ensure all arrays are truncated to the length of rdot_gr for fair comparison
    min_len = min(len(rdot_gr), len(rdot_pw), len(rdot_wegg), *[len(rdotpn) for rdotpn in rdotpn_list])
    print('min_len=', min_len)
    initial_index=np.where(rlist >= 5)[0][0]
    initial_index_wegg=np.where(rlist_wegg >= r_wegg_isdo-1)[0][0]
    print('max_diff_pw', ((rdot_pw[initial_index]- rdot_gr[initial_index])/np.abs(rdot_gr[initial_index]))*100,'%')
    print('max_diff_wegg', ((rdot_wegg[initial_index_wegg]- rdot_gr[initial_index_wegg])/np.abs(rdot_gr[initial_index_wegg]))*100,'%')
    for i in range(len(rdotpn_list)):
        initial_index=np.where(rlist_list[i] >= 5)[0][0]
        print(f'max_diff_pn_{i}', ((rdotpn_list[i][initial_index]- rdot_gr[initial_index])/np.abs(rdot_gr[initial_index]))*100,'%')
    ax.set_xlabel('r', fontsize=font_size)
    ax.set_ylabel(r"$\Delta \dot{r}$ ", fontsize=font_size, labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=font_size - 2)
    ax.text(0.05, 0.95, "(a)", transform=ax.transAxes, fontsize=font_size, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

def plot_isco_veff(ax, rlist_isco, N1_list, coefficient_lists, rs_list, Lweggisco, veff_gr_isco, veff_pw_isco, veff_wegg_isco, font_size):
    # dark_colors = [
    #     "#4B0082", "#2F4F4F", "#483D8B", "#191970", "#8B008B",
    #     "#556B2F", "#8B4513", "#2E0854", "#3B3B6D", "#5D3954", "#36454F"
    # ]
    for i in range(len(N1_list)):
        color = dark_colors[i % len(dark_colors)]
        L_pn_isco = np.sqrt((6**3) * u_dr(6, N1_list[i], coefficient_lists[i], rs_list[i]))
        print(f"L/sqrt(12) for run {i} is",L_pn_isco/np.sqrt(12))
        print(f"E/(-1/9) for run {i} is",(2*u(6, N1_list[i], coefficient_lists[i], rs_list[i])+(((L_pn_isco**2))/(36)))/(-1/9))
        veff_pn_isco = 2 * u(rlist_isco, N1_list[i], coefficient_lists[i], rs_list[i]) + ((L_pn_isco**2) / (rlist_isco**2))
        ax.plot(rlist_isco, veff_pn_isco, '-*', label=f'PN-N1={N1_list[i]},{i}_pn', color=color, markersize=0.3, linewidth=0.7)
    ax.plot(rlist_isco, veff_wegg_isco, 'r--', label='Pwegg', markersize=0.3, linewidth=0.7)
    ax.plot(rlist_isco, veff_gr_isco, 'k-', label='Gr')
    ax.plot(rlist_isco, veff_pw_isco, 'g-.', label='Pw')
    ax.set_xlim(2, 10)
    ax.set_ylim(-0.2, -0.07)
    # ax.set_ylim(-0.5, 0.5)
    ax.axvline(x=6, color='gray', linestyle=':', linewidth=3)
    ax.text(6.1, -0.1, 'r=6', fontsize=font_size - 2, color='gray', verticalalignment='center')
    ax.legend(markerscale=1, loc='upper right', fontsize=font_size - 6)
    ax.grid()
    ax.set_xlabel('r', fontsize=font_size)
    ax.set_ylabel(r"$V_{eff}$", fontsize=font_size, labelpad=20)
    ax.legend(markerscale=1, loc='lower right', fontsize=font_size - 6)
    ax.tick_params(axis='both', which='major', labelsize=font_size - 2)
    ax.text(0.05, 0.95, "(b)", transform=ax.transAxes, fontsize=font_size, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

def save_figure(fig, default_name, plots_dir):
    save_path = os.path.join(plots_dir, default_name)
    print(f"Default save path for {default_name}: {save_path}")
    save_choice = input(f"Save {default_name} as default name? (y/n): ").strip().lower()
    if save_choice == 'n':
        new_name = input(f"Enter new filename for {default_name} (with .png): ").strip()
        if not new_name.endswith('.png'):
            new_name += '.png'
        save_path = os.path.join(plots_dir, new_name)
    fig.savefig(save_path, transparent=True)
    plt.show()

def final_plots(rangelist, auto=False, N1_list=[], rs_list=[], conds_list=[]):
    font_size = 16
    marksizq = 1
    # Set folder path to the directory where this script is located
    folder_path_plots = r"C:\Users\itama\Desktop\My Projects\Msc_project_2025\Code"
    if not os.path.exists(folder_path_plots):
        os.makedirs(folder_path_plots, exist_ok=True)
    # Ask user for subplot orientation
    orientation = input("Choose subplot orientation for Figure 1 (h for horizontal, v for vertical): ").strip().lower()
    if orientation == 'h':
        fig1_shape = (1, 3)
        fig1_size = (18, 6)
    else:
        fig1_shape = (3, 1)
        fig1_size = (6, 18)
    orientation2 = input("Choose subplot orientation for Figure 2 (h for horizontal, v for vertical): ").strip().lower()
    if orientation2 == 'h':
        fig2_shape = (1, 2)
        fig2_size = (14, 7)
    else:
        fig2_shape = (2, 1)
        fig2_size = (10, 14)
    if not auto:
        N1_list = []
        rs_list = []
        conds_list = []
        while True:
            tr = input("Enter 'N1,rs,conds (space separated)' or 'stop' to continue plot:")
            if tr == 'stop':
                break
            else:
                l = tr.split(',')
                N1_list.append(int(l[0]))
                rs_list.append(float(l[1]))
                conds_list.append([int(i) for i in l[2].split()])
    coefficient_lists = []
    prcession_L_lists = []
    precession_value_lists = []
    precession_rp1_lists = []
    precession_rp2_lists = []
    for i in range(len(N1_list)):
        data_tmp = new_main(N1_list[i], rs_list[i], conds_list[i], rangelist)
        coefficient_lists.append(np.array(data_tmp['coeffs']))
        prcession_L_lists.append(np.array(data_tmp['Pn_L_list']))
        precession_value_lists.append(np.array(data_tmp['Pn_parb_prec_list']))
        precession_rp1_lists.append(np.array(data_tmp['rp1']))
        precession_rp2_lists.append(np.array(data_tmp['rp2']))
    gr_l_list = np.array(data_tmp['Gr_L_list'])
    gr_precession_list = np.array(data_tmp['Gr_parb_prec_list'])
    pw_l_list = np.array(data_tmp['Pw_L_list'])
    pw_precession_list = np.array(data_tmp['Pw_parb_prec_list'])
    wegg_l_list = np.array(data_tmp['Pwegg_L_list'])
    wegg_precession_list = np.array(data_tmp['Pwegg_parb_prec_list'])
    rp1_wegg = np.array(data_tmp['rp_wegg1'])
    rp2_wegg = np.array(data_tmp['rp_wegg2'])
    rlist = np.linspace(2, 13, 1000)
    fig, axs = plt.subplots(*fig1_shape, figsize=fig1_size, constrained_layout=True)
    # Make axs always a 1D array for consistent indexing
    axs = np.ravel(axs)
    plot_effective_potential(axs[0], rlist, N1_list, coefficient_lists, rs_list, 4, marksizq, font_size)
    plot_precession_near_L4(axs[1], prcession_L_lists, precession_value_lists, gr_l_list, gr_precession_list, pw_l_list, pw_precession_list, wegg_l_list, wegg_precession_list, rangelist, N1_list, marksizq, font_size)
    plot_precession_far_L4(axs[2], prcession_L_lists, precession_value_lists, gr_l_list, gr_precession_list, pw_l_list, pw_precession_list, wegg_l_list, wegg_precession_list, rangelist, N1_list, marksizq, font_size)
    msc_project_dir = os.path.join(os.getcwd(), "Msc_project_2025")
    plots_dir = folder_path_plots + r"\Plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir, exist_ok=True)
    # Ask if user wants to save
    save_choice = input("Do you want to save Figure 1? (y/n): ").strip().lower()
    if save_choice == 'y':
        save_figure(fig, "Figure1_paper.png", plots_dir)
    else:
        plt.show()
    # ISCO plots
    rlist = np.linspace(2, 6, 1000000)[1:-2]
    E_pn_list = []
    L_pn_list = []
    fpn_r_list = []
    xpn_list = []
    ypn_list = []
    rlist_list = []
    fdotpn_list = []
    rdotpn_list = []
    for i, j in enumerate(coefficient_lists):
        L_pn_list.append(np.sqrt((6**3) * (u_dr(6, N1_list[i], j, rs_list[i]))))
        E_pn_list.append(u(6, N1_list[i], j, rs_list[i]) + ((6**3) * u_dr(6, N1_list[i], j, rs_list[i]) / (2 * (6**2))))
    Lgrisco = np.sqrt(12)
    Lpwisco = np.sqrt((6**3) * (u_pw_dr(6)))
    r_wegg_isdo = 4.6784623564771834
    Lweggisco = np.sqrt((r_wegg_isdo**3) * (u_wegg_dr(r_wegg_isdo)))
    Egrisco = np.sqrt(8 / 9)
    Epwisco = u_pw(6) + ((Lpwisco**2) / (2 * (6**2)))
    Eweggisco = u_wegg(r_wegg_isdo) + ((Lweggisco**2) / (2 * (r_wegg_isdo**2)))
    for i in range(len(N1_list)):
        rdot_pn = np.sqrt(2 * E_pn_list[i] - 2 * u(rlist, N1_list[i], coefficient_lists[i], rs_list[i]) - ((L_pn_list[i]**2) / (rlist**2)))
        valid_indices = np.where((~np.isnan(rdot_pn)) & (np.isreal(rdot_pn)))[0]
        if len(valid_indices) > 0:
            rlist_valid = rlist[valid_indices]
            rlist_list.append(rlist_valid)
            rdot_pn = rdot_pn[valid_indices]
            fdot_pn = (L_pn_list[i] / (rlist_valid**2))
        else:
            rdot_pn = np.zeros(len(rlist))
            fdot_pn = np.zeros(len(rlist))
            rlist_valid = rlist
            rlist_list.append(rlist_valid)
        fpn_r = sp.integrate.cumulative_simpson(fdot_pn / rdot_pn, x=rlist_valid, initial=0)
        xpn = rlist_valid * np.cos(fpn_r)
        ypn = rlist_valid * np.sin(fpn_r)
        xpn_list.append(xpn)
        ypn_list.append(ypn)
        fpn_r_list.append(fpn_r)
        fdotpn_list.append(fdot_pn)
        rdotpn_list.append(rdot_pn)
    rdot_gr = ((1 - (2 / rlist)) / (Egrisco)) * ((Egrisco**2) - (1 - (2 / rlist)) * (1 + ((Lgrisco**2) / (rlist**2))))**(1 / 2)
    rdot_pw = np.sqrt(2 * Epwisco - 2 * u_pw(rlist) - ((Lpwisco**2) / (rlist**2)))
    rlist_wegg = np.linspace(2, r_wegg_isdo, 1000000)
    rdot_wegg = np.sqrt(2 * Eweggisco - 2 * u_wegg(rlist_wegg) - ((Lweggisco**2) / (rlist_wegg**2)))
    fig2, axs2 = plt.subplots(*fig2_shape, figsize=fig2_size, constrained_layout=True)
    axs2 = np.ravel(axs2)
    font_size2 = 23
    plot_isco_rdot(axs2[0], rlist_list, rdotpn_list, rdot_gr, rdot_pw, rdot_wegg, rlist, rlist_wegg, r_wegg_isdo, N1_list, font_size2)
    rlist_isco = np.linspace(2, 10, 100000)[1:-2]
    Lgrisco = np.sqrt(12)
    veff_gr_isco = (1 - (2 / rlist_isco)) * (1 + ((Lgrisco**2) / (rlist_isco**2))) - 1
    Lpwisco = np.sqrt((6**3) * u_pw_dr(6))
    veff_pw_isco = 2 * u_pw(rlist_isco) + ((Lpwisco**2) / (rlist_isco**2))
    veff_wegg_isco = 2 * u_wegg(rlist_isco) + ((Lweggisco**2) / (rlist_isco**2))
    plot_isco_veff(axs2[1], rlist_isco, N1_list, coefficient_lists, rs_list, Lweggisco, veff_gr_isco, veff_pw_isco, veff_wegg_isco, font_size2)
    # Ask if user wants to save
    save_choice2 = input("Do you want to save Figure 2? (y/n): ").strip().lower()
    if save_choice2 == 'y':
        save_figure(fig2, "Figure2_paper.png", plots_dir)
    else:
        plt.show()
    return


rangelist = [0.8, 5000, 7010, 7000]
# final_plots(rangelist,auto=True,N1_list=[1,7],rs_list=[2,2],conds_list=[[0,1,4,5,6,7,8,12],[0,1,4,5,6,7,8,12]])
# final_plots(rangelist,auto=True,N1_list=[1,8],rs_list=[2,2],conds_list=[[0,1,2,4,5,7,8,12,13],[0,1,2,4,5,7,8,12,13]])
# final_plots(rangelist,auto=True,N1_list=[1,7],rs_list=[2,2],conds_list=[[0,1,4,5,6,7,8,12],[0,1,4,5,6,7,8,12]])

# Calculate coefficients for the specified cases
def compare_effective_potentials(N1_list, rs_list, conds_list, L=4, marksizq=2, font_size=14):
    """
    Compare effective potentials for given N1, rs, and conds lists.
    Plots for L and for ISCO.
    """
    coefficient_lists = []
    for i in range(len(N1_list)):
        coeffs = Solve_coeffs(N1_list[i], rs_list[i], conds_list[i])[0]
        coefficient_lists.append(np.array(coeffs))

    # Plot only the effective potentials for L and ISCO
    rlist = np.linspace(2, 13, 1000)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # Plot for L
    plot_effective_potential(axs[0], rlist, N1_list, coefficient_lists, rs_list, L, marksizq=marksizq, font_size=font_size)
    axs[0].set_title(f"Effective Potential for L={L}", fontsize=font_size+2)

    # Plot for ISCO
    rlist_isco = np.linspace(2, 10, 1000)
    Lgrisco = np.sqrt(12)
    Lpwisco = np.sqrt((6**3) * u_pw_dr(6))
    r_wegg_isdo = 4.6784623564771834
    Lweggisco = np.sqrt((r_wegg_isdo**3) * (u_wegg_dr(r_wegg_isdo)))

    veff_gr_isco = (1 - (2 / rlist_isco)) * (1 + ((Lgrisco**2) / (rlist_isco**2))) - 1
    veff_pw_isco = 2 * u_pw(rlist_isco) + ((Lpwisco**2) / (rlist_isco**2))
    veff_wegg_isco = 2 * u_wegg(rlist_isco) + ((Lweggisco**2) / (rlist_isco**2))

    plot_isco_veff(axs[1], rlist_isco, N1_list, coefficient_lists, rs_list, Lweggisco, veff_gr_isco, veff_pw_isco, veff_wegg_isco, font_size=font_size)
    axs[1].set_title("Effective Potential at ISCO", fontsize=font_size+2)

    plt.show()

# Example usage:
# N1_list = [1, 9]
# rs_list = [1.5, 1.5]
# conds_list = [
#     [0, 1, 2, 4, 5, 6, 7, 8, 12, 13],
#     [0, 1, 2, 4, 5, 6, 7, 8, 12, 13]
# ]
# compare_effective_potentials(N1_list, rs_list, conds_list)
def plot_all_potentials_semilogy(N1_list, rs_list, conds_list, r_min=0.01, r_max=70, num_points=1000):
    """
    Plots all potentials (PN, PW, Pwegg, Newtonian) on a semilog-y plot.
    PN is computed for each (N1, rs, conds) in the input lists.
    Each potential is plotted only up to its divergence point.
    """
    # Ensure input lists are the same length
    if not (len(N1_list) == len(rs_list) == len(conds_list)):
        raise ValueError("N1_list, rs_list, and conds_list must have the same length.")

    # Newtonian: diverges at r=0
    rlist_newton = np.linspace(r_min, r_max, num_points)
    v_newton = -1 / rlist_newton

    # PW: diverges at r=0
    rlist_pw = np.linspace(2 + r_min, r_max, num_points)
    v_pw = u_pw(rlist_pw)

    # Pwegg: diverges at r=2
    rlist_pwegg = np.linspace((4 * np.sqrt(6)) - 9 + r_min, r_max, num_points)
    v_pwegg = u_wegg(rlist_pwegg)

    plt.figure(figsize=(8, 6))
    plt.semilogy(rlist_newton, np.abs(v_newton), label="Newtonian", color="black", linestyle="--")
    plt.semilogy(rlist_pw, np.abs(v_pw), label="PW", color="green")
    plt.semilogy(rlist_pwegg, np.abs(v_pwegg), label="Pwegg", color="red")

    for i, (N1, rs, conds) in enumerate(zip(N1_list, rs_list, conds_list)):
        coeffs = Solve_coeffs(N1, rs, conds)[0]
        rlist_pn = np.linspace(rs + r_min, r_max, num_points)
        v_pn = u(rlist_pn, N1, coeffs, rs)
        color = dark_colors[i % len(dark_colors)]
        plt.semilogy(rlist_pn, np.abs(v_pn), label=f"PN (N1={N1}, rs={rs})", color=color)

    plt.xlabel("r", fontsize=14)
    plt.ylabel(r"$-\Phi(r)$", fontsize=14)
    plt.title("Comparison of Potentials (semilog-y)", fontsize=16)
    plt.xlim(r_min, r_max)
    plt.ylim(1e-2, 10)
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=12)
    plt.gcf().patch.set_alpha(0)  # Make figure background transparent
    ax = plt.gca()
    ax.set_facecolor('none')      # Make subplot (axes) background transparent
    plt.show()

# plot_all_potentials_semilogy([7,1], [2,2], [[0, 1, 4, 5, 6, 7, 8, 12],[0, 1, 4, 5, 6, 7, 8, 12]])


def plot_precession_diff_near_L4(ax, prcession_L_lists, precession_value_lists, gr_l_list, gr_precession_list, pw_l_list, pw_precession_list, wegg_l_list, wegg_precession_list, rangelist, N1_list, marksizq, font_size):
    # Interpolate all precession curves to a common L grid for logical subtraction
    # Use the union of all L values in the near-L4 region
    L_gr = gr_l_list[:rangelist[1]]
    L_pw = pw_l_list[:rangelist[1]]
    L_wegg = wegg_l_list[:rangelist[1]]
    L_pn_list = [prc_L[:rangelist[1]] for prc_L in prcession_L_lists]
    # Build a common L grid (sorted unique values)
    L_common = np.unique(np.concatenate([L_gr] + L_pn_list + [L_pw, L_wegg]))
    # Interpolators
    interp_gr = interp1d(L_gr, gr_precession_list[:rangelist[1]], kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_pw = interp1d(L_pw, pw_precession_list[:rangelist[1]], kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_wegg = interp1d(L_wegg, wegg_precession_list[:rangelist[1]], kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_pn_list = [interp1d(L_pn, precession_value_lists[i][:rangelist[1]], kind='linear', bounds_error=False, fill_value="extrapolate") for i, L_pn in enumerate(L_pn_list)]
    # Plot PN - GR
    for i, interp_pn in enumerate(interp_pn_list):
        color = dark_colors[i % len(dark_colors)]
        diff = (interp_pn(L_common) - interp_gr(L_common)) / np.pi
        print(diff[50:])
        ax.plot(L_common - L_common[0], diff, '-*', label=f'PN-N1={N1_list[i]},{i}_pn', color=color, markersize=marksizq, linewidth=0.7 * marksizq)

    # PW - GR
    diff_pw = (interp_pw(L_common) - interp_gr(L_common)) / np.pi
    print(diff_pw[50:])
    ax.plot(L_common - L_common[0], diff_pw, 'g-.', label='Pw')
    # Pwegg - GR
    diff_wegg = (interp_wegg(L_common) - interp_gr(L_common)) / np.pi
    print(diff_wegg[50:])
    ax.plot(L_common - L_common[0], diff_wegg, 'r--', label='Pwegg')
    ax.set_xscale('log')
    ax.set_ylim(-1, 1)
    ax.set_xlim(10**(-5), 3*10**(-2))
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.set_ylabel(r'$\frac{\Delta\phi_{X} - \Delta\phi_{GR}}{\pi}$', fontsize=font_size + 4, rotation=0, labelpad=20)
    ax.set_xlabel('L-4', fontsize=font_size + 4)
    ax.set_title("Precession Difference Near L=4", fontsize=font_size + 2)
    ax.text(0.05, 0.95, "(II)", transform=ax.transAxes, fontsize=font_size, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))
    ax.legend(fontsize=font_size - 2)
    return interp_gr, interp_pw, interp_wegg, interp_pn_list

def plot_precession_diff_far_L4(ax, prcession_L_lists, precession_value_lists, gr_l_list, gr_precession_list, pw_l_list, pw_precession_list, wegg_l_list, wegg_precession_list, rangelist, N1_list, marksizq, font_size):
    # Interpolate all precession curves to a common L grid for logical subtraction (far from L=4)
    L_gr = gr_l_list[rangelist[1]:]
    L_pw = pw_l_list[rangelist[1]:]
    L_wegg = wegg_l_list[rangelist[1]:]
    L_pn_list = [prc_L[rangelist[1]:] for prc_L in prcession_L_lists]
    # Build a common L grid (sorted unique values)
    L_common = np.unique(np.concatenate([L_gr] + L_pn_list + [L_pw, L_wegg]))
    # Interpolators
    interp_gr = interp1d(L_gr, gr_precession_list[rangelist[1]:], kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_pw = interp1d(L_pw, pw_precession_list[rangelist[1]:], kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_wegg = interp1d(L_wegg, wegg_precession_list[rangelist[1]:], kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_pn_list = [interp1d(L_pn, precession_value_lists[i][rangelist[1]:], kind='linear', bounds_error=False, fill_value="extrapolate") for i, L_pn in enumerate(L_pn_list)]
    # Plot PN - GR
    for i, interp_pn in enumerate(interp_pn_list):
        color = dark_colors[i % len(dark_colors)]
        diff = (interp_pn(L_common) - interp_gr(L_common)) / np.pi
        ax.plot(L_common, diff, '-*', label=f'PN-N1={N1_list[i]},{i}_pn', color=color, markersize=marksizq, linewidth=0.7 * marksizq)
    # PW - GR
    diff_pw = (interp_pw(L_common) - interp_gr(L_common)) / np.pi
    ax.plot(L_common, diff_pw, 'g-.', label='Pw')
    # Pwegg - GR
    diff_wegg = (interp_wegg(L_common) - interp_gr(L_common)) / np.pi
    ax.plot(L_common, diff_wegg, 'r--', label='Pwegg')
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_ylim(-0.01, 0.01)
    ax.set_xlim(25, 10**2)
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.tick_params(axis='x', which='minor', labelsize=font_size)
    ax.set_xlabel('L', fontsize=font_size + 4)
    ax.set_ylabel(r'$\frac{\Delta\phi_{X} - \Delta\phi_{GR}}{\pi}$', fontsize=font_size + 4, rotation=0, labelpad=20)
    ax.set_title("Precession Difference Far from L=4", fontsize=font_size + 2)
    ax.legend(fontsize=font_size - 2)
    ax.text(0.05, 0.95, "(III)", transform=ax.transAxes, fontsize=font_size, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))
    return interp_gr, interp_pw, interp_wegg, interp_pn_list
    # Plot the precession differences for the specified rangelist and parameters
def plot_precession_diffs_for_cases(rangelist, N1_list, rs_list, conds_list, marksizq=1, font_size=16):
    """
    Plot precession differences for given cases, similar to final_plots input.

    Args:
        rangelist: List of 4 values specifying the rp range.
        N1_list: List of N1 values.
        rs_list: List of rs values.
        conds_list: List of conds lists.
        marksizq: Marker size for plots.
        font_size: Font size for plots.
    """
    # Gather data for each case
    coefficient_lists = []
    prcession_L_lists = []
    precession_value_lists = []
    for i in range(len(N1_list)):
        data_tmp = new_main(N1_list[i], rs_list[i], conds_list[i], rangelist)
        coefficient_lists.append(np.array(data_tmp['coeffs']))
        prcession_L_lists.append(np.array(data_tmp['Pn_L_list']))
        precession_value_lists.append(np.array(data_tmp['Pn_parb_prec_list']))
    gr_l_list = np.array(data_tmp['Gr_L_list'])
    gr_precession_list = np.array(data_tmp['Gr_parb_prec_list'])
    pw_l_list = np.array(data_tmp['Pw_L_list'])
    pw_precession_list = np.array(data_tmp['Pw_parb_prec_list'])
    wegg_l_list = np.array(data_tmp['Pwegg_L_list'])
    wegg_precession_list = np.array(data_tmp['Pwegg_parb_prec_list'])

    fig, axs = plt.subplots(1, 2, figsize=(20, 6), constrained_layout=True)
    inter_gr_near, inter_pw_near, inter_wegg_near, inter_pn_near_list = plot_precession_diff_near_L4(
        axs[0], prcession_L_lists, precession_value_lists,
        gr_l_list, gr_precession_list,
        pw_l_list, pw_precession_list,
        wegg_l_list, wegg_precession_list,
        rangelist, N1_list, marksizq, font_size
    )
    inter_gr_far, inter_pw_far, inter_wegg_far, inter_pn_far_list = plot_precession_diff_far_L4(
        axs[1], prcession_L_lists, precession_value_lists,
        gr_l_list, gr_precession_list,
        pw_l_list, pw_precession_list,
        wegg_l_list, wegg_precession_list,
        rangelist, N1_list, marksizq, font_size
    )
    # Plot all precessions for all L values (logscale x)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for i in range(len(N1_list)):
        color = dark_colors[i % len(dark_colors)]
        precession_vals = np.array(precession_value_lists[i])
        precession_vals[precession_vals <= 0] = 0.001
        ax2.plot(prcession_L_lists[i], np.array(precession_vals) / np.pi,
            '-*', label=f'PN-N1={N1_list[i]},{i}_pn', color=color, markersize=marksizq, linewidth=0.7 * marksizq)
    ax2.plot(gr_l_list, np.array(gr_precession_list) / np.pi, 'k-', label='Gr')
    ax2.plot(pw_l_list, np.array(pw_precession_list) / np.pi, 'g-.', label='Pw')
    ax2.plot(wegg_l_list, np.array(wegg_precession_list) / np.pi, 'r--', label='Pwegg')
    ax2.set_xscale('log')
    ax2.set_xlabel('L', fontsize=font_size + 4)
    ax2.set_ylabel(r'$\frac{\Delta\phi}{\pi}$', fontsize=font_size + 4, rotation=0, labelpad=20)
    ax2.set_title("All Precessions vs L (logscale)", fontsize=font_size + 2)
    ax2.legend(fontsize=font_size - 2)
    ax2.grid(True, which="both", ls="--")
    plt.show()

# Example usage:
rangelist = [0.8, 5000, 7010, 7000]
N1_list = [1, 7]
rs_list = [2, 2]
conds_list = [
    [0, 1, 4, 5, 6, 7, 8, 12,13],
    [0, 1, 4, 5, 6, 7, 8, 12,13]
]
plot_precession_diffs_for_cases(rangelist, N1_list, rs_list, conds_list)
# Call the function to plot the diffs

    