import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import mcint as mc
import sys
from matplotlib import cm, ticker
sys.path.append('Scripts/Research scripts/General Functions')
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


def plot_effctive_potential(rs,N1,x,type='MSCO'):
    if type=='MSCO':    
        Rp=4
        Rp_wegg=2*(np.sqrt(6)-1)
        E=0
        Lpn=np.sqrt(2*(Rp**2)*(E-u(Rp,N1,x,rs)))
        print(Lpn)
        Lgr1=np.sqrt((((E+1)**2)*(Rp**2)-(Rp**2)+2*Rp)/(1-(2/Rp)))
        Lpw=np.sqrt(2*(Rp**2)*(E-u_pw(Rp)))
        Lwegg=np.sqrt(2*(Rp_wegg**2)*(E-u_wegg(Rp_wegg)))
        data_for_save_1=f'Lpn={Lpn}, Lgr={Lgr1}, Lpw={Lpw}, Lwegg={Lwegg} \n marginally bound in gr={Rp} pn={Rp} pw={Rp} wegg={Rp_wegg}'

        rlist=np.linspace(Rp-2.5,100,100000)
        plt.plot(rlist, [2*u(r,N1,x,rs) + ((Lpn**2)/(r**2)) for r in rlist], label='Pn')
        plt.plot(rlist, [(1-(2/r))*(1+((Lgr1**2)/(r**2)))-1 for r in rlist], '--', label='Gr,rp=4')
        plt.plot(rlist, [2*u_pw(r) + ((Lpw**2)/(r**2)) for r in rlist], label='Pw')
        plt.plot(rlist, [2*u_wegg(r) + ((Lwegg**2)/((r)**2)) for r in rlist], label=f'Pwegg moved by {abs(Rp_wegg-Rp):.2f}')

        plt.title('Effective potential parbolic Rp=4')
        plt.grid()
        plt.xlim(0,50)
        plt.ylim(-0.8,0.8)
        return data_for_save_1
    if type=='ISCO':
        rlist=np.linspace(2,10,100000)
        rlist=rlist[1:-2]
        Lgrisco=np.sqrt(12)
        Lpnisco=np.sqrt((6**3)*u_dr(6,N1,x,rs))
        Lpwisco=np.sqrt((6**3)*u_pw_dr(6))
        Lweggisco=np.sqrt((6**3)*u_wegg_dr(6))
        veff_gr=(1-(2/rlist))*(1+((Lgrisco**2)/(rlist**2)))-1
        veff_pn=2*u(rlist,N1,x,rs)+((Lpnisco**2)/(rlist**2))
        veff_pw=2*u_pw(rlist)+((Lpwisco**2)/(rlist**2))
        veff_wegg=2*u_wegg(rlist)+((Lweggisco**2)/(rlist**2))

        plt.plot(rlist, veff_gr,'-', label='Gr')
        plt.plot(rlist, veff_pn,'--', label='Pn')
        plt.plot(rlist, veff_pw,'-.', label='Pw')
        plt.plot(rlist, veff_wegg,':', label='Pwegg')
        plt.legend()
        plt.grid()
        plt.title('Effective potential for ISCO')
        plt.xlim(2,10)
        plt.ylim(-1,0.5)
        return Lgrisco,Lpnisco,Lpwisco,Lweggisco
def plot_ISCO_trajectory(rs,N1,x,L_list):
    rlist=np.linspace(2,6,100000)
    Lgrisco=L_list[0]
    Lpnisco=L_list[1]
    Lpwisco=L_list[2]
    Lweggisco=L_list[3] 

    rlist=rlist[1:-2]
    Egrisco=np.sqrt(8/9)
    Episco=u(6,N1,x,rs)+((6**3)*u_dr(6,N1,x,rs)/(2*(6**2)))
    Epwisco=u_pw(6)+((Lpwisco**2)/(2*(6**2)))
    Eweggisco=u_wegg(6)+((Lweggisco**2)/(2*(6**2)))
    
    rdot_gr=((1-(2/rlist))/(Egrisco))*((Egrisco**2)-(1-(2/rlist))*(1+((Lgrisco**2)/(rlist**2))))**(1/2)
    rdot_pn=np.sqrt(2*Episco-2*u(rlist,N1,x,rs)-(Lpnisco**2)/(rlist**2))
    rdot_pw=np.sqrt(2*Epwisco-2*u_pw(rlist)-((Lpwisco**2)/(rlist**2)))
    # rdot_wegg=np.sqrt(2*Eweggisco-2*u_wegg(rlist)-((Lweggisco**2)/(rlist**2)))
    fdot_gr=(1-(2/rlist))*(Lgrisco/rlist**2)/(Egrisco)
    fdot_pn=(Lpnisco/(rlist**2))
    fdot_pw=(Lpwisco/(rlist**2))
    # fdot_wegg=(Lweggisco/(rlist**2))
    data_for_save_2=f'Lgrisco={Lgrisco}, Egrisco={Egrisco}, Lpnisco={Lpnisco}, Episco={Episco}, Lpwisco={Lpwisco}, Epwisco={Epwisco}, Lweggisco={Lweggisco}, Eweggisco={Eweggisco} \n for r=6'
    # print(data_for_save_2)
    # print((6**3)*u_dr(6))

    fgr_r=sp.integrate.cumulative_simpson(fdot_gr/rdot_gr,x=rlist,initial=0)
    # fgr_r=fgr_r-fgr_r[-1]
    fpn_r=sp.integrate.cumulative_simpson(fdot_pn/rdot_pn,x=rlist,initial=0)
    # fpn_r=fpn_r-fpn_r[-1]
    fpw_r=sp.integrate.cumulative_simpson(fdot_pw/rdot_pw,x=rlist,initial=0)
    # fpw_r=fpw_r-fpw_r[-1]
    # fwegg_r=sp.integrate.cumulative_simpson(fdot_wegg/rdot_wegg,x=rlist,initial=0)
    # fwegg_r=fwegg_r-fwegg_r[-1]

    xgr=rlist*np.cos(fgr_r)
    ygr=rlist*np.sin(fgr_r)
    xpn=rlist*np.cos(fpn_r)
    ypn=rlist*np.sin(fpn_r)
    xpw=rlist*np.cos(fpw_r)
    ypw=rlist*np.sin(fpw_r)
    # xwegg=rlist*np.cos(fwegg_r)
    # ywegg=rlist*np.sin(fwegg_r)
    ax=plt.subplot(1,2,1)
    ax.plot(xgr,ygr,'.',label='Gr',markersize=0.1)
    ax.plot(xpn,ypn,'*',label='Pn',markersize=0.5)
    ax.plot(xpw,ypw,'o',label='Pw',markersize=0.3)
    # ax.plot(xwegg,ywegg,'s',label='Pwegg',markersize=0.1)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid()
    ax.set_title('Trajectories for ISCO')
    ax=plt.subplot(1,2,2)
    ax.plot(rlist,fgr_r,label='Gr')
    ax.plot(rlist,fpn_r,label='Pn')
    ax.plot(rlist,fpw_r,label='Pw')
    # ax.plot(rlist,fwegg_r,label='Pwegg')
    ax.legend()
    ax.grid()
    ax.set_yscale('log')
    ax.set_title('Angular velocity for ISCO')


    return data_for_save_2
def pressecion_plots(L1,L2,Gr_Rp_list,Pn_Rp_list,Pw_Rp_list,Pwegg_Rp_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,g1,g2,f1,f2,type=1):
    L=np.concatenate((L1,L2))
    if type==1: 
        plt.plot(np.array(Gr_Rp_list),np.array(Gr_Rp_list)-np.array(Gr_Rp_list),label='Gr')
        plt.plot(np.array(Gr_Rp_list),(np.array(Pn_Rp_list)-np.array(Gr_Rp_list)),'*',label='Pn')
        plt.plot(np.array(Gr_Rp_list),(np.array(Pw_Rp_list)-np.array(Gr_Rp_list)),'.-',label='Pw')
        plt.plot(np.array(Gr_Rp_list),(np.array(Pwegg_Rp_list)-np.array(Gr_Rp_list)),'--',label='Pwegg')
        plt.legend()
        plt.grid()
        plt.xlabel('Rp_gr')
        plt.ylabel('Rp-Rp_gr')
        plt.title('Rp of potential ')
        # plt.xscale('log')
        # plt.yscale('log')
        return
    if type==2:
        # ind1=np.where((L1-4)>0.0001)[0][0]
        # ind2=np.where((L1-4)>0.001)[0][0]
        # indexgr1 =np.where((np.where(~np.isnan(Gr_parb_prec_list)))>ind1)[1][0]
        # indexpn1 =np.where((np.where(~np.isnan(Pn_parb_prec_list)))>ind1)[1][0]
        # indexpw1 =np.where((np.where(~np.isnan(Pw_parb_prec_list)))>ind1)[1][0]
        # indexwegg1 =np.where((np.where(~np.isnan(Pwegg_parb_prec_list)))>ind1)[1][0]
        # indexgr2 =np.where((np.where(~np.isnan(Gr_parb_prec_list)))>ind2)[1][0]
        # indexpn2 =np.where((np.where(~np.isnan(Pn_parb_prec_list)))>ind2)[1][0]
        # indexpw2 =np.where((np.where(~np.isnan(Pw_parb_prec_list)))>ind2)[1][0]
        # indexwegg2 =np.where((np.where(~np.isnan(Pwegg_parb_prec_list)))>ind2)[1][0]
        plt.plot(L1-4, np.array(Gr_parb_prec_list[:g2]), '*', label='Gr')
        plt.plot(L1-4, np.array(Pn_parb_prec_list[:g2]), '*', label='Pn')
        plt.plot(L1-4, np.array(Pw_parb_prec_list[:g2]), '*', label='Pw')
        plt.plot(L1-4, np.array(Pwegg_parb_prec_list[:g2]), '*', label='Pwegg')
          # # Highlight points from index# to index#+8
        # plt.plot(L1[indexgr1:indexgr2]-4, np.array(Gr_parb_prec_list[indexgr1:indexgr2]), 'o', color='red', label='Gr Highlight')
        # plt.plot(L1[indexpn1:indexpn2]-4, np.array(Pn_parb_prec_list[indexpn1:indexpn2]), 'o', color='green', label='Pn Highlight')
        # plt.plot(L1[indexpw1:indexpw2]-4, np.array(Pw_parb_prec_list[indexpw1:indexpw2]), 'o', color='blue', label='Pw Highlight')
        # plt.plot(L1[indexwegg1:indexwegg2]-4, np.array(Pwegg_parb_prec_list[indexwegg1:indexwegg2]), 'o', color='purple', label='Pwegg Highlight')

        # mgr=((Gr_parb_prec_list[indexgr2]-Gr_parb_prec_list[indexgr1])/(np.log10(L1[indexgr2]-4)-np.log10(L1[indexgr1]-4)))/(np.sqrt(2)/np.log10(np.e))
        # mpn=((Pn_parb_prec_list[indexpn2]-Pn_parb_prec_list[indexpn1])/(np.log10(L1[indexpn2]-4)-np.log10(L1[indexpn1]-4)))/(np.sqrt(2)/np.log10(np.e))
        # mpw=((Pw_parb_prec_list[indexpw2]-Pw_parb_prec_list[indexpw1])/(np.log10(L1[indexpw2]-4)-np.log10(L1[indexpw1]-4)))/(np.sqrt(2)/np.log10(np.e))
        # mwegg=((Pwegg_parb_prec_list[indexwegg2]-Pwegg_parb_prec_list[indexwegg1])/(np.log10(L1[indexwegg2]-4)-np.log10(L1[indexwegg1]-4)))/(np.sqrt(2)/np.log10(np.e))
        
        mgr=0
        mpn=0
        mpw=0
        mwegg=0

        # data_for_save_3=f'the slope of the precession near L=4 as a fraction of -sqrt(2) is: Gr={-mgr:.3f}, Pn={-mpn:.3f}, Pw={-mpw:.3f}, Pwegg={-mwegg:.3f}\n the slope was calculated between the points L: {L1[indexgr1]} to {L1[indexgr2]} for Gr, {L1[indexpn1]} to {L1[indexpn2]} for Pn, {L1[indexpw1]} to {L1[indexpw2]} for Pw, {L1[indexwegg1]} to {L1[indexwegg2]} for Pwegg'
        data_for_save_3=f'the slope of the precession near L=4 as a fraction of -sqrt(2) is: Gr={-mgr:.3f}, Pn={-mpn:.3f}, Pw={-mpw:.3f}, Pwegg={-mwegg:.3f}\n'

        plt.xscale('log')
        plt.ylim(0,16)
        plt.legend()
        plt.grid()
        plt.title('Precession parabolic near L=4')
        plt.xlabel('L-4')
        plt.ylabel('Precession')
        return data_for_save_3,mgr,mpn,mpw,mwegg
    if type==3:
        plt.plot(L2,np.array(Gr_parb_prec_list[g2:g2+f2]),'--',label='Gr')
        plt.plot(L2,(np.array(Pn_parb_prec_list[g2:g2+f2])),'--',label='Pn')
        plt.plot(L2,np.array(Pw_parb_prec_list[g2:g2+f2]),'-.',label='Pw')
        plt.plot(L2,np.array(Pwegg_parb_prec_list[g2:g2+f2]),'-',label='Pwegg')

        # y=6*np.pi/(L2**2)
        # plt.plot(L2,y,'--',label='$-2*log(L)+log(3*pi)$')

        ind_far_1=int(f2/2)
        ind_far_2=int(f2/4)

        # plt.plot(L2[ind_far_2:ind_far_1], np.array(Gr_parb_prec_list[g2+ind_far_2:g2+ind_far_1]), 'o', color='red', label='Gr Highlight')
        # plt.plot(L2[ind_far_2:ind_far_1], np.array(Pn_parb_prec_list[g2+ind_far_2:g2+ind_far_1]), 'o', color='green', label='Pn Highlight')
        # plt.plot(L2[ind_far_2:ind_far_1], np.array(Pw_parb_prec_list[g2+ind_far_2:g2+ind_far_1]), 'o', color='blue', label='Pw Highlight')
        # plt.plot(L2[ind_far_2:ind_far_1], np.array(Pwegg_parb_prec_list[g2+ind_far_2:g2+ind_far_1]), 'o', color='purple', label='Pwegg Highlight')

        plt.xscale('log')
        plt.yscale('log')
        # lower_ylim=np.nanmin([np.nanmin(Gr_parb_prec_list[g2:g2+f2]),np.nanmin(Pn_parb_prec_list[g2:g2+f2]),np.nanmin(Pw_parb_prec_list[g2:g2+f2]),np.nanmin(Pwegg_parb_prec_list[g2:g2+f2])])
        # upper_ylim=np.nanmax([Gr_parb_prec_list[g2],Pn_parb_prec_list[g2],Pw_parb_prec_list[g2],Pwegg_parb_prec_list[g2]])
        # print(lower_ylim,upper_ylim)
        plt.ylim(10**(-4),8)
        plt.legend()
        plt.grid()
        plt.title('Precession parabolic nfar from L=4')



        bgr=((np.log10(Gr_parb_prec_list[g2+int(f2/2)]))+2*(np.log10(L2[int(f2/2)])))/np.log10(6*np.pi)
        bpn=((np.log10(Pn_parb_prec_list[g2+int(f2/2)]))+2*(np.log10(L2[int(f2/2)])))/np.log10(6*np.pi)
        bpw=((np.log10(Pw_parb_prec_list[g2+int(f2/2)]))+2*(np.log10(L2[int(f2/2)])))/np.log10(6*np.pi)
        bwegg=((np.log10(Pwegg_parb_prec_list[g2+int(f2/2)]))+2*(np.log10(L2[int(f2/2)])))/np.log10(6*np.pi)




        mgr2=(np.log10(Gr_parb_prec_list[g2+int(f2/2)])-np.log10(Gr_parb_prec_list[g2+int(f2/4)]))/(np.log10(L2[int(f2/2)])-np.log10(L2[int(f2/4)]))
        mpn2=(np.log10(Pn_parb_prec_list[g2+int(f2/2)])-np.log10(Pn_parb_prec_list[g2+int(f2/4)]))/(np.log10(L2[int(f2/2)])-np.log10(L2[int(f2/4)]))
        mpw2=(np.log10(Pw_parb_prec_list[g2+int(f2/2)])-np.log10(Pw_parb_prec_list[g2+int(f2/4)]))/(np.log10(L2[int(f2/2)])-np.log10(L2[int(f2/4)]))
        mwegg2=(np.log10(Pwegg_parb_prec_list[g2+int(f2/2)])-np.log10(Pwegg_parb_prec_list[g2+int(f2/4)]))/(np.log10(L2[int(f2/2)])-np.log10(L2[int(f2/4)]))

        data_for_save_4=f'The far rp>>rs precession behavior: \n Gr:log(prec)={mgr2:.1f}*log(L)+{bgr:.2f}log(6pi) \n Pn:log(prec)={mpn2:.1f}*log(L)+{bpn:.2f}log(6pi) \n Pw:log(prec)={mpw2:.2f}*log(L)+{bpw:.2f}log(6pi) \n Pwegg:log(prec)={mwegg2:.1f}*log(L)+{bwegg:.2f}log(6pi) \n the slope was calculated between the points L: {L2[int(f2/4)]} to {L2[int(f2/2)]} for Gr, {L2[int(f2/4)]} to {L2[int(f2/2)]} for Pn, {L2[int(f2/4)]} to {L2[int(f2/2)]} for Pw, {L2[int(f2/4)]} to {L2[int(f2/2)]} for Pwegg'
        return data_for_save_4,mgr2,mpn2,mpw2,mwegg2,bgr,bpn,bpw,bwegg
    if type==4:
        
        plt.plot(L-4,((np.array(Pn_parb_prec_list)-np.array(Gr_parb_prec_list)))/np.array(Gr_parb_prec_list),'*',label='Pn_error')
        plt.plot(L-4,(np.array(Pw_parb_prec_list)-np.array(Gr_parb_prec_list))/np.array(Gr_parb_prec_list),label='Pw_error')
        plt.plot(L-4,(np.array(Pwegg_parb_prec_list)-np.array(Gr_parb_prec_list))/np.array(Gr_parb_prec_list),label='Pwegg_error')

        plt.plot(L-4,np.zeros(len(L)),'--',label='0')
        plt.legend()
        plt.xscale('log')
        lower_ylim=-1
        upper_ylim=0.5
        plt.ylim(lower_ylim,upper_ylim)

        plt.grid()
        plt.title('Precession difference precentage parabolic near L=4')
        plt.xlabel('L-4')
        plt.ylabel('Precession error %')
        return
    if type==5:
        Rp_wegg=2*(np.sqrt(6)-1)
        plt.plot(np.array(Gr_Rp_list[:g2]), np.array(Gr_parb_prec_list[:g2]), '*', label='Gr')
        plt.plot(np.array(Pn_Rp_list[:g2]), np.array(Pn_parb_prec_list[:g2]), '*', label='Pn')
        plt.plot(np.array(Pw_Rp_list[:g2]), np.array(Pw_parb_prec_list[:g2]), '*', label='Pw')
        plt.plot(np.array(Pwegg_Rp_list[:g2]), np.array(Pwegg_parb_prec_list[:g2]), '*', label='Pwegg')
        plt.plot(np.array(Pwegg_Rp_list[:g2])-Rp_wegg+4, np.array(Pwegg_parb_prec_list[:g2]), '*', label='Pwegg_moved')
        plt.legend()
        plt.grid()
        plt.title('Precession parabolic near L=4')
        plt.xlabel('Rp')
        plt.ylabel('Precession')
        plt.xscale('log')
        plt.yscale('log')
        return


def final_plots(rangelist,auto=False,N1_list=[],rs_list=[],conds_list=[]):
    font_size = 16
    marksizq = 1
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

    conds_to_set = {}
    cond_list_titles = []
    for conds in conds_list:
        conds_tuple = tuple(conds)
        if conds_tuple not in conds_to_set:
            conds_to_set[conds_tuple] = len(conds_to_set)
        cond_list_titles.append(conds_to_set[conds_tuple])

    for set_number, conds in enumerate(conds_to_set.keys()):
        print(f"Set {set_number} is {list(conds)}")

    gr_l_list = np.array(data_tmp['Gr_L_list'])
    gr_precession_list = np.array(data_tmp['Gr_parb_prec_list'])
    pw_l_list = np.array(data_tmp['Pw_L_list'])
    pw_precession_list = np.array(data_tmp['Pw_parb_prec_list'])
    wegg_l_list = np.array(data_tmp['Pwegg_L_list'])
    wegg_precession_list = np.array(data_tmp['Pwegg_parb_prec_list'])
    rp1_wegg = np.array(data_tmp['rp_wegg1'])
    rp2_wegg = np.array(data_tmp['rp_wegg2'])

    rlist = np.linspace(2, 13, 1000)
    fig, axs = plt.subplots(3, 1, figsize=(6, 18), constrained_layout=True)
    ax = axs[0]
    L = 4
    for i in range(len(N1_list)):
        color = f"C{i % 10}"
        ax.plot(rlist, 2 * u(rlist, N1_list[i], coefficient_lists[i], rs_list[i]) + (L**2 / rlist**2), '-*', label=f'PN-N1={N1_list[i]}', color=color, markersize=marksizq, linewidth=0.7 * marksizq)

    ax.plot(rlist, (1 - (2 / rlist)) * (1 + (L**2 / rlist**2)) - 1, 'k-', label='Gr')
    ax.plot(rlist, 2 * u_pw(rlist) + (L**2 / rlist**2), 'g-.', label='Pw')
    ax.plot(rlist, 2 * u_wegg(rlist) + (L**2 / rlist**2), 'r--', label='Pwegg')
    ax.set_ylim(-0.13, 0.025)
    ax.set_xlim(2, 13)
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.set_xlabel('r', fontsize=font_size + 4)
    ax.set_ylabel(r'$V_{eff}$', fontsize=font_size + 4)
    ax.text(0.05, 0.95, "(I)", transform=ax.transAxes, fontsize=font_size, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))
    ax.legend(fontsize=font_size - 2)

    ax = axs[1]
    for i in range(len(N1_list)):
        color = f"C{i % 10}"
        ax.plot(prcession_L_lists[i][:rangelist[1]] - prcession_L_lists[i][0], 
                np.array(precession_value_lists[i][:rangelist[1]]) / np.pi, 
                '-*', label=f'PN-N1={N1_list[i]}', color=color, markersize=marksizq, linewidth=0.7 * marksizq)

    ax.plot(gr_l_list[:rangelist[1]] - gr_l_list[0], 
        np.array(gr_precession_list[:rangelist[1]]) / np.pi, 
        'k-', label='Gr')
    ax.plot(pw_l_list[:rangelist[1]] - pw_l_list[0], 
        np.array(pw_precession_list[:rangelist[1]]) / np.pi, 
        'g-.', label='Pw')
    ax.plot(wegg_l_list[:rangelist[1]] - wegg_l_list[0], 
        np.array(wegg_precession_list[:rangelist[1]]) / np.pi, 
        'r--', label='Pwegg')

    ax.set_xscale('log')
    ax.set_ylim(1, 5)
    ax.set_xlim(10**(-5), 3*10**(-2))
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.set_ylabel(r'$\frac{\Delta\phi}{\pi}$', fontsize=font_size + 4, rotation=0, labelpad=20)
    ax.set_xlabel('L-4', fontsize=font_size + 4)
    ax.text(0.05, 0.95, "(II)", transform=ax.transAxes, fontsize=font_size, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))   
    ax.legend(fontsize=font_size - 2)

    ax = axs[2]
    for i in range(len(N1_list)):
        color = f"C{i % 10}"
        ax.plot(prcession_L_lists[i][rangelist[1]:], np.array(precession_value_lists[i][rangelist[1]:]) / np.pi, '-*', label=f'PN-N1={N1_list[i]}', color=color, markersize=marksizq, linewidth=0.7 * marksizq)

    ax.plot(gr_l_list[rangelist[1]:], np.array(gr_precession_list[rangelist[1]:]) / np.pi, 'k-', label='Gr')
    ax.plot(pw_l_list[rangelist[1]:], np.array(pw_precession_list[rangelist[1]:]) / np.pi, 'g-.', label='Pw')
    ax.plot(wegg_l_list[rangelist[1]:], np.array(wegg_precession_list[rangelist[1]:]) / np.pi, 'r--', label='Pwegg')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(3*10**(-4), 10**-2)
    ax.set_xlim(25, 10**2)
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel('L', fontsize=font_size + 4)
    ax.set_ylabel(r'$\frac{\Delta\phi}{\pi}$', fontsize=font_size + 4, rotation=0, labelpad=20)
    ax.text(0.05, 0.95, "(III)", transform=ax.transAxes, fontsize=font_size, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))
    ax.legend(fontsize=font_size - 2)

    # Save the figure to the desktop with no background
    plt.savefig("C:/Users/itama/Desktop/Figure1_papper.png", transparent=True)
    plt.show()
    ##fig 4 f(r) in Isco orbit
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
    fdot_wegg = (Lweggisco / (rlist_wegg**2))
    fdot_gr = (1 - (2 / rlist)) * (Lgrisco / rlist**2) / (Egrisco)
    fdot_pw = (Lpwisco / (rlist**2))
    fgr_r = sp.integrate.cumulative_simpson(fdot_gr / rdot_gr, x=rlist, initial=0)
    fpw_r = sp.integrate.cumulative_simpson(fdot_pw / rdot_pw, x=rlist, initial=0)
    xgr = rlist * np.cos(fgr_r)
    ygr = rlist * np.sin(fpw_r)
    xpw = rlist * np.cos(fpw_r)
    ypw = rlist * np.sin(fpw_r)

    fig, axs = plt.subplots(2, 1, figsize=(10, 14), constrained_layout=True)
    font_size = 23

    # Subplot (a)
    ax = axs[0]
    for i in range(len(N1_list)):
        color = f"C{i % 10}"
        ax.plot(rlist_list[i], (np.array(rdotpn_list[i]) - np.array(rdot_gr[:len(rdotpn_list[i])])), '-*', label=f'Pn-N1={N1_list[i]}', markersize=0.3, color=color)

    ax.plot(rlist, (np.array(rdot_pw) - (np.array(rdot_gr[:len(rdot_pw)]))), '-.', label='Pw', color='C16')
    ax.plot(rlist_wegg[:len(rdot_gr)] + (6 - r_wegg_isdo), (np.array(rdot_wegg[:len(rdot_gr)]) - (np.array(rdot_gr))), 'r--', label='Pwegg(shifted)')
    ax.legend(markerscale=10, loc='upper right', fontsize=font_size - 6)
    ax.grid()
    ax.set_ylim(-0.01, 0.11)
    ax.set_xlim(4, 6)
    ax.set_xlabel('r', fontsize=font_size)
    ax.set_ylabel(r"$\Delta \dot{r}$ ", fontsize=font_size, labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=font_size - 2)
    ax.text(0.05, 0.95, "(a)", transform=ax.transAxes, fontsize=font_size, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

    # Subplot (b)
    rlist_isco = np.linspace(2, 10, 100000)[1:-2]
    Lgrisco = np.sqrt(12)
    veff_gr_isco = (1 - (2 / rlist_isco)) * (1 + ((Lgrisco**2) / (rlist_isco**2))) - 1
    ax = axs[1]
    for i in range(len(N1_list)):
        color = f"C{i % 10}"
        L_pn_isco = np.sqrt((6**3) * u_dr(6, N1_list[i], coefficient_lists[i], rs_list[i]))
        veff_pn_isco = 2 * u(rlist_isco, N1_list[i], coefficient_lists[i], rs_list[i]) + ((L_pn_isco**2) / (rlist_isco**2))
        ax.plot(rlist_isco, veff_pn_isco, '-*', label=f'PN-N1={N1_list[i]}', color=color, markersize=0.3, linewidth=0.7)

    Lpwisco = np.sqrt((6**3) * u_pw_dr(6))
    veff_pw_isco = 2 * u_pw(rlist_isco) + ((Lpwisco**2) / (rlist_isco**2))
    veff_wegg_isco = 2 * u_wegg(rlist_isco) + ((Lweggisco**2) / (rlist_isco**2))
    ax.plot(rlist_isco, veff_wegg_isco, 'r--', label='Pwegg', markersize=0.3, linewidth=0.7)
    ax.plot(rlist_isco, veff_gr_isco, 'k-', label='Gr')
    ax.plot(rlist_isco, veff_pw_isco, 'g-.', label='Pw')
    ax.set_xlim(2, 10)
    ax.set_ylim(-0.2, -0.07)
    ax.axvline(x=6, color='gray', linestyle=':', linewidth=3)
    ax.text(6.1, -0.1, 'r=6', fontsize=font_size - 2, color='gray', verticalalignment='center')
    ax.legend(markerscale=1, loc='upper right', fontsize=font_size - 6)
    ax.grid()
    ax.set_xlabel('r', fontsize=font_size)
    ax.set_ylabel(r"$V_{eff}$", fontsize=font_size, labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=font_size - 2)
    ax.text(0.05, 0.95, "(b)", transform=ax.transAxes, fontsize=font_size, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

    plt.savefig("C:/Users/itama/Desktop/Figure2_papper.png", transparent=True)
    plt.show()

    return
