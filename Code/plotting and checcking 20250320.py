import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import mcint as mc
import sys
from matplotlib import cm, ticker
sys.path.append('Scripts/Research scripts/General Functions')
from Monte_Carlo import Monte_Carlo_Integrate as mci
from New_Potential2025_02_26_addwd_term import semi2 
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



# Create a new folder named after today's date in the format yyyymmdd

def set_log_today():
    today_date = datetime.today().strftime('%Y%m%d')
    new_folder_path = f"C:\\Users\\itama\\Documents\\.venv\\Scripts\\Research scripts\\PNP_research_assets\\{today_date}"
    new_folder_path_drafts = f"C:\\Users\\itama\\Documents\\.venv\\Scripts\\Research scripts\\PNP_research_assets\\{today_date}\\Drafts"
    load_folder_path = f"C:\\Users\\itama\\Documents\\.venv\\Scripts\\Research scripts\\PNP_research_assets\\"
    new_json_folder_path=f"C:\\Users\\itama\\Documents\\.venv\\Scripts\\Research scripts\\PNP_research_assets\\{today_date}\\Jsons"
    load_json_folder_path=f"C:\\Users\\itama\\Documents\\.venv\\Scripts\\Research scripts\\PNP_research_assets\\"
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path, exist_ok=True)
    if not os.path.exists(new_json_folder_path):
        os.makedirs(new_json_folder_path, exist_ok=True)
    if not os.path.exists(new_folder_path_drafts):
        os.makedirs(new_folder_path_drafts, exist_ok=True)
    imagepath=new_folder_path
    return new_folder_path, new_folder_path_drafts, load_folder_path, new_json_folder_path, load_json_folder_path, imagepath
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
b_global=np.array([1,0,2,(1/3),3,0,4,2,9,36,(2/9),(5/27),2,-1/9])
#region condition functions:
def c1_alpha(n,rs):
    tmp=6*(1/((6-rs)**3))*((6/(6-rs))**n)*((rs**2)*(n*(n+2))+6*(n+3)*rs-36)
    return tmp
def c1_beta(n):
    return ((n**2)-1)*(6**(-n))
def c2_alpha(n,rs):
    tmp=6*(1/((6-rs)**2))*((6/(6-rs))**n)*(6+n*rs)
    return tmp
def c2_beta(n):
    return (n+1)*(6**(-n))
def c3_alpha(n,rs):
    tmp=3*(1/((3-rs)**2))*((3/(3-rs))**n)*(3+rs*n)
    return tmp
def c3_beta(n):
    return (n+1)*(3**(-n))
def c4_alpha(n,rs):
    tmp=(n+1)*rs
    return tmp
def c5_alpha(n,rs):
    tmp=4*(1/((4-rs)**2))*((4/(4-rs))**n)*((n+2)*rs-4)
    return tmp
def c5_beta(n):
    return (n-1)*(4**(-n))
def c6_alpha(n,rs):
    tmp=4*(1/((4-rs)**2))*((4/(4-rs))**n)*(4+n*rs)
    return tmp
def c6_beta(n):
    return (n+1)*(4**(-n))
def c7_alpha(n,rs):
    tmp=4*(1/((4-rs)**3))*((4/(4-rs))**n)*(-16+(rs**2)*n*(n+2)+4*rs*(n+3))
    return tmp
def c7_beta(n):
    return (4**(-n))*((n**2)-1)
def c8_alpha(n,rs):
    tmp=2*((4/(4-rs))**n)*(1/((4-rs)**4))*(-384+rs*(96*(n+4)+rs*(-48+n*(60+36*n+rs*(n-5)*(n+2)))))
    return tmp
def c8_beta(n):
    return (1/2)*(4**(-n))*((n-1)*(n+1)*(n+6))
def c9_alpha(n,rs):
    tmp=(1/((4-rs)**5))*((4/(4-rs))**n)*(-9216+2304*rs*(n+5)+576*(rs**2)*(n-1)*(2*n+5)+(rs**4)*n*(n+2)*(27+(n-8)*n)+(rs**3)*16*(15+n*(-37+4*(n-3)*n)))
    return tmp
def c9_beta(n):
    return (1/4)*(4**(-n))*(n-1)*(n+1)*(36+n*(n+10))
def c10_alpha(n,rs):
    tmp=2*(1/(12-rs)**2)*((12/(12-rs))**n)*(12+n*rs)
    return tmp
def c10_beta(n):
    return (1/6)*(n+1)*(12**(-n))
def c11_alpha(n,rs):
    tmp=2*(1/(12-rs))*((12/(12-rs))**n)
    return tmp
def c11_beta(n):
    return (1/6)*(12**(-n))
def c12_alpha(n,rs):
    tmp=6*(1/(6-rs)**4)*((6/(6-rs))**n)*(-1296+rs*(216*(n+4)+rs*(-72+n*(90+54*n+rs*(n-5)*(n+2)))))
    return tmp
def c12_beta(n):
    return (6**(-n))*(n-1)*(n+1)*(n+6)
def c13_alpha(n,rs):
    tmp=(1/((6-rs)**2))*((6/(6-rs))**n)*(rs*(2+n)-6)
    return tmp
def c13_beta(n):
    return (1/6)*(6**(-n))*(n-1)
#endregion
#region functions:
def Solve_coeffs(N1,rs,conds):
    N=len(conds)
    N2=N-N1
    b_global=np.array([1,0,2,(1/3),3,0,4,2,9,36,(2/9),(5/27),2,-(1/9)])
    b=b_global[conds]
    tmpvecbeta1=np.concatenate((np.array([1]),np.zeros(N-1)))
    tmpvecbeta2=np.concatenate((np.array([0,1]),np.zeros(N-2)))


    r0=np.concatenate((np.ones(N1),tmpvecbeta1[:N2]))
    r1=np.concatenate((np.array([c1_alpha(i,rs) for i in range(N1)]),np.array([c1_beta(i) for i in range(N2)])))
    r2=np.concatenate((np.array([c2_alpha(i,rs) for i in range(N1)]),np.array([c2_beta(i) for i in range(N2)])))
    r3=np.concatenate((np.array([c3_alpha(i,rs) for i in range(N1)]),np.array([c3_beta(i) for i in range(N2)])))
    r4=np.concatenate((np.array([c4_alpha(i,rs) for i in range(N1)]),tmpvecbeta2[:N2]))
    r5=np.concatenate((np.array([c5_alpha(i,rs) for i in range(N1)]),np.array([c5_beta(i) for i in range(N2)])))
    r6=np.concatenate((np.array([c6_alpha(i,rs) for i in range(N1)]),np.array([c6_beta(i) for i in range(N2)])))
    r7=np.concatenate((np.array([c7_alpha(i,rs) for i in range(N1)]),np.array([c7_beta(i) for i in range(N2)])))
    r8=np.concatenate((np.array([c8_alpha(i,rs) for i in range(N1)]),np.array([c8_beta(i) for i in range(N2)])))
    r9=np.concatenate((np.array([c9_alpha(i,rs) for i in range(N1)]),np.array([c9_beta(i) for i in range(N2)])))
    r10=np.concatenate((np.array([c10_alpha(i,rs) for i in range(N1)]),np.array([c10_beta(i) for i in range(N2)])))
    r11=np.concatenate((np.array([c11_alpha(i,rs) for i in range(N1)]),np.array([c11_beta(i) for i in range(N2)])))
    r12=np.concatenate((np.array([c12_alpha(i,rs) for i in range(N1)]),np.array([c12_beta(i) for i in range(N2)])))
    r13=np.concatenate((np.array([c13_alpha(i,rs) for i in range(N1)]),np.array([c13_beta(i) for i in range(N2)])))

    A_global=np.array([r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13])
    A=A_global[conds]
    # print(A)
    x=np.linalg.solve(A,b)
    print(x)
    data = {}
    data['x'] = x.tolist()
    return x,data
def u(r,N1,x,rs):
    N2=len(x)-N1
    N=len(x)
    c1=x[:N1]  
    c2=x[N1:]
    vec1=np.array([c1[n]*((r**n)/((r-rs)**(n+1)))  for n in range(N1)])
    vec2=np.array([c2[n]*((r**(-n-1)))  for n in range(N2)])
    if N1==N:
        return -1*sum(vec1)
    elif N1==0:
        return -1*sum(vec2)
    else:
        vec=np.concatenate((vec1,vec2))
        return -1*sum(vec)
def u_pw(r):
    return -1/(r-2)
def u_wegg(r):
    alph=-(4/3)*(2+np.sqrt(6))
    Rx=(4*np.sqrt(6))-9
    Ry=-(4/3)*(-3+2*np.sqrt(6))
    return -alph/(r)-((1-alph)/(r-Rx))-Ry/(r**2)
def u_dr(r,N1,x,rs):
    N2=len(x)-N1
    N=len(x)
    c1=x[:N1]   
    c2=x[N1:]
    vec1=np.array([c1[n]*((r**(n-1))/((r-rs)**(n+2)))*(r+rs*n)  for n in range(N1)])
    vec2=np.array([c2[n]*(((n+1)*(r**(-n-2))))  for n in range(N2)])
    if N1==N:
        return sum(vec1)
    elif N1==0:
        return sum(vec2)
    else:
        vec=np.concatenate((vec1,vec2))
        return sum(vec)
def u_pw_dr(r):
    return 1/((r-2)**2)
def u_wegg_dr(r):
    alph=-(4/3)*(2+np.sqrt(6))
    Rx=(4*np.sqrt(6))-9
    Ry=-(4/3)*(-3+2*np.sqrt(6))
    return alph/(r**2)+((1-alph)/(((r-Rx)**2)))+2*Ry/(r**3)
def Gr_precession_gen(L,E,lim1=10**8,lim2=10**-7):
    Rp=sp.optimize.fsolve(lambda r: ((E+1)**2)-(1-(2/r))*(1+((L**2)/(r**2))),4)[0]
    
    
    def tmp1(r):
        if  np.isclose(((E+1)**2)-(1-(2/r))*(1+((L**2)/(r**2))),0) and ((E+1)**2)-(1-(2/r))*(1+((L**2)/(r**2)))<0:
            return 0
        if r==Rp:
            return 0
        else:
            return((E+1)**2)-(1-(2/r))*(1+((L**2)/(r**2)))



    def integrand(r):
        return L/((r**2)*(tmp1(r))**(1/2))
    
    # print("parabolic orbit")
    res=2*sp.integrate.quad(integrand,Rp,np.inf,limit=lim1,epsabs=lim2, epsrel=lim2)[0]-2*np.pi
    # if np.isnan(res):
    #     print('oh no gr')
    #     res= mci(integrand,Rp,1000,n=10**7)
    return res,Rp
def Pn_precession_gen(L,E,rs,x,N1,lstep=1,Rp_old=4,lim1=10**8,lim2=10**-7):

    Rp=sp.optimize.fsolve(lambda r: 2*E-2*u(r,N1,x,rs)-((L**2)/(r**2)),Rp_old+2*(lstep))[0]
    if Rp>10**7 or Rp<0:
        return 0,0
    # plt.plot(np.linspace(3,10000,1000),[-2*u(r,N1,x,rs)-(((4.04)**2)/(r**2)) for r in np.linspace(3,10000,1000)])
    # plt.show()
    # print(Rp)
    
    # print(Rp)
    # print(Rp)
    #Rp_old+2*(lstep)


    def tmp1(r):
        if  np.isclose(2*E-2*u(r,N1,x,rs)-((L**2)/(r**2)),0) and 2*E-2*u(r,N1,x,rs)-((L**2)/(r**2))<0:
            return 0
        if r==Rp:
            return 0
        else:
            return 2*E-2*u(r,N1,x,rs)-((L**2)/(r**2))
    # def tmp1(r):
    #     return 2*E-2*u(r)-((L**2)/(r**2))

    def integrand(r):
        return L/((r**2)*(tmp1(r))**(1/2))
    # print(f'Rp is {Rp},L is {L}')
    # plt.plot(np.linspace(Rp,10000,1000),[2*E-tmp1(r) for r in np.linspace(Rp,10000,1000)])
    # plt.show()
    


    res=2*sp.integrate.quad(integrand,Rp,np.inf,limit=lim1*10,epsabs=lim2, epsrel=lim2)[0]-2*np.pi

    return res,Rp    
def Pw_precession_gen(L,E,lim1=10**8,lim2=10**-7):
    # L=np.sqrt(2*(Rp**2)*(E-u_pw(Rp)))
    # Rp=sp.optimize.fsolve(lambda p: (2*(p**2)*(E-u_pw(p)))-(L**2),4)[0]
    Rp=sp.optimize.fsolve(lambda r: 2*E-2*u_pw(r)-((L**2)/(r**2)),4)[0]
    def integrand(r):
        return L/((r**2)*((2*E-2*u_pw(r)-((L**2)/(r**2)))**(1/2)))
    # print("parabolic orbit")
    res=2*sp.integrate.quad(integrand,Rp,np.inf,limit=lim1,epsabs=lim2, epsrel=lim2)[0]-2*np.pi
    # if np.isnan(res):
    #     print('oh no pw')
    #     res= mci(integrand,Rp,1000,n=10**7)
    return res,Rp
def Pwegg_precession_gen(L,E,lim1=10**8,lim2=10**-7):
    # L=np.sqrt(2*(Rp**2)*(E-u_wegg(Rp)))
    Rp=sp.optimize.fsolve(lambda p: (2*(p**2)*(E-u_wegg(p)))-(L**2),4)[0]
    def integrand(r):
        return L/((r**2)*((2*E-2*u_wegg(r)-((L**2)/(r**2)))**(1/2)))
    # print("parabolic orbit")
    res=2*sp.integrate.quad(integrand,Rp,np.inf,limit=lim1,epsabs=lim2, epsrel=lim2)[0]-2*np.pi
    # if np.isnan(res):
    #     print('oh no pwegg')
    #     res= mci(integrand,Rp,1000,n=10**7)
    return res,Rp
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
def calculate_precession(rs,N1,x,g1,g2,f1,f2,lim1=10**8,lim2=10**-7):
        Gr_Rp_list=[]
        Pn_Rp_list=[]
        Pw_Rp_list=[]
        Pwegg_Rp_list=[]
        Gr_parb_prec_list=[]
        Pn_parb_prec_list=[]
        Pw_parb_prec_list=[]
        Pwegg_parb_prec_list=[]
        if (g1/g2)>10**-3:
            print('The step size is too big, the results may be inaccurate')
        L1= np.linspace(4,4+g1,g2)
        L2= np.linspace(4+g1,f1,f2)
        L=np.concatenate((L1,L2))
        timestep=[]
        for i,l in enumerate(L):
            t1=time.time()
            p,rp=Gr_precession_gen(l,0,lim1,lim2)
            Gr_parb_prec_list.append(p)
            Gr_Rp_list.append(rp)
            if i==0:
                p,rp=Pn_precession_gen(l,0,rs,x,N1,lstep=L[1]-L[0],lim1=lim1,lim2=lim2)
                Pn_parb_prec_list.append(p)
                Pn_Rp_list.append(rp)
            else:
                p,rp=Pn_precession_gen(l,0,rs,x,N1,lstep=L[1]-L[0],Rp_old=Pn_Rp_list[-1],lim1=lim1,lim2=lim2)

                Pn_parb_prec_list.append(p)
                Pn_Rp_list.append(rp)
            p,rp=Pw_precession_gen(l,0,lim1,lim2)
            Pw_parb_prec_list.append(p)
            Pw_Rp_list.append(rp)
            p,rp=Pwegg_precession_gen(l,0,lim1,lim2)
            Pwegg_parb_prec_list.append(p)
            Pwegg_Rp_list.append(rp)
            timestep.append(time.time()-t1)
            if i%500==0:
                # step=np.mean(timestep)
                # print(f'Approximate time left : {(step*(len(L)-i-1))/60:.3f} m, time passed from begining: {(time.time()-t0)/60:.3f} m ',end='\r',flush=True) 
                progress = (i + 1) / len(L)
                bar_length = 40
                block = int(round(bar_length * progress))
                text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}%"
                print(text, end='', flush=True)
        return Gr_Rp_list,Pn_Rp_list,Pw_Rp_list,Pwegg_Rp_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,L1,L2
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
def scan_parameters(list_now,part):
    string_list_now=''.join([str(i) for i in list_now])
    conditions_explained=[conditions_text_list[i] for i in list_now]

    def check_existing_file(string_list_now, part):
        base_path = "C:\\Users\\itama\\Documents\\.venv\\Scripts\\Research scripts\\PNP_research_assets\\all_comd_50_rs_values_all_n1"
        for file_name in os.listdir(base_path):
            if file_name.startswith(f"{string_list_now}__{part}"):
                return True
        return False

    check=check_existing_file(string_list_now, part)
    base_path = "C:\\Users\\itama\\Documents\\.venv\\Scripts\\Research scripts\\PNP_research_assets\\all_comd_50_rs_values_all_n1"
    # new_folder_path, new_folder_path_drafts, load_folder_path, new_json_folder_path, load_json_folder_path, imagepath = set_log_today()
    # details_file_path = os.path.join(new_folder_path, f"{string_list_now}__{part}_detailes_readme.txt")
    coeffs=[]
    N1_list=[]  
    rs_list=[]
    if not check:
        print('new combination of combitions and part, need to calculate')
        cnt=0
        for i in range(len(list_now)):
            for s in np.linspace(0.1,2,part):
                a,data=Solve_coeffs(i,s,list_now)
                coeffs.append(a)
                N1_list.append(i)
                rs_list.append(s)
                alpha_beta_data = {
                    'coeffs': [a.tolist() for a in coeffs],
                    'N1': [b for b in N1_list],
                    'rs': [c for c in rs_list],
                    'cnt': [cnt]
                }

        with open(base_path + '\\'  + f'{string_list_now}__{part}_alpha_beta_data.json', 'w') as outfile:
            json.dump(alpha_beta_data, outfile)


    #load coeffs AND n1_list
    with open(base_path + '\\'  + f'{string_list_now}__{part}_alpha_beta_data.json', 'r') as infile:
        alpha_beta_data = json.load(infile)
        coeffs = [np.array(a) for a in alpha_beta_data['coeffs']]
        N1_list = alpha_beta_data['N1']
        rs_list = alpha_beta_data['rs']
        cnt = len(N1_list)
    rlist=np.linspace(2,10,10000)
    Lgrisco=np.sqrt(12)
    veff_gr=(1-(2/rlist))*(1+((Lgrisco**2)/(rlist**2)))-1
    for i in range(cnt):
        if i % part == 0:
            continue
        color = f"C{N1_list[i] % 10}"  # Use modulo to cycle through 10 colors
        u_list = 2 * u(rlist, N1_list[i], coeffs[i], rs_list[i]) + ((6**3) * (u_dr(6, N1_list[i], coeffs[i], rs_list[i]))) / (rlist**2)
        if (((6**3) * (u_dr(6, N1_list[i], coeffs[i], rs_list[i])))) < 0:
            continue
        if i % part == 2:
            plt.plot(rlist, u_list, color=color, linestyle='--', linewidth=4)
        elif i % part-1 == 0:
            plt.plot(rlist, u_list, color=color, linestyle='-.', linewidth=4)
        else:
            plt.plot(rlist, u_list, color=color)
        handles = []
        for n1 in set(N1_list):
            color = f"C{n1 % 10}"
            handles.append(plt.Line2D([0], [0], color=color, label=f'N1={n1}'))
        plt.legend(handles=handles)
    plt.plot(rlist, veff_gr, label='Gr')
    plt.xlim(2, 10)
    plt.ylim(-1, 0.5)
    plt.grid()
    plt.title('u(r)')
    plt.show()
    
    # Plot for each N1 only the rs=0.14 and the rs=2 potentials and the gr potentials
    u_list_tmp=[]
    for i in range(cnt):
        if rs_list[i] not in [rs_list[2], rs_list[-2]]:
            continue
        color = f"C{N1_list[i] % 10}"  # Use modulo to cycle through 10 colors
        u_list = 2 * u(rlist, N1_list[i], coeffs[i], rs_list[i]) + ((6**3) * (u_dr(6, N1_list[i], coeffs[i], rs_list[i]))) / (rlist**2)
        if (((6**3) * (u_dr(6, N1_list[i], coeffs[i], rs_list[i])))) < 0:
            continue
        if rs_list[i] == rs_list[2]:
            plt.plot(rlist, u_list, color=color, linestyle='--', linewidth=2)
            u_list_tmp=u_list
        elif rs_list[i] == rs_list[-2]:
            plt.plot(rlist, u_list, color=color, linestyle='-.', linewidth=2)
            plt.fill_between(rlist, u_list,  u_list_tmp, color=color, alpha=0.3)
        handles = []
        for n1 in set(N1_list):
            color = f"C{n1 % 10}"
            handles.append(plt.Line2D([0], [0], color=color, label=f'N1={n1}'))
        plt.legend(handles=handles)
    plt.plot(rlist, veff_gr, label='Gr')
    plt.xlim(2, 10)
    plt.ylim(-1, 0.5)
    plt.grid()
    plt.title(f'u(r) for rs={rs_list[2]:.2f} and rs={rs_list[-2]:.2f}')
    plt.show()
    # New plot with L^2 = -2 * u(4) * 16

    veff_gr_new = (1 - (2 / rlist)) * (1 + (16 / (rlist**2))) - 1

    for i in range(cnt):
        if i % part == 0:
            continue
        color = f"C{N1_list[i] % 10}"  # Use modulo to cycle through 10 colors
        L_squared = -2 * u(4, N1_list[i], coeffs[i], rs_list[i]) * 16
        u_list_new = 2 * u(rlist, N1_list[i], coeffs[i], rs_list[i]) + (L_squared / (rlist**2))
        if L_squared < 0:
            continue
        if i % part == 2:
            plt.plot(rlist, u_list_new, color=color, linestyle='--', linewidth=4)
        elif i % part-1 == 0:
            plt.plot(rlist, u_list_new, color=color, linestyle='-.', linewidth=4)
        else:
            plt.plot(rlist, u_list_new, color=color)
        handles = []
        for n1 in set(N1_list):
            color = f"C{n1 % 10}"
            handles.append(plt.Line2D([0], [0], color=color, label=f'N1={n1}'))
        plt.legend(handles=handles)
        
    plt.plot(rlist, veff_gr_new, label='Gr')
    plt.xlim(2, 10)
    plt.ylim(-1, 0.5)
    plt.grid()
    plt.show()
    # Plot for each N1 only the rs=0.14 and the rs=2 potentials and the gr potentials
    u_list_tmp=[]
    for i in range(cnt):
        if rs_list[i] not in [rs_list[2], rs_list[-2]]:
            continue
        color = f"C{N1_list[i] % 10}"
        L_squared = -2 * u(4, N1_list[i], coeffs[i], rs_list[i]) * 16
        u_list_new = 2 * u(rlist, N1_list[i], coeffs[i], rs_list[i]) + (L_squared / (rlist**2))
        if L_squared < 0:
            continue
        if rs_list[i] == rs_list[2]:
            plt.plot(rlist, u_list_new, color=color, linestyle='--', linewidth=2)
            u_list_tmp=u_list_new
        elif rs_list[i] ==  rs_list[-2]:
            plt.plot(rlist, u_list_new, color=color, linestyle='-.', linewidth=2)
            plt.fill_between(rlist, u_list_new,  u_list_tmp, color=color, alpha=0.3)
        handles = []
        for n1 in set(N1_list):
            color = f"C{n1 % 10}"
            handles.append(plt.Line2D([0], [0], color=color, label=f'N1={n1}'))
        plt.legend(handles=handles)
        plt.figtext(0.15, 0.85, f"'--' line is rs={rs_list[2]:.2f}\n'-.'' line is rs={rs_list[-2]:.2f}", fontsize=10, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
    plt.plot(rlist, veff_gr_new, label='Gr')
    plt.xlim(2, 10)
    plt.ylim(-1, 0.5)
    plt.grid()
    plt.title(f'u(r) for rs={rs_list[2]:.2f} and rs={rs_list[-2]:.2f}')
    plt.show()
    return
def generate_combinations():
        all_combinations = []
        for r in range(4, 14):
            for comb in combinations(range(14), r):
                if comb[0] == 0:
                    if 9 in comb or 10 in comb or 13 in comb:
                        continue
                    else:
                        all_combinations.append(comb)
        return all_combinations
def semi_old(N1,rs,conds=range(0,14),save=False,load=False,show=False,prec=True,Isco=True):

    new_folder_path, new_folder_path_drafts, load_folder_path, new_json_folder_path, load_json_folder_path, imagepath=set_log_today()

    if save and load:
        print("You can't save and load at the same time")
        return    
    t0=time.time()
    today_time = datetime.today().strftime('%Y%m%d_%H%M%S')   
    if load:
        loadname= input('Enter the date abd time of the json file to load from(yyyymmdd_hhmmss): ')
        loaddate=loadname[:8]
        load_details_file_path = os.path.join(load_folder_path+loaddate+'\\', f"{loadname}_detailes_readme.txt")
        with open(load_details_file_path, 'r') as details_file:
            details = details_file.read()
            print(details)
    # b_global=np.array([1,0,2,(1/3),3,0,4,2,9,36,(2/9),(5/27),2,-1/9])
    # b=b_global[conds]
    condsstring=''.join([str(i) for i in conds])
    N=len(conds)
    N2=N-N1
    if load:
        N1=int(input('Enter the number of alphas: '))
        N=int(input('Enter the number of alphas: '))
        N2=N-N1
    run_data=f"The number of conditions is {N}, there are {N1} alphas and {N2} betas"
    lim1=10**8
    lim2=10**-7   
    if save:
        x,data=Solve_coeffs(N1,rs,conds)
        with open(new_json_folder_path+'\\'+today_time+'coeffs_json.json', 'w') as outfile:
             json.dump(data, outfile)
    if load:
        with open(load_json_folder_path+loaddate+'\\'+'Jsons'+'\\'+loadname+'coeffs_json.json') as json_file:
            data = json.load(json_file)
            x = np.array(data['x'])
    if not save and not load:
        x,data=Solve_coeffs(N1,rs,conds)

########################################################################################### tests ##########################################################################################################
    rsstring=f'{rs}:.f2'.replace('.','_')
    if prec:
    #region plotting the effective potentail for L=4
        if show:
            data_for_save_1=plot_effctive_potential(rs,N1,x,type='MSCO')
            if save:
                plt.savefig(f'{imagepath}\\Effective_potential_plot_{today_time}.png')
            plt.show()
            
        #save plot as image
        if not show and save:
            data_for_save_1=plot_effctive_potential(rs,N1,x,type='MSCO')
            plt.savefig(f'{imagepath}\\Effective_potential_plot_'+today_time+'.png')
            plt.close()
        #endregion
     
     
     #region calculating and plotting the precession for parabolic orbits

        
        g1=0.5
        g2=3000
        f1=100
        f2=1000

        temp_files_1=new_folder_path_drafts+'\\'+f'tmp_prec_data_json_{N}_{N1}_{condsstring}_{g1}_{g2}_{f1}_{f2}_{rsstring}.json'
        temp_files_2=new_folder_path_drafts+'\\'+f'tmp_periapsis_data_json_{N}_{N1}_{condsstring}_{g1}_{g2}_{f1}_{f2}_{rsstring}.json'
        temp_files_exist = os.path.exists(temp_files_1) and os.path.exists(temp_files_2)
        
        if not load:
            if  not temp_files_exist or save:
                Gr_Rp_list,Pn_Rp_list,Pw_Rp_list,Pwegg_Rp_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,L1,L2=calculate_precession(rs,N1,x,g1,g2,f1,f2,lim1,lim2)

                # # # # #save to json
                data = {}
                data['Gr_parb_prec_list'] = Gr_parb_prec_list
                data['Pn_parb_prec_list'] = Pn_parb_prec_list
                data['Pw_parb_prec_list'] = Pw_parb_prec_list
                data['Pwegg_parb_prec_list'] = Pwegg_parb_prec_list
                data['L1'] = L1.tolist()
                data['L2'] = L2.tolist()


                data_rp={}
                data_rp['Gr_Rp_list'] = Gr_Rp_list
                data_rp['Pn_Rp_list'] = Pn_Rp_list
                data_rp['Pw_Rp_list'] = Pw_Rp_list
                data_rp['Pwegg_Rp_list'] = Pwegg_Rp_list

                if save:
                    with open(new_json_folder_path+'\\'+today_time+'prec_data_json.json', 'w') as outfile:
                        json.dump(data, outfile)
                    with open(new_json_folder_path+'\\'+today_time+'periapsis_data_json.json', 'w') as outfile:
                        json.dump(data_rp, outfile)
                else:
                    with open(new_folder_path_drafts+'\\'+f'tmp_prec_data_json_{N}_{N1}_{condsstring}_{g1}_{g2}_{f1}_{f2}_{rsstring}.json', 'w') as outfile:
                        json.dump(data, outfile)
                    with open(new_folder_path_drafts+'\\'+f'tmp_periapsis_data_json_{N}_{N1}_{condsstring}_{g1}_{g2}_{f1}_{f2}_{rsstring}.json', 'w') as outfile:
                        json.dump(data_rp, outfile)

        
        if temp_files_exist and not save and not load:
            with open(temp_files_1) as json_file:
                data = json.load(json_file)
                Gr_parb_prec_list = data['Gr_parb_prec_list']
                Pn_parb_prec_list = data['Pn_parb_prec_list']
                Pw_parb_prec_list = data['Pw_parb_prec_list']
                Pwegg_parb_prec_list = data['Pwegg_parb_prec_list']
                L1 = np.array(data['L1'])
                L2 = np.array(data['L2'])
            with open(temp_files_2) as json_file:
                data_rp = json.load(json_file)
                Gr_Rp_list = data_rp['Gr_Rp_list']
                Pn_Rp_list = data_rp['Pn_Rp_list']
                Pw_Rp_list = data_rp['Pw_Rp_list']
                Pwegg_Rp_list = data_rp['Pwegg_Rp_list']  



        # load from json
        if load:
            with open(load_json_folder_path+loaddate+'\\'+'Jsons'+'\\'+loadname+'prec_data_json.json') as json_file:
                data = json.load(json_file)
                Gr_parb_prec_list = data['Gr_parb_prec_list']
                Pn_parb_prec_list = data['Pn_parb_prec_list']
                Pw_parb_prec_list = data['Pw_parb_prec_list']
                Pwegg_parb_prec_list = data['Pwegg_parb_prec_list']

            with open(load_json_folder_path+loaddate+'\\'+'Jsons'+'\\'+loadname+'periapsis_data_json.json') as json_file:
                data_rp = json.load(json_file)
                Gr_Rp_list = data_rp['Gr_Rp_list']
                Pn_Rp_list = data_rp['Pn_Rp_list']
                Pw_Rp_list = data_rp['Pw_Rp_list']
                Pwegg_Rp_list = data_rp['Pwegg_Rp_list']
    #plotting the precession
        if show:
            data_for_save_3,mgr,mpn,mpw,mwegg=pressecion_plots(L1,L2,Gr_Rp_list,Pn_Rp_list,Pw_Rp_list,Pwegg_Rp_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,g1,g2,f1,f2,type=2)
            if save:
                plt.savefig(f'{imagepath}\\Precession_parabolic_near_L=4_{today_time}.png')
            plt.show()
            if save:
                plt.close()
            pressecion_plots(L1,L2,Gr_Rp_list,Pn_Rp_list,Pw_Rp_list,Pwegg_Rp_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,g1,g2,f1,f2,type=1)
            if save:
                plt.savefig(f'{imagepath}\\Rp_error_{today_time}.png')
            plt.show()
            if save:
                plt.close()
            data_for_save_4,mgr2,mpn2,mpw2,mwegg2,bgr,bpn,bpw,bwegg=pressecion_plots(L1,L2,Gr_Rp_list,Pn_Rp_list,Pw_Rp_list,Pwegg_Rp_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,g1,g2,f1,f2,type=3)
            if save:
                plt.savefig(f'{imagepath}\\Precession_far_from_L=4_{today_time}.png')
            plt.show()
            if save:
                plt.close()
            pressecion_plots(L1,L2,Gr_Rp_list,Pn_Rp_list,Pw_Rp_list,Pwegg_Rp_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,g1,g2,f1,f2,type=4)
            if save:
                plt.savefig(f'{imagepath}\\Precession_difference_{today_time}.png')
            plt.show()
            if save:
                plt.close()
            pressecion_plots(L1,L2,Gr_Rp_list,Pn_Rp_list,Pw_Rp_list,Pwegg_Rp_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,g1,g2,f1,f2,type=5)
            if save:
                plt.savefig(f'{imagepath}\\Precession_Rp_{today_time}.png')
            plt.show()
            if save:
                plt.close()
        if not show and save:
            data_for_save_3,mgr,mpn,mpw,mwegg=pressecion_plots(L1,L2,Gr_Rp_list,Pn_Rp_list,Pw_Rp_list,Pwegg_Rp_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,g1,g2,f1,f2,type=2)
            plt.savefig(f'{imagepath}\\Precession_parabolic_near_L=4_{today_time}.png')
            plt.close()
            pressecion_plots(L1,L2,Gr_Rp_list,Pn_Rp_list,Pw_Rp_list,Pwegg_Rp_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,g1,g2,f1,f2,type=1)
            plt.savefig(f'{imagepath}\\Rp_error_{today_time}.png')
            plt.close()
            data_for_save_4,mgr2,mpn2,mpw2,mwegg2,bgr,bpn,bpw,bwegg=pressecion_plots(L1,L2,Gr_Rp_list,Pn_Rp_list,Pw_Rp_list,Pwegg_Rp_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,g1,g2,f1,f2,type=3)
            plt.savefig(f'{imagepath}\\Precession_far_from_L=4_{today_time}.png')
            plt.close()
            pressecion_plots(L1,L2,Gr_Rp_list,Pn_Rp_list,Pw_Rp_list,Pwegg_Rp_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,g1,g2,f1,f2,type=4)
            plt.savefig(f'{imagepath}\\Precession_difference_{today_time}.png')
            plt.close()
            pressecion_plots(L1,L2,Gr_Rp_list,Pn_Rp_list,Pw_Rp_list,Pwegg_Rp_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,g1,g2,f1,f2,type=5)
            plt.savefig(f'{imagepath}\\Precession_Rp_{today_time}.png')
            plt.close()
        if not save:    
            print(data_for_save_3)
            print(data_for_save_4)
    #endregion

    if Isco:
    #region plotting the effective potentail for rp=6:
        if show:
            Lgrisco,Lpnisco,Lpwisco,Lweggisco=plot_effctive_potential(rs,N1,x,type='ISCO')
            if save:
                plt.savefig(f'{imagepath}\\Effective_potential_ISCO_{today_time}.png')
            plt.show()
            
        if not show and save:
            Lgrisco,Lpnisco,Lpwisco,Lweggisco=plot_effctive_potential(rs,N1,x,type='ISCO')
            plt.savefig(f'{imagepath}\\Effective_potential_ISCO_{today_time}.png')
            plt.close()
        #endregion
        #region plotting the ISCO trajectory.
        if show:  
            data_for_save_2=plot_ISCO_trajectory(rs,N1,x,[Lgrisco,Lpnisco,Lpwisco,Lweggisco])
            if save:
                plt.savefig(f'{imagepath}\\ISCO_trajectory_and_f_of_r_{today_time}.png')
            plt.show()
        if not show and save:
            data_for_save_2=plot_ISCO_trajectory(rs,N1,x,[Lgrisco,Lpnisco,Lpwisco,Lweggisco])
            plt.savefig(f'{imagepath}\\ISCO_trajectory_and_f_of_r_{today_time}.png')
            plt.close()

    if not prec:
        g1=88
        g2=88
        f1=88
        f2=88
        data_for_save_1='Precission data was not calculated'
        data_for_save_3='Precission data was not calculated'
        data_for_save_4='Precission data was not calculated'
    if not Isco:
        data_for_save_2='ISCO data was not calculated'
    ttt=(time.time()-t0)/60
    temp_files_3=new_folder_path_drafts+'\\'+f'tmp_detailes_readme_{N}_{N1}_{condsstring}_{g1}_{g2}_{f1}_{f2}.txt'
    temp_files_exist2 = os.path.exists(temp_files_3)
    if not save and temp_files_exist2:
        with open(temp_files_3, 'r') as details_file:
            print(details_file.read())
# Create a text file with details
    coeff_string_list1=','.join([str(i) for i in x[:N1]])
    coeff_string_list2=','.join([str(i) for i in x[N1:]])
    if save or not temp_files_exist2:
        if save:
            details_file_path = os.path.join(new_folder_path, f"{today_time}_detailes_readme.txt")
        else:
            # details_file_path = temp_files_3
            details_file_path = temp_files_3
        with open(details_file_path, 'w') as details_file:
            details_file.write("Details of the Run\n")
            details_file.write("==================\n\n")
            details_file.write("MetaData:\n")
            details_file.write(f"N1={N1}, N2={N2}, N={N}, conds={conds},rs={rs}\n\n")
            details_file.write("alpha values:\n")
            details_file.write(f"[{coeff_string_list1}]\n\n")
            details_file.write("beta values:\n")
            details_file.write(f"[{coeff_string_list2}]\n\n")
            details_file.write("Run Data:\n")
            details_file.write(f"{run_data}\n\n")
            details_file.write(f"Integration limitations: iterations number={lim1}, absolute error threshhold={lim2}\n")
            details_file.write("============== precession test ==============\n")
            details_file.write(data_for_save_1+'\n')
            details_file.write(f"L is taken from 4 to {g1}, with {g2} points\n")
            details_file.write(f"and also is taken from {g1+4} to {f1}, with {f2} points\n")
            if g1/g2>10**-3:
                details_file.write('Warning:The step size is too big, the results may be inaccurate\n')
            details_file.write(data_for_save_3+'\n')
            details_file.write(data_for_save_4+'\n')
            details_file.write("============== ISCO trajectory ==============\n")
            details_file.write("Angular momentum and energy values:\n")
            details_file.write(data_for_save_2+'\n')
            details_file.write(f"Total time taken: {ttt:.3f} m\n")
            details_file.write("The conditions taken into account:\n")
            for i in conds:
                details_file.write(conditions_text_list[i]+'\n')
            details_file.write("\n\n")
    return x





#endregion

#endregion

#region New functions
def Gr_precession_gen_2(rp,lim1=10**8,lim2=10**-7):
    L=np.sqrt(2*(rp**2)/(rp-2))
    def tmp1(r):
        if  np.isclose(((0+1)**2)-(1-(2/r))*(1+((L**2)/(r**2))),0) and ((0+1)**2)-(1-(2/r))*(1+((L**2)/(r**2)))<0:
            return 0
        if r==rp:
            return 0
        else:
            return((0+1)**2)-(1-(2/r))*(1+((L**2)/(r**2)))



    def integrand(r):
        return L/((r**2)*(tmp1(r))**(1/2))
    
    # print("parabolic orbit")
    res=2*sp.integrate.quad(integrand,rp,np.inf,limit=lim1,epsabs=lim2, epsrel=lim2)[0]-2*np.pi
    # if np.isnan(res):
    #     print('oh no gr')
    #     res= mci(integrand,Rp,1000,n=10**7)
    return res,L
def Pn_precession_gen_2(rp,rs,x,N1,lim1=10**8,lim2=10**-7):
    L=np.sqrt((rp**2)*(-2*u(rp,N1,x,rs)))
    if (-2*u(rp,N1,x,rs))<0:
        return 0,-1
    def tmp1(r):
        if  np.isclose(-2*u(r,N1,x,rs)-((L**2)/(r**2)),0) and -2*u(r,N1,x,rs)-((L**2)/(r**2))<0:
            return 0
        if r==rp:
            return 0
        else:
            return -2*u(r,N1,x,rs)-((L**2)/(r**2))

    def integrand(r):
        return L/((r**2)*(tmp1(r))**(1/2))
    
    res=2*sp.integrate.quad(integrand,rp,np.inf,limit=lim1*10,epsabs=lim2, epsrel=lim2)[0]-2*np.pi

    return res,L    
def Pw_precession_gen_2(rp,lim1=10**8,lim2=10**-7):
    L=np.sqrt(2*(rp**2)*(-u_pw(rp)))
    if -u_pw(rp)<0:
        return 0,-1
    def integrand(r):
        return L/((r**2)*((-2*u_pw(r)-((L**2)/(r**2)))**(1/2)))
    res=2*sp.integrate.quad(integrand,rp,np.inf,limit=lim1,epsabs=lim2, epsrel=lim2)[0]-2*np.pi
    return res,L
def Pwegg_precession_gen_2(rp,lim1=10**8,lim2=10**-7):
    L=np.sqrt(2*(rp**2)*(-u_wegg(rp)))
    if -u_wegg(rp)<0:
        return 0,-1
    def integrand(r):
        return L/((r**2)*((-2*u_wegg(r)-((L**2)/(r**2)))**(1/2)))
    res=2*sp.integrate.quad(integrand,rp,np.inf,limit=lim1,epsabs=lim2, epsrel=lim2)[0]-2*np.pi
    # if np.isnan(res):
    #     print('oh no pwegg')
    #     res= mci(integrand,Rp,1000,n=10**7)
    return res,L
def calculate_precession_2(rs,N1,x,g1,g2,f1,f2,lim1=10**8,lim2=10**-7):
        Gr_L_list=[]
        Pn_L_list=[]
        Pw_L_list=[]
        Pwegg_L_list=[]
        Gr_parb_prec_list=[]
        Pn_parb_prec_list=[]
        Pw_parb_prec_list=[]
        Pwegg_parb_prec_list=[]
        if (g1/g2)>10**-3:
            print('The step size is too big, the results may be inaccurate')
        rp_wegg=2*(np.sqrt(6)-1)
        rp=4
        

        rp1= np.linspace(rp,rp+g1,g2)
        rp2= np.linspace(rp+g1,f1,f2)
        rplist=np.concatenate((rp1,rp2))
        rp_wegg1= np.linspace(rp_wegg,rp_wegg+g1,g2)
        rp_wegg2= np.linspace(rp_wegg+g1,f1,f2)
        rplist_wegg=np.concatenate((rp_wegg1,rp_wegg2))


        for i,pri in enumerate(rplist):
            p,l=Gr_precession_gen_2(pri,lim1,lim2)
            Gr_parb_prec_list.append(p)
            Gr_L_list.append(l)
            p,l=Pn_precession_gen_2(pri,rs,x,N1,lim1=lim1,lim2=lim2)
            Pn_parb_prec_list.append(p)
            Pn_L_list.append(l)
            p,l=Pw_precession_gen_2(pri,lim1,lim2)
            Pw_parb_prec_list.append(p)
            Pw_L_list.append(l)
            p,l=Pwegg_precession_gen_2(rplist_wegg[i],lim1,lim2)
            Pwegg_parb_prec_list.append(p)
            Pwegg_L_list.append(l)
            if i%500==0:
                progress = (i + 1) / len(rplist)
                bar_length = 40
                block = int(round(bar_length * progress))
                text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}%"
                print(text, end='', flush=True)
        return Gr_L_list,Pn_L_list,Pw_L_list,Pwegg_L_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,rp1,rp2,rp_wegg1,rp_wegg2
def pressecion_plots_2(rp1,rp2,rp_wegg1,rp_wegg2,Gr_L_list,Pn_L_list,Pw_L_list,Pwegg_L_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,g1,g2,f1,f2,type=1):
    rp=np.concatenate((rp1,rp2))
    rp_wegg=np.concatenate((rp_wegg1,rp_wegg2))
    if type==1: 
        plt.plot(rp,np.array(Gr_L_list)**2,'k-',label='Gr')
        plt.plot(rp,(np.array(Pn_L_list))**2,'b*',label='Pn',markersize=2)
        plt.plot(rp,(np.array(Pw_L_list))**2,'g-.',label='Pw')
        plt.plot(rp_wegg,(np.array(Pwegg_L_list))**2,'r--',label='Pwegg')
        plt.legend()
        plt.grid()
        plt.xlabel('Rp')
        plt.ylabel('L^2')
        plt.title('L^2 of potential ')
        plt.xscale('log')
        plt.yscale('log')
        return
    if type==2:
        plt.plot(Gr_L_list[:g2]-Gr_L_list[0], np.array(Gr_parb_prec_list[:g2]), 'k-', label='Gr')
        plt.plot(Pn_L_list[:g2]-Pn_L_list[0], np.array(Pn_parb_prec_list[:g2]), 'b*', label='Pn',markersize=2)
        plt.plot(Pw_L_list[:g2]-Pn_L_list[0], np.array(Pw_parb_prec_list[:g2]), 'g-.', label='Pw')
        plt.plot(Pwegg_L_list[:g2]-Pn_L_list[0], np.array(Pwegg_parb_prec_list[:g2]), 'r--', label='Pwegg')
        plt.xscale('log')
        plt.ylim(0,16)
        plt.xlim(10**(-5),10**(-1))
        plt.legend()
        plt.grid()
        plt.title('Precession parabolic near L=4')
        plt.xlabel('L(rp)-L(4)')
        plt.ylabel('Precession')
        return 
    if type==3:
        plt.plot(Gr_L_list[g2:],np.array(Gr_parb_prec_list[g2:]),'k-',label='Gr')
        plt.plot(Pn_L_list[g2:],(np.array(Pn_parb_prec_list[g2:])),'b*',label='Pn',markersize=2)
        plt.plot(Pw_L_list[g2:],np.array(Pw_parb_prec_list[g2:]),'g-.',label='Pw')
        plt.plot(Pwegg_L_list[g2:],np.array(Pwegg_parb_prec_list[g2:]),'r--',label='Pwegg')

        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(10**(-4),8)
        plt.legend()
        plt.grid()
        plt.title('Precession parabolic nfar from L=4')
        return
    if type==4:  
        plt.plot(Gr_L_list-Gr_L_list[0],((np.array(Pn_parb_prec_list)-np.array(Gr_parb_prec_list)))/np.array(Gr_parb_prec_list),'b*',label='Pn_error',markersize=2)
        plt.plot(Pn_L_list-Pn_L_list[0],(np.array(Pw_parb_prec_list)-np.array(Gr_parb_prec_list))/np.array(Gr_parb_prec_list),'g-.',label='Pw_error')
        plt.plot(Pw_L_list-Pw_L_list[0],(np.array(Pwegg_parb_prec_list)-np.array(Gr_parb_prec_list))/np.array(Gr_parb_prec_list),'r--',label='Pwegg_error')
        plt.plot(Gr_L_list-Gr_L_list[0],np.zeros(len((Gr_L_list))),'k',label='0')
        plt.legend()
        plt.xscale('log')
        lower_ylim=-1
        upper_ylim=0.5
        plt.ylim(lower_ylim,upper_ylim)

        plt.grid()
        plt.title('Precession difference precentage parabolic near L=4')
        plt.xlabel('L-L(4)')
        plt.ylabel('Precession error')
        return
    if type==5:
        Rp_wegg = 2 * (np.sqrt(6) - 1)
        plt.plot(np.array(rp[1:g2]), (np.array(Pn_parb_prec_list[1:g2]) - np.array(Gr_parb_prec_list[1:g2])) / np.array(Gr_parb_prec_list[1:g2]), 'b*', label='Pn_error',markersize=2)
        plt.plot(np.array(rp[1:g2]), (np.array(Pw_parb_prec_list[1:g2]) - np.array(Gr_parb_prec_list[1:g2])) / np.array(Gr_parb_prec_list[1:g2]), 'g-.', label='Pw_error')
        plt.plot(np.array(rp_wegg[1:g2]), (np.array(Pwegg_parb_prec_list[1:g2]) - np.array(Gr_parb_prec_list[1:g2])) / np.array(Gr_parb_prec_list[1:g2]), 'r--', label='Pwegg_error')
        plt.plot(np.array(rp[1:g2]), np.zeros(len(rp[1:g2])), 'k-', label='0')
        plt.legend()
        plt.grid()
        plt.ylim(-0.4, 0.05)
        plt.title('Precession Error Percentage near L=4')
        plt.xlabel('Rp')
        plt.ylabel('Precession Error')
        return
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
    save=False
    conds_list=''.join([str(i) for i in conds])
    save_json_title = f'N1_{N1}_rs_{rs:.2f}_conds_{"".join(map(str, conds))}_range_{rangelist[0]}_{rangelist[1]:.0e}_{rangelist[2]}_{rangelist[3]:.0e}'.replace('+', '_').replace('.', '')
    folder_path_plots = f"C:\\Users\\itama\\Documents\\.venv\\Scripts\\Research scripts\\PNP_research_assets\\Papper_final_plots"
    folder_path_jsons = f"C:\\Users\\itama\\Documents\\.venv\\Scripts\\Research scripts\\PNP_research_assets\\Papper_final_jsons"
    if not os.path.exists(folder_path_plots):
        os.makedirs(folder_path_plots, exist_ok=True)
    if not os.path.exists(folder_path_jsons):
        os.makedirs(folder_path_jsons, exist_ok=True)

    if not os.path.exists(folder_path_jsons + '\\' + save_json_title + '.json'):
        save = True

    if save:
         x=Solve_coeffs(N1,rs,conds)[0]
         r_L_list,Pn_L_list,Pw_L_list,Pwegg_L_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,rp1,rp2,rp_wegg1,rp_wegg2=calculate_precession_2(rs,N1,x,rangelist[0],rangelist[1],rangelist[2],rangelist[3])
         data = {
                'r_L_list': r_L_list,
                'Pn_L_list': Pn_L_list,
                'Pw_L_list': Pw_L_list,
                'Pwegg_L_list': Pwegg_L_list,
                'Gr_parb_prec_list': Gr_parb_prec_list,
                'Pn_parb_prec_list': Pn_parb_prec_list,
                'Pw_parb_prec_list': Pw_parb_prec_list,
                'Pwegg_parb_prec_list': Pwegg_parb_prec_list,
                'rp1': rp1.tolist(),
                'rp2': rp2.tolist(),
                'rp_wegg1': rp_wegg1.tolist(),
                'rp_wegg2': rp_wegg2.tolist(),
                'N1': N1,
                'rs': rs,
                'conds': conds,
                'rangelist': rangelist,
                'coeffs': x.tolist()
                }
         with open(folder_path_jsons + '\\' + save_json_title + '.json', 'w') as outfile:
            json.dump(data, outfile)
    else:
        with open(folder_path_jsons + '\\' + save_json_title + '.json', 'r') as outfile:
            data = json.load(outfile)
            r_L_list = np.array(data['r_L_list'])
            Pn_L_list = np.array(data['Pn_L_list'])
            Pw_L_list = np.array(data['Pw_L_list'])
            Pwegg_L_list = np.array(data['Pwegg_L_list'])
            Gr_parb_prec_list = np.array(data['Gr_parb_prec_list'])
            Pn_parb_prec_list = np.array(data['Pn_parb_prec_list'])
            Pw_parb_prec_list = np.array(data['Pw_parb_prec_list'])
            Pwegg_parb_prec_list = np.array(data['Pwegg_parb_prec_list'])
            rp1 = np.array(data['rp1'])
            rp2 = np.array(data['rp2'])
            rp_wegg1 = np.array(data['rp_wegg1'])
            rp_wegg2 = np.array(data['rp_wegg2'])
            x = np.array(data['coeffs'])
    save1 = False
    save2 = False
    save3 = False
    save4 = False
    save5 = False
    if not os.path.exists(folder_path_plots + '\\' + save_json_title + 'rp_L_plot.png'):
        save1 = True
    if not os.path.exists(folder_path_plots + '\\' + save_json_title + 'precession_plot_near.png'):
        save2 = True
    if not os.path.exists(folder_path_plots + '\\' + save_json_title + 'precession_plot_far.png'):
        save3 = True
    if not os.path.exists(folder_path_plots + '\\' + save_json_title + 'precession_plot_error.png'):
        save4 = True
    if not os.path.exists(folder_path_plots + '\\' + save_json_title + 'precession_plot_Rp.png'):
        save5 = True

    # pressecion_plots_2(rp1,rp2,rp_wegg1,rp_wegg2,r_L_list,Pn_L_list,Pw_L_list,Pwegg_L_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,rangelist[0],rangelist[1],rangelist[2],rangelist[3],type=1)
    # if  save1:
    #     plt.savefig(folder_path_plots + '\\' + save_json_title + 'rp_L_plot.png')
    # # plt.show()
    # plt.close()
    # pressecion_plots_2(rp1,rp2,rp_wegg1,rp_wegg2,r_L_list,Pn_L_list,Pw_L_list,Pwegg_L_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,rangelist[0],rangelist[1],rangelist[2],rangelist[3],type=2)
    # if  save2:
    #     plt.savefig(folder_path_plots + '\\' + save_json_title + 'precession_plot_near.png')
    # # plt.show()
    # plt.close()
    # pressecion_plots_2(rp1,rp2,rp_wegg1,rp_wegg2,r_L_list,Pn_L_list,Pw_L_list,Pwegg_L_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,rangelist[0],rangelist[1],rangelist[2],rangelist[3],type=3)
    # if save3:
    #     plt.savefig(folder_path_plots + '\\' + save_json_title + 'precession_plot_far.png')    
    # # plt.show()
    # plt.close()
    # pressecion_plots_2(rp1,rp2,rp_wegg1,rp_wegg2,r_L_list,Pn_L_list,Pw_L_list,Pwegg_L_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,rangelist[0],rangelist[1],rangelist[2],rangelist[3],type=4)
    # if  save4:
    #     plt.savefig(folder_path_plots + '\\' + save_json_title + 'precession_plot_error.png')
    # # plt.show()
    # plt.close()
    # pressecion_plots_2(rp1,rp2,rp_wegg1,rp_wegg2,r_L_list,Pn_L_list,Pw_L_list,Pwegg_L_list,Gr_parb_prec_list,Pn_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,rangelist[0],rangelist[1],rangelist[2],rangelist[3],type=5)
    # if  save5:
    #     plt.savefig(folder_path_plots + '\\' + save_json_title + 'precession_plot_Rp.png')
    # # plt.show()
    # plt.close()

    return


# Define the combinations and rs values
combinations = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 4, 5, 6, 7, 12],
    [0, 3, 4, 5, 6, 7, 8, 9, 10],
    [0, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 12],
    [0, 1, 2, 3, 12, 13],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    [0,1,4,5,6,7,8,12],
    [0,1,4,5,6,7,8,9,12],
    [0,1,2,3,4,12,13]
]
rs_values = np.linspace(1, 2, 10)
rangelist = [0.5, 2000, 200, 2000]

# Loop through combinations and rs values
# for comb in combinations:
#     for rs in rs_values:
#         for N1 in range(len(comb) + 1):
#             try:
#                 print(f"Running new_main for N1={N1}, rs={rs}, conds={comb}")
#                 new_main(N1=N1, rs=rs, conds=comb, rangelist=rangelist)
#             except Exception as e:
#                 print(f"Error encountered for N1={N1}, rs={rs}, conds={comb}")
#                 print(traceback.format_exc())
#                 continue
#endregion


######trying stuff
def plot_3d_potential(rs, N1, conds):
    x = Solve_coeffs(N1, rs, conds)[0]
    r = np.linspace(rs+0.5, 50, 500)
    L = np.linspace(4, 30, 500)
    R, L_grid = np.meshgrid(r, L)
    potential = 2 * u(R, N1, x, rs) + ((L_grid**2) / (R**2))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(R, L_grid, potential, cmap='viridis', edgecolor='none')

    # Plot black dots where potential is zero
    zero_points = np.where(np.isclose(potential, 0, atol=1e-3))
    ax.scatter(R[zero_points], L_grid[zero_points], potential[zero_points], color='black', s=1)

    ax.set_title('3D Plot of pn')
    ax.set_xlabel('r')
    ax.set_ylabel('L')
    ax.set_zlabel('Potential')
    plt.show()

def plot_3d_gr_potential():
    r = np.linspace(1, 50, 500)
    L = np.linspace(4, 30, 500)
    R, L_grid = np.meshgrid(r, L)
    potential = (1 - (2 / R)) * (1 + ((L_grid**2) / (R**2))) - 1

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(R, L_grid, potential, cmap='plasma', edgecolor='none')

    # Plot black dots where potential is zero
    zero_points = np.where(np.isclose(potential, 0, atol=1e-3))
    ax.scatter(R[zero_points], L_grid[zero_points], potential[zero_points], color='black', s=1)

    ax.set_title('3D Plot of gr')
    ax.set_xlabel('r')
    ax.set_ylabel('L')
    ax.set_zlabel('Potential')
    plt.show()

def plot_combined_3d_potentials_wegg():
    rp_wegg = 2 * (np.sqrt(6) - 1)
    r = np.linspace(2, 50, 500)
    L = np.linspace(4, 30, 500)
    R, L_grid = np.meshgrid(r, L)

    # GR potential
    gr_potential = (1 - (2 / R)) * (1 + ((L_grid**2) / (R**2))) - 1

    # Wegg potential
    wegg_potential = 2 * u_wegg(R) + ((L_grid**2) / (R**2))

    fig = plt.figure(figsize=(14, 7))

    # GR potential plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(R, L_grid, gr_potential, cmap='plasma', edgecolor='none')
    zero_points_gr = np.where(np.isclose(gr_potential, 0, atol=1e-3))
    ax1.scatter(R[zero_points_gr], L_grid[zero_points_gr], gr_potential[zero_points_gr], color='black', s=1)
    ax1.set_title('3D Plot of GR Potential')
    ax1.set_xlabel('r')
    ax1.set_ylabel('L')
    ax1.set_zlabel('Potential')
    ax1.set_zlim(-3, 30)

    # Wegg potential plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(R, L_grid, wegg_potential, cmap='viridis', edgecolor='none')
    zero_points_wegg = np.where(np.isclose(wegg_potential, 0, atol=1e-3))
    ax2.scatter(R[zero_points_wegg], L_grid[zero_points_wegg], wegg_potential[zero_points_wegg], color='black', s=1)
    ax2.set_title('3D Plot of Wegg Potential')
    ax2.set_xlabel('r')
    ax2.set_ylabel('L')
    ax2.set_zlabel('Potential')
    ax2.set_zlim(-3, 30)

    plt.tight_layout()
    plt.show()

def plot_combined_3d_potentials_pn_gr(rs, N1, conds):
    x = Solve_coeffs(N1, rs, conds)[0]
    r = np.linspace(rs + 0.5, 50, 500)
    L = np.linspace(4, 30, 500)
    R, L_grid = np.meshgrid(r, L)

    # GR potential
    gr_potential = (1 - (2 / R)) * (1 + ((L_grid**2) / (R**2))) - 1

    # PN potential
    pn_potential = 2 * u(R, N1, x, rs) + ((L_grid**2) / (R**2))

    fig = plt.figure(figsize=(14, 7))

    # GR potential plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(R, L_grid, gr_potential, cmap='plasma', edgecolor='none')
    zero_points_gr = np.where(np.isclose(gr_potential, 0, atol=1e-3))
    ax1.scatter(R[zero_points_gr], L_grid[zero_points_gr], gr_potential[zero_points_gr], color='black', s=1)
    ax1.set_title('3D Plot of GR Potential')
    ax1.set_xlabel('r')
    ax1.set_ylabel('L')
    ax1.set_zlabel('Potential')
    ax1.set_zlim(-3, 30)

    # PN potential plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(R, L_grid, pn_potential, cmap='viridis', edgecolor='none')
    zero_points_pn = np.where(np.isclose(pn_potential, 0, atol=1e-3))
    ax2.scatter(R[zero_points_pn], L_grid[zero_points_pn], pn_potential[zero_points_pn], color='black', s=1)
    ax2.set_title('3D Plot of PN Potential')
    ax2.set_xlabel('r')
    ax2.set_ylabel('L')
    ax2.set_zlabel('Potential')
    ax2.set_zlim(-3, 30)

    plt.tight_layout()
    plt.show()


def plot_combined_angular_velocity_heatmap_pn_gr_wegg(rs, N1, conds):
    x = Solve_coeffs(N1, rs, conds)[0]
    r = np.linspace(3, 2000, 10000)
    L = np.linspace(4, 50, 500)
    R, L_grid = np.meshgrid(r, L)

    # GR angular velocity
    gr_potential = (1 - (2 / R)) * (1 + ((L_grid**2) / (R**2))) - 1
    gr_angular_velocity = L_grid / ((R**2) * np.sqrt(-gr_potential))

    # PN angular velocity
    pn_potential = 2 * u(R, N1, x, rs) + ((L_grid**2) / (R**2))
    pn_angular_velocity = L_grid / ((R**2) * np.sqrt(-pn_potential))

    # Wegg angular velocity
    wegg_potential = 2 * u_wegg((R)) + ((L_grid**2) / ((R)**2))
    wegg_angular_velocity = L_grid / (((R)**2) * np.sqrt(-wegg_potential))

     # Wegg moved angular velocity
    wegg_potential = 2 * u_wegg((R-1)) + ((L_grid**2) / ((R-1)**2))
    wegg_angular_velocity = L_grid / (((R-1)**2) * np.sqrt(-wegg_potential))
    zero_points_gr = np.where(np.isclose(gr_potential, 0, atol=1e-5))
    zero_points_pn = np.where(np.isclose(pn_potential, 0, atol=1e-5))
    zero_points_wegg = np.where(np.isclose(wegg_potential, 0, atol=1e-5))

    L_zeros_gr = L_grid[zero_points_gr]
    L_zeros_pn = L_grid[zero_points_pn]
    L_zeros_wegg = L_grid[zero_points_wegg]
    R_zeros_gr = R[zero_points_gr]
    R_zeros_pn = R[zero_points_pn]
    R_zeros_wegg = R[zero_points_wegg]

    r_zeros_final_gr = []
    r_zeros_final_pn = []
    r_zeros_final_wegg = []
    l_zeros_final_gr = []
    l_zeros_final_pn = []
    l_zeros_final_wegg = []
    for l in L:
        tmp = np.where(L_zeros_gr == l)
        if len(tmp[0]) > 1:
            m = np.mean(R_zeros_gr[tmp[0]])
            r_zeros_final_gr.append(m)
            l_zeros_final_gr.append(l)
        elif len(tmp[0]) == 1:
            r_zeros_final_gr.append(R_zeros_gr[tmp[0][0]])
            l_zeros_final_gr.append(l)
        else:
            r_zeros_final_gr.append(np.nan)
            l_zeros_final_gr.append(np.nan)
        tmp = np.where(L_zeros_pn == l)
        if len(tmp[0]) > 1:
            m = np.mean(R_zeros_pn[tmp[0]])
            r_zeros_final_pn.append(m)
            l_zeros_final_pn.append(l)
        elif len(tmp[0]) == 1:
            r_zeros_final_pn.append(R_zeros_pn[tmp[0][0]])
            l_zeros_final_pn.append(l)
        else:
            r_zeros_final_pn.append(np.nan)
            l_zeros_final_pn.append(np.nan)
        tmp = np.where(L_zeros_wegg == l)
        if len(tmp[0]) > 1:
            m = np.mean(R_zeros_wegg[tmp[0]])
            r_zeros_final_wegg.append(m)
            l_zeros_final_wegg.append(l)
        elif len(tmp[0]) == 1:
            r_zeros_final_wegg.append(R_zeros_wegg[tmp[0][0]])
            l_zeros_final_wegg.append(l)
        else:
            r_zeros_final_wegg.append(np.nan)
            l_zeros_final_wegg.append(np.nan)

    # Wegg moved angular velocity
    wegg_moved_potential = 2 * u_wegg((R - 1)) + ((L_grid**2) / ((R - 1)**2))
    wegg_moved_angular_velocity = L_grid / (((R - 1)**2) * np.sqrt(-wegg_moved_potential))
    zero_points_wegg_moved = np.where(np.isclose(wegg_moved_potential, 0, atol=1e-5))
    L_zeros_wegg_moved = L_grid[zero_points_wegg_moved]
    R_zeros_wegg_moved = R[zero_points_wegg_moved]

    r_zeros_final_wegg_moved = []
    l_zeros_final_wegg_moved = []
    for l in L:
        tmp = np.where(L_zeros_wegg_moved == l)
        if len(tmp[0]) > 1:
            m = np.mean(R_zeros_wegg_moved[tmp[0]])
            r_zeros_final_wegg_moved.append(m)
            l_zeros_final_wegg_moved.append(l)
        elif len(tmp[0]) == 1:
            r_zeros_final_wegg_moved.append(R_zeros_wegg_moved[tmp[0][0]])
            l_zeros_final_wegg_moved.append(l)
        else:
            r_zeros_final_wegg_moved.append(np.nan)
            l_zeros_final_wegg_moved.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 7))

    levelsgr = np.linspace(min(gr_angular_velocity.flatten()), 0.5, 50)
    levelspn = np.linspace(min(pn_angular_velocity.flatten()), max(pn_angular_velocity.flatten()), 50)
    levels_wegg = np.linspace(min(wegg_angular_velocity.flatten()), max(wegg_angular_velocity.flatten()), 50)

    # GR angular velocity contours
    c1 = ax.contour(R, L_grid-4, gr_angular_velocity, levels=levelsgr, cmap='coolwarm', linewidths=1.5)
    plt.colorbar(c1, ax=ax, label='GR Angular Velocity')

    # PN angular velocity contours
    c2 = ax.contour(R, L_grid-4, pn_angular_velocity, levels=levelsgr, cmap='cool', linewidths=1.5, linestyles='--')
    plt.colorbar(c2, ax=ax, label='PN Angular Velocity')

    # Wegg angular velocity contours
    # c3 = ax.contour(R, L_grid-4, wegg_angular_velocity, levels=levelsgr, cmap='autumn', linewidths=1.5, linestyles='-.')
    # plt.colorbar(c3, ax=ax, label='Wegg Angular Velocity')

    # Wegg moved angular velocity contours
    c4 = ax.contour(R, L_grid-4, wegg_moved_angular_velocity, levels=levelsgr, cmap='winter', linewidths=1.5, linestyles=':')
    plt.colorbar(c4, ax=ax, label='Wegg Moved Angular Velocity')

    # Zero potential lines
    ax.plot(r_zeros_final_gr, l_zeros_final_gr, 'k-', label='GR Zero Potential Line')
    ax.plot(r_zeros_final_pn, l_zeros_final_pn, 'r--', label='PN Zero Potential Line')
    # ax.plot(r_zeros_final_wegg, l_zeros_final_wegg, 'g-.', label='Wegg Zero Potential Line')
    ax.plot(r_zeros_final_wegg_moved, l_zeros_final_wegg_moved, 'b:', label='Wegg Moved Zero Potential Line')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Combined Angular Velocity Heatmap for GR, PN, Wegg, and Wegg Moved Potentials')
    ax.set_xlabel('r')
    ax.set_ylabel('L')
    ax.legend()
    ax.grid()

    plt.tight_layout()
    plt.show()

def plot_zero_potential_lines_combined(rs, N1, conds):
            x = Solve_coeffs(N1, rs, conds)[0]
            r = np.linspace(rs + 0.5, 50, 500)
            L = np.linspace(4, 30, 500)
            R, L_grid = np.meshgrid(r, L)

            # GR potential
            gr_potential = (1 - (2 / R)) * (1 + ((L_grid**2) / (R**2))) - 1

            # PN potential
            pn_potential = 2 * u(R, N1, x, rs) + ((L_grid**2) / (R**2))

            # Wegg potential
            wegg_potential = 2 * u_wegg(R) + ((L_grid**2) / (R**2))

            fig, ax = plt.subplots(figsize=(10, 7))

            # Plot zero-potential lines
            gr_zero_points = np.where(np.isclose(gr_potential, 0, atol=1e-3))
            pn_zero_points = np.where(np.isclose(pn_potential, 0, atol=1e-3))
            wegg_zero_points = np.where(np.isclose(wegg_potential, 0, atol=1e-3))

            ax.plot(R[gr_zero_points], L_grid[gr_zero_points], 'r-', label='GR Zero Potential')
            ax.plot(R[pn_zero_points], L_grid[pn_zero_points], 'b--', label='PN Zero Potential')
            ax.plot(R[wegg_zero_points], L_grid[wegg_zero_points], 'g-.', label='Wegg Zero Potential')

            ax.set_title('Zero Potential Lines for GR, PN, and Wegg Potentials')
            ax.set_xlabel('r')
            ax.set_ylabel('L')
            ax.legend()
            ax.grid()

            plt.show()

plot_combined_angular_velocity_heatmap_pn_gr_wegg(1.5, 3, [0, 1, 2, 3, 4, 5, 6, 7, 8])
