
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
from Calculate_Potential import u, u_pw, u_wegg, Solve_coeffs
import time
import json
import hashlib



#### Calculating using angular momentum - less efficient
def Gr_precession_gen(L,lim1=10**8,lim2=10**-7):
    '''
    Calculates orbital precession for in Schwarzschild metric, parabolic trajectories.'''
    Rp=sp.optimize.fsolve(lambda r: ((1)**2)-(1-(2/r))*(1+((L**2)/(r**2))),4)[0]
    
    
    def tmp1(r):
        if  np.isclose(((1)**2)-(1-(2/r))*(1+((L**2)/(r**2))),0) and ((1)**2)-(1-(2/r))*(1+((L**2)/(r**2)))<0:
            return 0
        if r==Rp:
            return 0
        else:
            return((1)**2)-(1-(2/r))*(1+((L**2)/(r**2)))


    def integrand(r):
        return L / ((r ** 2) * (tmp1(r)) ** 0.5)

    res = 2 * sp.integrate.quad(integrand, Rp, np.inf, limit=lim1, epsabs=lim2, epsrel=lim2)[0] - 2 * np.pi
    return res, Rp

def Pn_precession_gen(L, rs, x, N1, lstep=1, Rp_old=4, lim1=10 ** 8, lim2=10 ** -7):
    """
    Calculates the precession of parabolic orbits for a given pnp potential.
    Parameters:
        L (float): Angular momentum of the orbiting body.
        rs (float): Characteristic scale (e.g., Schwarzschild radius or similar).
        x (float): Additional parameter for the potential function u.
        N1 (float): Parameter for the potential function u.
        lstep (float, optional): Step size for root finding. Default is 1.
        Rp_old (float, optional): Initial guess for periapsis. Default is 4.
        lim1 (float, optional): Integration limit parameter. Default is 1e8.
        lim2 (float, optional): Integration tolerance. Default is 1e-7.
    Returns:
        tuple:
            res (float): Calculated precession (in radians).
            Rp (float): Calculated periapsis radius.
    Notes:
        - The function uses numerical root finding and integration to compute the precession.
        - Returns (0, 0) if the periapsis is unphysical.
    """
    Rp = sp.optimize.fsolve(lambda r: -2 * u(r, N1, x, rs) - ((L ** 2) / (r ** 2)), Rp_old + 2 * (lstep))[0]
    if Rp > 10 ** 7 or Rp < 0:
        return 0, 0

    def tmp1(r):
        val = -2 * u(r, N1, x, rs) - ((L ** 2) / (r ** 2))
        if np.isclose(val, 0) and val < 0:
            return 0
        if r == Rp:
            return 0
        else:
            return val

    def integrand(r):
        return L / ((r ** 2) * (tmp1(r)) ** 0.5)

    res = 2 * sp.integrate.quad(integrand, Rp, np.inf, limit=lim1 * 10, epsabs=lim2, epsrel=lim2)[0] - 2 * np.pi
    return res, Rp

def Pw_precession_gen(L, lim1 =10 ** 8, lim2=10 ** -7):
    '''
    Calculates orbital precession for in PW potential, parabolic trajectories.'''
    Rp = sp.optimize.fsolve(lambda r: -2 * u_pw(r) - ((L ** 2) / (r ** 2)), 4)[0]

    def integrand(r):
        return L / ((r ** 2) * ((-2 * u_pw(r) - ((L ** 2) / (r ** 2))) ** 0.5))

    res = 2 * sp.integrate.quad(integrand, Rp, np.inf, limit=lim1, epsabs=lim2, epsrel=lim2)[0] - 2 * np.pi
    return res, Rp

def Pwegg_precession_gen(L, lim1=10 ** 8, lim2=10 ** -7):
    '''
    Calculates orbital precession for in wegg potential, parabolic trajectories.'''
    Rp = sp.optimize.fsolve(lambda p: (-2 * u_wegg(p) * (p ** 2)) - (L ** 2), 4)[0]

    def integrand(r):
        return L / ((r ** 2) * ((-2 * u_wegg(r) - ((L ** 2) / (r ** 2))) ** 0.5))

    res = 2 * sp.integrate.quad(integrand, Rp, np.inf, limit=lim1, epsabs=lim2, epsrel=lim2)[0] - 2 * np.pi
    return res, Rp

def calculate_precession(rs, N1, x, g1, g2, f1, f2, lim1=10 ** 8, lim2=10 ** -7):
    """
    Calculates the precession of parabolic orbits for various potential models over a specified range of angular momenta.
    This function computes the precession and periapsis distances for four different models (Gr, Pn, Pw, Pwegg) across a concatenated range of angular momentum values. It also tracks computation time and prints progress updates.
    Parameters:
        rs (float): Schwarzschild radius or a characteristic radius for the Pn model.
        N1 (int): Number of alpha values for the Pn model.
        x (float): Additional parameter for the Pn model (context-specific).
        g1 (float): Step size or range parameter for the first segment of angular momentum.
        g2 (int): Number of steps in the first segment of angular momentum (L1).
        f1 (float): End value for the second segment of angular momentum.
        f2 (int): Number of steps in the second segment of angular momentum (L2).
        lim1 (float, optional): Upper limit for numerical integration or solver (default: 1e8).
        lim2 (float, optional): Lower limit for numerical integration or solver (default: 1e-7).
    Returns:
        tuple: A tuple containing the following lists:
            - Gr_Rp_list (list): Periapsis distances for the Gr model.
            - Pn_Rp_list (list): Periapsis distances for the Pn model.
            - Pw_Rp_list (list): Periapsis distances for the Pw model.
            - Pwegg_Rp_list (list): Periapsis distances for the Pwegg model.
            - Gr_parb_prec_list (list): Precession values for the Gr model.
            - Pn_parb_prec_list (list): Precession values for the Pn model.
            - Pw_parb_prec_list (list): Precession values for the Pw model.
            - Pwegg_parb_prec_list (list): Precession values for the Pwegg model.
            - L1 (numpy.ndarray): First segment of angular momentum values.
            - L2 (numpy.ndarray): Second segment of angular momentum values.
    Notes:
        - Prints a warning if the step size is too large, which may affect accuracy.
        - Displays a progress bar in the console during computation.
        - Requires the functions Gr_precession_gen, Pn_precession_gen, Pw_precession_gen, and Pwegg_precession_gen to be defined elsewhere.
        - Assumes numpy and time modules are imported.
    """
    Gr_Rp_list = []
    Pn_Rp_list = []
    Pw_Rp_list = []
    Pwegg_Rp_list = []
    Gr_parb_prec_list = []
    Pn_parb_prec_list = []
    Pw_parb_prec_list = []
    Pwegg_parb_prec_list = []

    if (g1 / g2) > 10 ** -3:
        print('The step size is too big, the results may be inaccurate')

    L1 = np.linspace(4, 4 + g1, g2)
    L2 = np.linspace(4 + g1, f1, f2)
    L = np.concatenate((L1, L2))
    timestep = []

    for i, l in enumerate(L):
        t1 = time.time()
        p, rp = Gr_precession_gen(l, lim1, lim2)
        Gr_parb_prec_list.append(p)
        Gr_Rp_list.append(rp)

        if i == 0:
            p, rp = Pn_precession_gen(l, rs, x, N1, lstep=L[1] - L[0], lim1=lim1, lim2=lim2)
            Pn_parb_prec_list.append(p)
            Pn_Rp_list.append(rp)
        else:
            p, rp = Pn_precession_gen(l, rs, x, N1, lstep=L[1] - L[0], Rp_old=Pn_Rp_list[-1], lim1=lim1, lim2=lim2)
            Pn_parb_prec_list.append(p)
            Pn_Rp_list.append(rp)

        p, rp = Pw_precession_gen(l, lim1, lim2)
        Pw_parb_prec_list.append(p)
        Pw_Rp_list.append(rp)

        p, rp = Pwegg_precession_gen(l, lim1, lim2)
        Pwegg_parb_prec_list.append(p)
        Pwegg_Rp_list.append(rp)

        timestep.append(time.time() - t1)

        if i % 500 == 0:
            progress = (i + 1) / len(L)
            bar_length = 40
            block = int(round(bar_length * progress))
            text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}%"
            print(text, end='', flush=True)

    return (Gr_Rp_list, Pn_Rp_list, Pw_Rp_list, Pwegg_Rp_list,
            Gr_parb_prec_list, Pn_parb_prec_list, Pw_parb_prec_list, Pwegg_parb_prec_list,
            L1, L2)


#### Calculating using periapsis - more efficient
def Gr_precession_gen_2(rp,lim1=10**8,lim2=10**-7):
    '''
    Calculates orbital precession for in Schwarzschild metric, parabolic trajectories.'''
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
    '''
    Calculates the precession of parabolic orbits for a given pnp potential.'''
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
    ''''Calculates orbital precession for in PW potential, parabolic trajectories.'''
    L=np.sqrt(2*(rp**2)*(-u_pw(rp)))
    if -u_pw(rp)<0:
        return 0,-1
    def integrand(r):
        return L/((r**2)*((-2*u_pw(r)-((L**2)/(r**2)))**(1/2)))
    res=2*sp.integrate.quad(integrand,rp,np.inf,limit=lim1,epsabs=lim2, epsrel=lim2)[0]-2*np.pi
    return res,L

def Pwegg_precession_gen_2(rp,lim1=10**8,lim2=10**-7):
    ''''Calculates orbital precession for in wegg potential, parabolic trajectories.'''
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

def calculate_precession_PN(rs,N1,x,g1,g2,f1,f2,lim1=10**6,lim2=10**-6):
        '''
        Calculates precession and angular momentum for the Pn potential over a range of periapsis distances.
        This function computes the precession and angular momentum for the Pn potential across a concatenated range of periapsis distances. The periapsis range is split into two segments, and the results are collected in lists. Progress is printed to the console during computation.
        Parameters
        ----------
        rs : float
            Schwarzschild radius or a characteristic radius for the Pn model.
        N1 : int
            Number of alpha values for the Pn model.    
        x : float
            Additional parameter for the Pn model (context-specific).
        g1 : float
            The end value of the first periapsis segment.
        g2 : int
            The number of points in the first periapsis segment.
        f1 : float
            The end value of the second periapsis segment.
        f2 : int
            The number of points in the second periapsis segment.
        lim1 : float, optional
            The first limit parameter for the precession calculation functions (default is 1e6).
        lim2 : float, optional
            The second limit parameter for the precession calculation functions (default is 1e-6).
        Returns
        -------
        Pn_L_list : list of float
            Angular momentum values for the Pn potential.
        Pn_parb_prec_list : list of float
            Precession values for the Pn potential.
        rplist : numpy.ndarray
            The concatenated periapsis distances for the Pn potential.
        Notes
        -----
        - Prints a warning if the step size is too large.
        - Prints a progress bar to the console during computation.
        - Requires the functions `Pn_precession_gen_2` to be defined.

        '''
        
        Pn_L_list=[]
        Pn_parb_prec_list=[]

        if (g1/g2)>10**-3:
            print('The step size is too big, the results may be inaccurate')
        rp_wegg=2*(np.sqrt(6)-1)
        rp=4
        

        rp1= np.linspace(rp,rp+g1,g2)
        rp2= np.linspace(rp+g1,f1,f2)
        rplist=np.concatenate((rp1,rp2))


        for i,pri in enumerate(rplist):
            p,l=Pn_precession_gen_2(pri,rs,x,N1,lim1=lim1,lim2=lim2)
            Pn_parb_prec_list.append(p)
            Pn_L_list.append(l)
            if i%500==0:
                progress = (i + 1) / len(rplist)
                bar_length = 40
                block = int(round(bar_length * progress))
                text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}%"
                print(text, end='', flush=True)
        return Pn_L_list,Pn_parb_prec_list

def calculate_precession_others(g1,g2,f1,f2,lim1=10**6,lim2=10**-6):
        """
        Calculates precession and angular momentum for different potentials over a range of periapsis distances.
        This function computes the precession and angular momentum for three different potentials (Gr, Pw, Pwegg)
        across a concatenated range of periapsis distances. The periapsis range is split into two segments for each
        potential, and the results are collected in lists. Progress is printed to the console during computation.
        Parameters
        ----------
        g1 : float
            The end value of the first periapsis segment.
        g2 : int
            The number of points in the first periapsis segment.
        f1 : float
            The end value of the second periapsis segment.
        f2 : int
            The number of points in the second periapsis segment.
        lim1 : float, optional
            The first limit parameter for the precession calculation functions (default is 1e6).
        lim2 : float, optional
            The second limit parameter for the precession calculation functions (default is 1e-6).
        Returns
        -------
        Gr_L_list : list of float
            Angular momentum values for the Gr potential.
        Pw_L_list : list of float
            Angular momentum values for the Pw potential.
        Pwegg_L_list : list of float
            Angular momentum values for the Pwegg potential.
        Gr_parb_prec_list : list of float
            Precession values for the Gr potential.
        Pw_parb_prec_list : list of float
            Precession values for the Pw potential.
        Pwegg_parb_prec_list : list of float
            Precession values for the Pwegg potential.
        rp1 : numpy.ndarray
            The first periapsis segment for the Gr and Pw potentials.
        rp2 : numpy.ndarray
            The second periapsis segment for the Gr and Pw potentials.
        rp_wegg1 : numpy.ndarray
            The first periapsis segment for the Pwegg potential.
        rp_wegg2 : numpy.ndarray
            The second periapsis segment for the Pwegg potential.
        Notes
        -----
        - Prints a warning if the step size is too large.
        - Prints a progress bar to the console during computation.
        - Requires the functions `Gr_precession_gen_2`, `Pw_precession_gen_2`, and `Pwegg_precession_gen_2` to be defined.
        - Assumes `numpy` is imported as `np`.
        """
        Gr_L_list=[]
        Pw_L_list=[]
        Pwegg_L_list=[]
        Gr_parb_prec_list=[]
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
        return Gr_L_list,Pw_L_list,Pwegg_L_list,Gr_parb_prec_list,Pw_parb_prec_list,Pwegg_parb_prec_list,rp1,rp2,rp_wegg1,rp_wegg2



#### A function to prevent repeated calculations

##insperation:
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
    folder_path_plots = f"C:\\Users\\itama\\Documents\\.venv\\Scripts\\Research scripts\\PNP_research_assets\\Papper_final_plots"
    folder_path_jsons = f"C:\\Users\\itama\\Documents\\.venv\\Scripts\\Research scripts\\PNP_research_assets\\Papper_final_jsons"
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


def Calculate_all_prec_data(N1, rs, conds, rangelist):
    """
    Calculates and caches precession and coefficient data for given parameters.
    If data exists for the same N1, rs, conds, and rangelist, loads it.
    If data exists for the same N1, rs, conds but different rangelist, prompts user.
    Only calculates missing L values if partial data exists.
    Saves all results in a 'data_dump' folder in the script directory.
    """

    # Prepare folder and filenames
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dump_dir = os.path.join(base_dir, "data_dump")
    os.makedirs(data_dump_dir, exist_ok=True)

    # Create a unique hash for N1, rs, conds
    param_str = f"N1_{N1}_rs_{rs}_conds_{'_'.join(map(str, conds))}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    coeffs_file = os.path.join(data_dump_dir, f"coeffs_{param_hash}.json")
    prec_file = os.path.join(data_dump_dir, f"prec_{param_hash}.json")

    # Helper to flatten rangelist for comparison
    def rangelist_key(r):
        return tuple(float(x) for x in r)

    # Load coefficients if exist, else calculate and save
    if os.path.exists(coeffs_file):
        with open(coeffs_file, "r") as f:
            coeffs_data = json.load(f)
            x = np.array(coeffs_data["coeffs"])
    else:
        x = Solve_coeffs(N1, rs, conds)[0]
        with open(coeffs_file, "w") as f:
            json.dump({"N1": N1, "rs": rs, "conds": conds, "coeffs": x.tolist()}, f)

    # Check if precession data exists
    if os.path.exists(prec_file):
        with open(prec_file, "r") as f:
            prec_data = json.load(f)
        prev_rangelist = prec_data.get("rangelist", None)
        prev_rangelist_key = rangelist_key(prev_rangelist) if prev_rangelist else None
        curr_rangelist_key = rangelist_key(rangelist)
        if prev_rangelist_key == curr_rangelist_key:
            print("Loading existing precession data for these parameters and L vector.")
            return prec_data
        else:
            print("This calculation was made for a different L vector, are you sure you want to do it again? (y/n)")
            ans = input().strip().lower()
            if ans != "y":
                print("Aborting calculation.")
                return prec_data
            # Calculate only for missing L values
            prev_L = np.array(prec_data.get("Pn_L_list", []))
            # Generate new L vector
            rp = 4
            rp1 = np.linspace(rp, rp + rangelist[0], int(rangelist[1]))
            rp2 = np.linspace(rp + rangelist[0], rangelist[2], int(rangelist[3]))
            rplist = np.concatenate((rp1, rp2))
            missing_idx = [i for i, val in enumerate(rplist) if not np.isclose(val, prev_L).any()]
            print(f"Calculating {len(missing_idx)} new L values...")
            new_Pn_L = []
            new_Pn_prec = []
            for i in missing_idx:
                pri = rplist[i]
                p, l = Pn_precession_gen_2(pri, rs, x, N1)
                new_Pn_prec.append(p)
                new_Pn_L.append(l)
            # Append new results to existing
            Pn_L_list = list(prev_L) + new_Pn_L
            Pn_parb_prec_list = list(prec_data.get("Pn_parb_prec_list", [])) + new_Pn_prec
            # Save updated data
            prec_data["Pn_L_list"] = Pn_L_list
            prec_data["Pn_parb_prec_list"] = Pn_parb_prec_list
            prec_data["rangelist"] = rangelist
            with open(prec_file, "w") as f:
                json.dump(prec_data, f)
            return prec_data
    else:
        # No previous data, calculate all
        print("Calculating coefficients and precession for new parameter set.")
        Pn_L_list, Pn_parb_prec_list = calculate_precession_PN(rs, N1, x, rangelist[0], rangelist[1], rangelist[2], rangelist[3])
        data = {
            "N1": N1,
            "rs": rs,
            "conds": conds,
            "coeffs": x.tolist(),
            "Pn_L_list": Pn_L_list,
            "Pn_parb_prec_list": Pn_parb_prec_list,
            "rangelist": rangelist
        }
        with open(prec_file, "w") as f:
            json.dump(data, f)
        return data