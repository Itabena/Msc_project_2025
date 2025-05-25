import numpy as np

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
b_global=np.array([1,0,2,(1/3),3,0,4,2,9,36,(2/9),(5/27),1,-1/9]) # right-hand side of the system of equations, from cond0 to cond13 left to right.
#region condition functions:
### Conditions for the potential definition, set of linear equations

def c0_alpha(n, rs):
    """
    Condition 1: Covergence to the classical Newtonian potential at large distances
    """
    return 1
def c0_beta(n):
    """
    Condition 1: Covergence to the classical Newtonian potential at large distances
    """
    if n == 0:
        return 1
    else:
        return 0
def c1_alpha(n, rs):
    """
    Condition 2: Reproducing the ISCO at r=6M
    """
    tmp = 6 * (1 / ((6 - rs) ** 3)) * ((6 / (6 - rs)) ** n) * ((rs ** 2) * ((n * (n + 2))) + 6 * (n + 3) * rs - 36)
    return tmp
def c1_beta(n):
    """
    Condition 2: Reproducing the ISCO at r=6M
    """
    return ((n ** 2) - 1) * (6 ** (-n))
def c2_alpha(n, rs):
    """
    Condition 3: Reproducing the same angular momentum of the ISCO L=sqrt(12)M
    """
    tmp = 6 * (1 / ((6 - rs) ** 2)) * ((6 / (6 - rs)) ** n) * (6 + n * rs)
    return tmp
def c2_beta(n):
    """
    Condition 3: Reproducing the same angular momentum of the ISCO L=sqrt(12)M
    """
    return (n + 1) * (6 ** (-n))
def c3_alpha(n, rs):
    """
    Condition 4: Reproducing the photon sphere at r=3M
    """
    tmp = 3 * (1 / ((3 - rs) ** 2)) * ((3 / (3 - rs)) ** n) * (3 + rs * n)
    return tmp
def c3_beta(n):
    """
    Condition 4: Reproducing the photon sphere at r=3M
    """
    return (n + 1) * (3 ** (-n))
def c4_alpha(n, rs):
    """
    Condition 5: Reproducing the same precession as in GR for rp>>rs
    """
    tmp = (n + 1) * rs
    return tmp
def c4_beta(n):
    """
    Condition 5: Reproducing the same precession as in GR for rp>>rs
    """
    if n == 1:
        return 1
    else:
        return 0
def c5_alpha(n, rs):
    """
    Condition 6: Reproducing max point with value 0 for the effective potential at r=4M
    """
    tmp = 4 * (1 / ((4 - rs) ** 2)) * ((4 / (4 - rs)) ** n) * ((n + 2) * rs - 4)
    return tmp
def c5_beta(n):
    """
    Condition 6: Reproducing max point with value 0 for the effective potential at r=4M
    """
    return (n - 1) * (4 ** (-n))
def c6_alpha(n, rs):
    """
    Condition 7: Reproducing the same angular momentum for marginally bound orbit as in gr L=4M
    """
    tmp = 4 * (1 / ((4 - rs) ** 2)) * ((4 / (4 - rs)) ** n) * (4 + n * rs)
    return tmp
def c6_beta(n):
    """
    Condition 7: Reproducing the same angular momentum for marginally bound orbit as in gr L=4M
    """
    return (n + 1) * (4 ** (-n))
def c7_alpha(n, rs):
    """
    Condition 8: Reproducing the divergence of the precession of a parabolic orbit for L->4M
    """
    tmp = 4 * (1 / ((4 - rs) ** 3)) * ((4 / (4 - rs)) ** n) * (-16 + (rs ** 2) * n * (n + 2) + 4 * rs * (n + 3))
    return tmp
def c7_beta(n):
    """
    Condition 8: Reproducing the divergence of the precession of a parabolic orbit for L->4M
    """
    return (4 ** (-n)) * ((n ** 2) - 1)
def c8_alpha(n, rs):
    """
    Condition 9: Equating the 3rd derivative of the effective potential at r=4M to the GR value
    """
    tmp = 2 * ((4 / (4 - rs)) ** n) * (1 / ((4 - rs) ** 4)) * (-384 + rs * (96 * (n + 4) + rs * (-48 + n * (60 + 36 * n + rs * (n - 5) * (n + 2)))))
    return tmp
def c8_beta(n):
    """
    Condition 9: Equating the 3rd derivative of the effective potential at r=4M to the GR value
    """
    return (1 / 2) * (4 ** (-n)) * ((n - 1) * (n + 1) * (n + 6))
def c9_alpha(n, rs):
    """
    Condition 10: Equating the 4th derivative of the effective potential at r=4M to the GR value
    """
    tmp = (1 / ((4 - rs) ** 5)) * ((4 / (4 - rs)) ** n) * (
        -9216 + 2304 * rs * (n + 5) + 576 * (rs ** 2) * (n - 1) * (2 * n + 5) +
        (rs ** 4) * n * (n + 2) * (27 + (n - 8) * n) +
        (rs ** 3) * 16 * (15 + n * (-37 + 4 * (n - 3) * n))
    )
    return tmp
def c9_beta(n):
    """
    Condition 10: Equating the 4th derivative of the effective potential at r=4M to the GR value
    """
    return (1 / 4) * (4 ** (-n)) * (n - 1) * (n + 1) * (36 + n * (n + 10))
def c10_alpha(n, rs):
    """
    Condition 11: Having a minimal point at r=12M for the effective potential
    """
    tmp = 2 * (1 / (12 - rs) ** 2) * ((12 / (12 - rs)) ** n) * (12 + n * rs)
    return tmp
def c10_beta(n):
    """
    Condition 11: Having a minimal point at r=12M for the effective potential
    """
    return (1 / 6) * (n + 1) * (12 ** (-n))
def c11_alpha(n, rs):
    """
    Condition 12: Reproducing the minimal point at r=12M value as the GR one
    """
    tmp = 2 * (1 / (12 - rs)) * ((12 / (12 - rs)) ** n)
    return tmp
def c11_beta(n):
    """
    Condition 12: Reproducing the minimal point at r=12M value as the GR one
    """
    return (1 / 6) * (12 ** (-n))
def c12_alpha(n, rs):
    """
    Condition 13: Equating the 3rd derivative of the effective potential at r=6M to the GR value
    """
    tmp = 6 * (1 / (6 - rs) ** 4) * ((6 / (6 - rs)) ** n) * (
        -1296 + rs * (216 * (n + 4) + rs * (-72 + n * (90 + 54 * n + rs * (n - 5) * (n + 2))))
    )
    return tmp
def c12_beta(n):
    """
    Condition 13: Equating the 3rd derivative of the effective potential at r=6M to the GR value
    """
    return (6 ** (-n)) * (n - 1) * (n + 1) * (n + 6)
def c13_alpha(n, rs):
    """
    Condition 14: Equating the energy of the Isco to the GR value
    """
    tmp = (1 / ((6 - rs) ** 2)) * ((6 / (6 - rs)) ** n) * (rs * (2 + n) - 6)
    return tmp
def c13_beta(n):
    """
    Condition 14: Equating the energy of the Isco to the GR value
    """
    return (1 / 6) * (6 ** (-n)) * (n - 1)
#endregion

def Solve_coeffs(N1, rs, conds):
    """
        Solves for the coefficients of a linear system defined by a set of constraints and basis functions.
        This function constructs a system of linear equations based on the provided constraints and basis functions,
        then solves for the coefficients that satisfy these constraints. The system is constructed by combining
        two sets of basis functions (alpha and beta) for each constraint, and the right-hand side vector is selected
        from a global array according to the specified conditions.
        Parameters
        ----------
        N1 : int
            The number of constraints (or basis functions) of the "alpha" type.
        rs : float
            A parameter passed to the alpha basis functions, typically representing the translation radius of the alpha series.
        conds : array-like of int
            Indices specifying which constraints (rows) to select from the global system. Used to select both the
            right-hand side vector and the rows of the coefficient matrix.
        Returns
        -------
        x : numpy.ndarray
            The solution vector containing the coefficients that satisfy the system of equations.
        data : dict
            A dictionary containing the solution vector under the key 'x' as a list.
        Notes
        -----
        - The function relies on a set of basis functions named `cN_alpha` and `cN_beta` (for N=1 to 13), which must be
          defined elsewhere in the code.
        - The system matrix is constructed by concatenating the outputs of these basis functions for the specified
          number of alpha and beta constraints.
        - The right-hand side vector is selected from a predefined global array `b_global` using the provided `conds`.
        - The function prints a check message with the sizes of the constraint sets for debugging purposes.
        """
    N=len(conds)
    N2=N-N1
    print('Important check!',N,N1,N2)
    b_global=np.array([1,0,2,(1/3),3,0,4,2,9,36,(2/9),(5/27),1,-(1/9)])
    b=b_global[conds]
 


    r0=np.concatenate((np.array([c0_alpha(i,rs) for i in range(N1)]),np.array([c0_beta(i) for i in range(N2)])))
    r1=np.concatenate((np.array([c1_alpha(i,rs) for i in range(N1)]),np.array([c1_beta(i) for i in range(N2)])))
    r2=np.concatenate((np.array([c2_alpha(i,rs) for i in range(N1)]),np.array([c2_beta(i) for i in range(N2)])))
    r3=np.concatenate((np.array([c3_alpha(i,rs) for i in range(N1)]),np.array([c3_beta(i) for i in range(N2)])))
    r4=np.concatenate((np.array([c4_alpha(i,rs) for i in range(N1)]),np.array([c4_beta(i) for i in range(N2)])))
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
    data = {}
    data['x'] = x.tolist()
    return x,data


# def Solve_coeffs(N1, rs, conds):
#     """
#         Solves for the coefficients of a linear system defined by a set of constraints and basis functions.
#         This function constructs a system of linear equations based on the provided constraints and basis functions,
#         then solves for the coefficients that satisfy these constraints. The system is constructed by combining
#         two sets of basis functions (alpha and beta) for each constraint, and the right-hand side vector is selected
#         from a global array according to the specified conditions.
#         Parameters
#         ----------
#         N1 : int
#             The number of constraints (or basis functions) of the "alpha" type.
#         rs : float
#             A parameter passed to the alpha basis functions, typically representing the translation radius of the alpha series.
#         conds : array-like of int
#             Indices specifying which constraints (rows) to select from the global system. Used to select both the
#             right-hand side vector and the rows of the coefficient matrix.
#         Returns
#         -------
#         x : numpy.ndarray
#             The solution vector containing the coefficients that satisfy the system of equations.
#         data : dict
#             A dictionary containing the solution vector under the key 'x' as a list.
#         Notes
#         -----
#         - The function relies on a set of basis functions named `cN_alpha` and `cN_beta` (for N=1 to 13), which must be
#           defined elsewhere in the code.
#         - The system matrix is constructed by concatenating the outputs of these basis functions for the specified
#           number of alpha and beta constraints.
#         - The right-hand side vector is selected from a predefined global array `b_global` using the provided `conds`.
#         - The function prints a check message with the sizes of the constraint sets for debugging purposes.
#         """
#     N=len(conds)
#     N2=N-N1
#     print('Important check!',N,N1,N2)
#     b_global=np.array([1,0,2,(1/3),3,0,4,2,9,36,(2/9),(5/27),1,-(1/9)])
#     b=b_global[conds]
#     tmpvecbeta1=np.concatenate((np.array([1]),np.zeros(N-1)))
#     tmpvecbeta2=np.concatenate((np.array([0,1]),np.zeros(N-2)))


#     r0=np.concatenate((np.ones(N1),tmpvecbeta1[:N2]))
#     r1=np.concatenate((np.array([c1_alpha(i,rs) for i in range(N1)]),np.array([c1_beta(i) for i in range(N2)])))
#     r2=np.concatenate((np.array([c2_alpha(i,rs) for i in range(N1)]),np.array([c2_beta(i) for i in range(N2)])))
#     r3=np.concatenate((np.array([c3_alpha(i,rs) for i in range(N1)]),np.array([c3_beta(i) for i in range(N2)])))
#     r4=np.concatenate((np.array([c4_alpha(i,rs) for i in range(N1)]),tmpvecbeta2[:N2]))
#     r5=np.concatenate((np.array([c5_alpha(i,rs) for i in range(N1)]),np.array([c5_beta(i) for i in range(N2)])))
#     r6=np.concatenate((np.array([c6_alpha(i,rs) for i in range(N1)]),np.array([c6_beta(i) for i in range(N2)])))
#     r7=np.concatenate((np.array([c7_alpha(i,rs) for i in range(N1)]),np.array([c7_beta(i) for i in range(N2)])))
#     r8=np.concatenate((np.array([c8_alpha(i,rs) for i in range(N1)]),np.array([c8_beta(i) for i in range(N2)])))
#     r9=np.concatenate((np.array([c9_alpha(i,rs) for i in range(N1)]),np.array([c9_beta(i) for i in range(N2)])))
#     r10=np.concatenate((np.array([c10_alpha(i,rs) for i in range(N1)]),np.array([c10_beta(i) for i in range(N2)])))
#     r11=np.concatenate((np.array([c11_alpha(i,rs) for i in range(N1)]),np.array([c11_beta(i) for i in range(N2)])))
#     r12=np.concatenate((np.array([c12_alpha(i,rs) for i in range(N1)]),np.array([c12_beta(i) for i in range(N2)])))
#     r13=np.concatenate((np.array([c13_alpha(i,rs) for i in range(N1)]),np.array([c13_beta(i) for i in range(N2)])))

#     A_global=np.array([r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13])
#     A=A_global[conds]
#     # print(A)
#     x=np.linalg.solve(A,b)
#     data = {}
#     data['x'] = x.tolist()
#     return x,data

#region - Potential functions:
def u(r,N1,x,rs):
    """
    Computes a pseudo-Newtonian potential `u` as a sum of two series: a Paczyńsky-Wiita (PW)-like series and a negative powers of r series.

    This function allows for flexible construction of potentials by combining:
    - The PW-like series (terms proportional to r^n / (r - rs)^(n+1)), which generalizes the Paczyńsky & Wiita potential (1980).
    - The negative powers of r series (terms proportional to r^(-n-1)))

    Parameters:
        r (float or np.ndarray): Radial coordinate(s) at which to evaluate the potential.
        N1 (int): Number of terms in the PW-like series.
        x (array-like): Coefficient vector of length N1 + N2, where N2 = len(x) - N1.
            - The first N1 elements correspond to the PW-like series coefficients.
            - The remaining N2 elements correspond to the negative powers series coefficients.
        rs (float): Characteristic radius (e.g., Schwarzschild radius) used in the PW-like terms.

    Returns:
        float or np.ndarray: The value(s) of the pseudo-Newtonian potential at r.

    Notes:
        - If N1 == len(x), only the PW-like series is used (equivalent to a generalized Paczyńsky & Wiita potential).
        - If N1 == 0, only the negative powers series.
        - For intermediate values, both series are combined.
        - upw refers to the Paczyńsky & Wiita potential (1980).
        - u_wgg refers to the potential of Christopher Wegg (2012).
    """
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
    '''
    Computes the Paczyńsky-Wiita potential (1980) for a given radial coordinate `r`.
    The potential is defined as:
        u(r) = -1 / (r - 2)
        This potential is a simplified model for the gravitational field around a black hole.
        Parameters:
            
            r (float or np.ndarray): Radial coordinate(s) at which to evaluate the potential.
            Returns:
            float or np.ndarray: The value(s) of the Paczyńsky-Wiita potential at r.'''
    return -1/(r-2)
def u_wegg(r):
    '''
    Computes the potential of Christopher Wegg (2012) for a given radial coordinate `r`.
    The potential is defined as:
        u(r) = -alph/r - (1-alph)/(r-Rx) - Ry/r^2
        where:
            - alph = -(4/3)*(2+sqrt(6))
            - Rx = (4*sqrt(6))-9
            - Ry = -(4/3)*(-3+2*sqrt(6))
        This potential is a more complex model for the gravitational field around a black hole.
        Parameters:
            r (float or np.ndarray): Radial coordinate(s) at which to evaluate the potential.
        Returns:
            float or np.ndarray: The value(s) of the Wegg potential at r.
    '''
    alph=-(4/3)*(2+np.sqrt(6))
    Rx=(4*np.sqrt(6))-9
    Ry=-(4/3)*(-3+2*np.sqrt(6))
    return -alph/(r)-((1-alph)/(r-Rx))-Ry/(r**2)
def u_dr(r,N1,x,rs):
    '''
    The radial derivative of the potential function `u` defined above.
    This function computes the derivative of the potential with respect to the radial coordinate `r`.'''
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
    '''
    Computes the radial derivative of the Paczyńsky-Wiita potential (1980) for a given radial coordinate `r`.'''
    return 1/((r-2)**2)
def u_wegg_dr(r):
    '''
    Computes the radial derivative of the potential of Christopher Wegg (2012) for a given radial coordinate `r`.'''
    alph=-(4/3)*(2+np.sqrt(6))
    Rx=(4*np.sqrt(6))-9
    Ry=-(4/3)*(-3+2*np.sqrt(6))
    return alph/(r**2)+((1-alph)/(((r-Rx)**2)))+2*Ry/(r**3)
#endregion




