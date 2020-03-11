import numpy as np
import matplotlib.pylab as mpl

def advection1D_Upwind_FVM(a, b, beta, u0, dx, dt, nt, verbosity=False):
    """
    1D Transport Equation: Upwind Finite Volume method

    Periodic boundary conditions are assumed:
       u_1^n = u_{N_x}^n for all time step, n.

    a, b : [a,b] interval
    beta : advection velocity
    u0: array of initial values [u_1^0, u_2^0..., u_{N}^0]
    dx, dt:  x step and t stp
    nt: number of time iterations

    The function returns a list of
    """
    nx = len(u0)
    solution=[u0] # List of solutions at each time step

    # For boundary conditions, we will add two ghost cells to the solution,
    # u[0] and u[nx+1] (while u[1],...,u[nx] are actual unknowns)
    # Here we apply periodic b.c.
    u0 = np.concatenate( ([u0[-1]], u0, [u0[0]]) )

    u = np.empty(nx+2)
    r = beta*dt/dx

    # assert r<1  # If CFL is not verified, throw exception AssertionError
    for n in range(nt):
        if verbosity:
            print(f"Time iteration {n}")
            print(u0)

        #u[1:nx] = u0[1:nx] - r*(u[1:nx] - u[0:nx-1])
        for i in range(1,nx+1): # 1,...,nx
            u[i] = u0[i] - r*(u0[i]-u0[i-1])

        # Save data and prepare next iteration
        u0[0]=u[nx];
        u0[1:nx+1] = u[1:nx+1]
        u0[-1]=u[0]

        solution.append( u[1:nx+1].copy() )

    return solution
