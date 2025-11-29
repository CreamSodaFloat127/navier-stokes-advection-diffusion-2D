# Homework 4 As Cool As It Gets

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import gmres, bicgstab
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
import imageio

# Problem 1

# Part a)

# Initializing Variables

N = 64
L = 20.0
nu = 0.001

x_start = -L/2
x_end = L/2
y_start = x_start
y_end = x_end
delta = L/N

t_start = 0.0
t_end = 40.0
t_span = [t_start, t_end]
delta_t = 0.5
t_eval = np.arange(t_start, t_end + delta_t, delta_t)
delta_t = t_eval[1] - t_eval[0]

# Partial Derivative Matrices from Central Difference

# Matrix A - Laplacian matrix

# A[1,1] = 2 To avoid having a singular matrix
main_diag = -4*np.ones(N**2)
main_diag[0] = 2

A_D = sp.spdiags(data = [main_diag], diags = 0, m = N**2, n = N**2)

boundary_diag_y_lower = np.zeros(N)
boundary_diag_y_lower[0] = 1
boundary_diag_y_lower = np.concatenate([boundary_diag_y_lower]*N)

boundary_diag_y_upper = np.zeros(N)
boundary_diag_y_upper[-1] = 1
boundary_diag_y_upper = np.concatenate([boundary_diag_y_upper]*N)

upper_first_diag = np.ones(N**2)
lower_first_diag = np.ones(N**2)

for i in range(N):
    upper_first_diag[N*i] = 0

for i in range(N):
    lower_first_diag[N*i - 1] = 0

data_y = np.array([upper_first_diag, lower_first_diag, boundary_diag_y_upper, boundary_diag_y_lower])

A_y = sp.spdiags(data = data_y, diags = [1, -1, N-1, -(N-1)], m = N**2, n = N**2)

data_x = np.ones((4, N**2))

A_x = sp.spdiags(data = data_x, diags = [N, -N, N**2 - N, -(N**2 - N)], m = N**2, n = N**2)

A = A_D + A_x + A_y

A = (1/(delta**2))*A

# Matrix B - Partial derivative in x

data_Bx = np.ones((4, N**2))
data_Bx[1] = -np.ones(N**2)
data_Bx[2] = -np.ones(N**2)

B_x = sp.spdiags(data = data_Bx, diags = [N, -N, N**2 - N, -(N**2 - N)], m = N**2, n = N**2)

B = (1/(2*delta))*B_x

# Matrix C - Partial derivative in y

boundary_diag_y_lower = np.zeros(N)
boundary_diag_y_lower[0] = 1
boundary_diag_y_lower = np.concatenate([boundary_diag_y_lower]*N)

boundary_diag_y_upper = np.zeros(N)
boundary_diag_y_upper[-1] = 1
boundary_diag_y_upper = np.concatenate([boundary_diag_y_upper]*N)

data_y = np.array([upper_first_diag, -lower_first_diag, -boundary_diag_y_upper, boundary_diag_y_lower])

C_y = sp.spdiags(data = data_y, diags = [1, -1, N-1, -(N-1)], m = N**2, n = N**2)

C = (1/(2*delta))*C_y

# Matrices to CSC for faster computation

A = A.tocsc()
B = B.tocsc()
C = C.tocsc()

# Initial Condition for Omega at t = 0

def initial_omega(x,y):
    return x*y/100

# Initialize omega_0

x = np.arange(x_start, x_end, delta)
y = np.arange(y_start, y_end, delta)

X,Y = np.meshgrid(x,y)

omega_0_matrix = initial_omega(X,Y)

omega_0 = omega_0_matrix.flatten(order = 'F')

# Fourier Method

k_x = 2*np.pi*np.fft.fftfreq(N, d = delta)
k_y = 2*np.pi*np.fft.fftfreq(N, d = delta)
k_x[0] = 1e-6
k_y[0] = 1e-6

K_X, K_Y = np.meshgrid(k_x, k_y)

reciprocal_K = 1/(K_X**2 + K_Y**2)

def F_Fourier(t, omega):

    omega_matrix = np.reshape(omega, (N,N), order = 'F')

    fft2_omega = np.fft.fft2(omega_matrix)

    fft2_psi = -fft2_omega*reciprocal_K

    psi_matrix = np.real(np.fft.ifft2(fft2_psi))

    psi = psi_matrix.flatten(order = 'F')

    output = (C @ psi)*(B @ omega) - (B @ psi)*(C @ omega) + nu*(A @ omega)

    return output

sol_Fourier = solve_ivp(F_Fourier, t_span = t_span, y0 = omega_0, t_eval = t_eval)

# GIF creator

def plotting_function(t, omega, X, Y, N):
    frames = []
    for j in range(len(t)):
        current_omega = omega[:, j].reshape((N, N), order='F')

        plt.figure(figsize=(5, 4))
        plt.imshow(current_omega, origin = 'lower', cmap = 'coolwarm')
        plt.axis('off')
        
        plt.savefig("frame.png", dpi=100)
        plt.close()

        frames.append(imageio.v2.imread("frame.png"))
    return frames

frames = plotting_function(sol_Fourier.t, sol_Fourier.y, X, Y, N)
imageio.mimsave("shmoovin.gif", frames, duration=delta_t)



# 2D Top Down Heat Map Plotter

"""plt.ion()

fig, ax = plt.subplots(figsize=(10, 8))

# sol_Fourier.y = np.fliplr(sol_Fourier.y) If used iterates backwards in time

for i in range(len(sol_Fourier.y.T)):
    psi = sol_Fourier.y[:, i]

    X, Y = np.meshgrid(x, y)
    Z = np.reshape(psi, (N, N), order = 'F')

    ax.clear()  # Clear previous frame
    plt.imshow(Z, origin = 'lower', cmap='coolwarm')

    ax.set_title(f'2D Plot (t = {t_eval[i]:.3f})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.pause(0.01)   # brief pause lets GUI update

plt.ioff()  # (optional) turn off interactive mode at end
plt.show()
"""
# Plot 3D solution in time

"""plt.ion()

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection='3d')

for i in range(len(t_eval)):
    psi = sol_Fourier.y[:, i]

    X, Y = np.meshgrid(x, y)
    Z = np.flipud(np.reshape(psi, (N, N), order='F'))

    ax.clear()  # Clear previous frame
    ax.plot_surface(X, Y, Z, cmap='coolwarm')

    ax.set_title(f'3D Plot (t = {t_eval[i]:.3f})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.pause(0.01)   # brief pause lets GUI update

plt.ioff()  # (optional) turn off interactive mode at end

plt.show()
"""





