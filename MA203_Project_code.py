import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Defining all the parameters
Lx = 0.04      # Length of the brake components in the x-direction
Ly = 0.01      # Length of the brake components in the y-direction
Nx = 50        # Number of spatial grid points in the x-direction
Ny = 10        # Number of spatial grid points in the y-direction
alpha = 1.17e-4   # Thermal diffusivity specific to brake materials
t_simulation = 1.0  # Total simulation time (in seconds)
dx = Lx / Nx  # Length of a single grid point in x-direction
dy = Ly / Ny  # Length of a single grid point in y-direction
dt = 0.001    # Time step 
Q = 5000.0   # Frictional Heat Generation
c = 460      # Specific Heat Capacity of the brake material
ro = 7870   # Specific Density
k = 400 
# Creating a grid for spatial coordinates (x, y)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initializing the temperature array with uniform initial conditions
T0 = 25.0
T_high = 400
T = np.ones((Nx, Ny)) * T0  # Uniform initial temperature (T0 degree Celcius)

# Boundary Conditions
T[:, 0] = T_high  # High temperature at one end
T[:, -1] = T0  # Low temperature at the other end

# Creating a list to store snapshots of temperature over time
temp_ss = []

# A for loop for performing Finite Difference Method
num_time_steps = int(t_simulation / dt)
for step in range(num_time_steps):
    T_new = T.copy()
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            # Updating temp. using the transient heat eqn.
            T_new[i, j] = T[i,j] + dt * alpha * ((T[i+1,j] - 2 * T[i,j] 
                      + T[i-1,j]) / dx**2 + (T[i,j+1] - 2 * T[i,j] 
                      + T[i,j-1]) / dy**2
                      + Q / (alpha * ro * c * (T_high - T0)))
    
    # Update temperature array for the next time step
    T = T_new
    
    # Append a copy of the temperature array to the snapshots list
    temp_ss.append(T.copy())

# Visualization of temperature evolution over time with constant temperature axis scaling
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.clear()
ax.plot_surface(X, Y, temp_ss[1].T, cmap='hot')  # Transpose temp_snapshot
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Temperature (°C)')
ax.set_title(f'Temperature Distribution at Time Step 0')
plt.pause(5)
for i, temp_snapshot in enumerate(temp_ss):
    if i % 10 == 0:  # Plot every 10 time steps for clarity
        ax.clear()
        ax.plot_surface(X, Y, temp_snapshot.T, cmap='hot')  # Transpose temp_snapshot
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Temperature (°C)')
        ax.set_title(f'Temperature Distribution at Time Step {i}')
        plt.pause(0.01)  # Pause for a short time to display the plot
plt.show()

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.contourf(X, Y , temp_ss[0].T, cmap='hot')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.set_title('Initial Temperature Distribution')

ax2.contourf(X, Y, temp_ss[-1].T, cmap='hot')
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')
ax2.set_title('Final Temperature Distribution')

plt.show()

center_temp = []
corner_temp = []

for step in temp_ss:
    center_temp.append(step[Nx//2,Ny//2])
    corner_temp.append(step[0,0])

plt.figure(figsize=(8,6))
plt.plot(np.arange(num_time_steps), center_temp, label='Center Temperature')
plt.plot(np.arange(num_time_steps), corner_temp, label='Corner Temperature')
plt.xlabel('Time Step')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.title('Temperature Evolution at Specific Locations')
plt.grid()
plt.show()

Q_values = [2000.0, 4000.0, 6000.0]
plt.figure(figsize=(10,6))

for Q_val in Q_values:
    T = np.ones((Nx,Ny)) * T0
    temperature_snapshots_Q = []

    for step in range(num_time_steps):
        T_new = T.copy()
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                T_new[i, j] = T[i,j] + dt * alpha * ((T[i+1,j] - 2 * T[i,j] 
                        + T[i-1,j]) / dx**2 + (T[i,j+1] - 2 * T[i,j] 
                        + T[i,j-1]) / dy**2 + Q_val 
                        / (alpha * ro * c * (T_high - T0)))
        T = T_new
        temperature_snapshots_Q.append(T.copy())
    center_temperatures = [step[Nx // 2, Ny // 2] for step in temperature_snapshots_Q]
    plt.plot(np.arange(num_time_steps), center_temperatures, label=f'Q={Q_val}')

plt.xlabel('Time Step')
plt.ylabel('Center Temperature (°C)')
plt.legend()
plt.title('Temperature vs. Heat Generation (Q)')
plt.grid()
plt.show()