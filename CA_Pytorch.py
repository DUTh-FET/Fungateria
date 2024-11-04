import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Set grid size and simulation parameters
grid_size = (100, 100)  # Size of the simulation grid
num_steps = 1000  # Number of simulation steps
ca_update_interval = 10  # Interval for applying CA rules

# Define RD parameters
diffusion_rate_u = 0.16  # Diffusion rate for morphogen u
diffusion_rate_v = 0.08  # Diffusion rate for morphogen v
reaction_rate_f = 0.035  # Reaction rate parameter
reaction_rate_k = 0.065  # Reaction rate parameter
dt = 1.0  # Time step size

# Initialize the RD concentrations for u and v
u = torch.ones(grid_size, dtype=torch.float32)  # Initial concentration of u
v = torch.zeros(grid_size, dtype=torch.float32)  # Initial concentration of v
v[grid_size[0]//2 - 10:grid_size[0]//2 + 10, grid_size[1]//2 - 10:grid_size[1]//2 + 10] = 1.0  # Seed for v

# Initialize grid for cellular automata colors (0: black, 1: green)
color_grid = torch.zeros(grid_size, dtype=torch.int32)

# Define function to apply reaction-diffusion equations
def rd_update(u, v, diffusion_rate_u, diffusion_rate_v, f, k, dt):
    # Calculate Laplacian using finite differences
    laplace_u = (
        -4 * u + torch.roll(u, shifts=1, dims=0) + torch.roll(u, shifts=-1, dims=0) +
        torch.roll(u, shifts=1, dims=1) + torch.roll(u, shifts=-1, dims=1)
    )
    laplace_v = (
        -4 * v + torch.roll(v, shifts=1, dims=0) + torch.roll(v, shifts=-1, dims=0) +
        torch.roll(v, shifts=1, dims=1) + torch.roll(v, shifts=-1, dims=1)
    )
    
    # Gray-Scott reaction-diffusion model equations
    du = diffusion_rate_u * laplace_u - u * v * v + f * (1 - u)
    dv = diffusion_rate_v * laplace_v + u * v * v - (f + k) * v

    # Update concentrations
    u += du * dt
    v += dv * dt

    return u, v

# Cellular automaton color update function
def apply_ca_rules(color_grid, u, v, threshold=0.5):
    new_color_grid = color_grid.clone()
    for x in range(1, color_grid.shape[0] - 1):
        for y in range(1, color_grid.shape[1] - 1):
            current_state = color_grid[x, y]
            morphogen_sum = u[x, y] - v[x, y]  # Simplified decision rule

            # Example rule: if morphogen sum exceeds threshold, flip color
            if current_state == 0 and morphogen_sum > threshold:
                new_color_grid[x, y] = 1  # Flip to green
            elif current_state == 1 and morphogen_sum < threshold:
                new_color_grid[x, y] = 0  # Flip to black

    return new_color_grid

# Visualization function
def visualize(color_grid, step):
    plt.figure(figsize=(6, 6))
    plt.imshow(color_grid.cpu(), cmap='Greens', interpolation='nearest')
    plt.title(f"Step {step}")
    plt.axis('off')
    plt.show()
    clear_output(wait=True)

# Simulation loop
for step in range(num_steps):
    # Run the reaction-diffusion update
    u, v = rd_update(u, v, diffusion_rate_u, diffusion_rate_v, reaction_rate_f, reaction_rate_k, dt)
    
    # Apply CA rules at specified intervals
    if step % ca_update_interval == 0:
        color_grid = apply_ca_rules(color_grid, u, v)
    
    # Visualize at intervals
    if step % 50 == 0:
        visualize(color_grid, step)
