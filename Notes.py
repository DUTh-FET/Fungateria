torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

# Define pa, pb, pe as trainable parameters
pa = torch.tensor(0.8, requires_grad=True, device=device)
pb = torch.tensor(0.8, requires_grad=True, device=device)
pe = torch.tensor(3.5, requires_grad=True, device=device)

# Define the optimizer (e.g., Adam)
optimizer = torch.optim.Adam([pa, pb, pe], lr=0.01)

# Main optimization loop
for step in range(100):  # Limiting the number of optimization steps
    # Reset matrices before each simulation run
    u = torch.zeros((grid_size, grid_size), device=device, requires_grad=True)
    v = torch.zeros((grid_size, grid_size), device=device, requires_grad=True)
    c = torch.zeros((grid_size, grid_size), device=device, requires_grad=True)

    # Initialize the state of u, v, c (avoid in-place operations)
    cen_point = int(grid_size / 2)

    for k in range(-cen_span, cen_span + 1):
        for l in range(-cen_span, cen_span + 1):
            if math.sqrt(k * k + l * l) <= cen_span:
                with torch.no_grad():  # Avoid in-place operation
                    u = u.index_put_(
                        (torch.tensor([cen_point + k], device=device),
                         torch.tensor([cen_point + l], device=device)),
                        0.5 * (1 - random_tensor + torch.rand((1, 1), device=device) * random_tensor * 2).flatten()
                    )
                    v = v.index_put_(
                        (torch.tensor([cen_point + k], device=device),
                         torch.tensor([cen_point + l], device=device)),
                        0.1 * (1 - random_tensor + torch.rand((1, 1), device=device) * random_tensor * 2).flatten()
                    )
                    c = c.index_put_(
                        (torch.tensor([cen_point + k], device=device),
                         torch.tensor([cen_point + l], device=device)),
                        0.8 * (1 - random_tensor + torch.rand((1, 1), device=device) * random_tensor * 2).flatten()
                    )

    shifts = []
    weights = []
    for dx in range(-depth, depth + 1):
        for dy in range(-depth, depth + 1):
            if dx == 0 and dy == 0:
                continue  # Skip the center
            distance = (dx**2 + dy**2) ** 0.5
            if distance <= 1:
                weight = 1.0
            else:
                weight = 1.0 / distance
            shifts.append((dx, dy))
            weights.append(weight)

    # Convert weights to a tensor for vectorized operations
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # Run the simulation for a limited number of iterations (adjust for faster feedback)
    for sim_step in range(100):  # Limiting the iterations for faster optimization
        # 1. Calculate Reaction Terms
        f_uv = (pa * u + u ** 2 - pb * u * v) * n
        g_uv = pe * u ** 3 - v

        # 2. Create a binary mask where c > 0.5
        c_positive = (c > 0.5).float()

        # 3. Initialize a list to store all shifted contributions
        shifted_contributions = []

        for idx, (dx, dy) in enumerate(shifts):
            weight = weights[idx]

            # Shift the c_positive mask
            if dx > 0:
                shifted = F.pad(c_positive, (0, 0, dx, 0))[:-dx, :]
            elif dx < 0:
                shifted = F.pad(c_positive, (0, 0, 0, -dx))[-dx:, :]
            else:
                shifted = c_positive.clone()

            if dy > 0:
                shifted = F.pad(shifted, (dy, 0, 0, 0))[:, :-dy]
            elif dy < 0:
                shifted = F.pad(shifted, (0, -dy, 0, 0))[:, -dy:]

            # Multiply by the corresponding weight
            contribution = shifted * weight

            # Append to the list
            shifted_contributions.append(contribution)

        # 4. Stack all contributions and take the element-wise maximum
        if shifted_contributions:
            # Stack along a new dimension and compute max across that dimension
            stacked_contributions = torch.stack(shifted_contributions, dim=0)  # Shape: (num_shifts, H, W)
            ij_mat = torch.max(stacked_contributions, dim=0).values  # Shape: (H, W)
        else:
            ij_mat = torch.zeros_like(c_positive, device=device)

        # 5. For cells where distance <=1, set ij_mat to 1 where c >0.5
        ij_mat = torch.where(c_positive == 1, torch.ones_like(ij_mat), ij_mat)

        # 6. Update ij_new based on neighbor conditions
        pad_ij = F.pad(ij_mat, (1, 1, 1, 1), mode='constant', value=0)
        neighbors_immediate = torch.stack([
            pad_ij[:-2, 1:-1],  # Up
            pad_ij[2:, 1:-1],   # Down
            pad_ij[1:-1, :-2],  # Left
            pad_ij[1:-1, 2:]    # Right
        ], dim=0)
        has_zero_immediate = (neighbors_immediate == 0).any(dim=0)

        neighbors_diagonal = torch.stack([
            pad_ij[:-2, :-2],  # Up-Left
            pad_ij[:-2, 2:],   # Up-Right
            pad_ij[2:, :-2],   # Down-Left
            pad_ij[2:, 2:]     # Down-Right
        ], dim=0)
        has_zero_diagonal = (neighbors_diagonal == 0).any(dim=0)

        ij_new = ij_mat.clone()
        ij_new = torch.where(
            (ij_mat > 0) & has_zero_immediate,
            torch.zeros_like(ij_new),
            ij_new
        )
        ij_new = torch.where(
            (ij_mat > 0) & has_zero_diagonal,
            ij_mat / 2.0,
            ij_new
        )

        # 7. Compute Diffusion Using Convolution with Laplacian Kernel L
        conv_u = conv2_same(u, L)
        conv_v = conv2_same(v, L)

        # 8. Update Concentrations
        v_new = v + dt * (d * conv_v + gamma * g_uv)
        u_new = u + dt * (ij_new * (conv_u + gamma * f_uv))

        # 9. Apply Constraints Based on n
        u_new = torch.where(n == -1, torch.zeros_like(u_new), u_new)

        # 10. Compute alpha
        alpha = torch.where(
            u <= threshold,
            torch.tensor(0.49, device=device),
            0.49 - 2.5 * (u - threshold)
        )

        # 11. Update c_new
        c = c + dt * gamma * ph * c * (alpha - c) * (c - 1)

        # 12. Apply Random Noise Where alpha < 0
        noise_condition = (alpha < 0)
        random_noise = torch.rand_like(c) * rnd_subs / 10
        c_new = torch.where(noise_condition, c_new + random_noise, c_new)

        # 13. Ensure u_new and v_new Are Non-Negative and Clamp Values
        u_new = torch.clamp(u_new, min=0, max=amax)
        v_new = torch.clamp(v_new, min=0)
        
        c = torch.where(c>1,1.0,c)
        c = torch.clamp(c_new, max=1)

        # 14. Check for NaNs or Infs
        if torch.isnan(u_new).any() or torch.isnan(v_new).any() or torch.isnan(c_new).any():
            print(f"NaN detected at step {step + 1}. Terminating simulation.")
            break
        if torch.isinf(u_new).any() or torch.isinf(v_new).any() or torch.isinf(c_new).any():
            print(f"Infinity detected at step {step + 1}. Terminating simulation.")
            break

        # 15. Update Variables for the Next Iteration
        u, v, c = u_new, v_new, c_new

        # Compute the loss
        loss = torch.nn.MSELoss()(c, skeleton)

        # Backpropagate the loss to compute gradients
        optimizer.zero_grad()
        loss.backward(retain_graph=False)

        # Update parameters using the optimizer
        optimizer.step()

        if sim_step % 20 == 0:
            print(f"Sim step: {sim_step}, Loss: {loss.item()}")

    # Print the current loss and parameter values
    print(f"Step {step}, Loss: {loss.item()}, pa: {pa.item()}, pb: {pb.item()}, pe: {pe.item()}")