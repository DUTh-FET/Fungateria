
dt = 1e-05;

%% Parameters
params = struct(...
    'q',                1.602176634e-19, ...
    'kb',               1.38064852e-23, ...
    'rT',               20e-9, ...
    'rB',               4e-9, ...
    'L',                20e-9, ...
    'g',                1e-9, ...
    'phi_max',          12e-9, ...
    'phi_min',          0.01e-9, ...
    'Tamb',             300, ... 
    'Tau_th',           2.3e-9, ...
    'm',                2000e-9, ...
    'rbulkm',           3e-13, ...
    'rbulkox',          1000, ...
    'gamma',            20e-9, ...
    'el',               28e-9, ...
    'p',                0.5, ...
    'AP',               1.5, ...
    'BP',               1e-4, ...
    'CP',               1.2e-24, ...
    'Edrift',           1.15 * 1.602176634e-19, ...
    'Ediff',            2.5 * 1.602176634e-19, ...
    'alphaP',           1.0, ...
    'Aexp',             0.5, ...
    'Es',               0.8 * 1.602176634e-19, ...
    'phi_sat',          2e-9, ...
    'phi_B',            0.65 * 1.602176634e-19, ...
    'A_star',           1.2e6, ...
    'n_schottky',       1.5, ...
    'lambda_plasticity',200, ...
    'gamma_dose',       1, ...
    'T_trigger',        400, ...
    'T_slope',          20, ...
    'Vmax',             30, ...
    'Km',               1e-15, ...
    'healing_rate',     1 ...
);

% --- Unpack model parameters from struct --- Getting lazy!
q     = params.q;
kb    = params.kb;
rT    = params.rT;
rB    = params.rB;
L     = params.L;
g     = params.g;
phi_max = params.phi_max;
phi_min = params.phi_min;
Tamb  = params.Tamb;
Tau_th = params.Tau_th;
m     = params.m;
rbulkm = params.rbulkm;
rbulkox = params.rbulkox;
gamma  = params.gamma;
el    = params.el;
p     = params.p;
AP    = params.AP;
BP    = params.BP;
CP    = params.CP;
Edrift = params.Edrift;
Ediff  = params.Ediff;
alphaP = params.alphaP;
Aexp  = params.Aexp;
Es    = params.Es;
phi_sat = params.phi_sat;
phi_B = params.phi_B;
A_star = params.A_star;
n_schottky = params.n_schottky;
lambda_plasticity = params.lambda_plasticity;
gamma_dose = params.gamma_dose;
T_trigger = params.T_trigger;
T_slope = params.T_slope;
Vmax  = params.Vmax;
Km    = params.Km;
healing_rate = params.healing_rate;


use_IV_sweep = true; % Set to false for pulse train

if use_IV_sweep
    % --- IV Sweep
    seg_len = 1000;
    %seg_len = 242/4;
    V1 = linspace(0.0, 0.7, seg_len);       % 0 → +0.5
    V2 = linspace(0.7, 0.0, seg_len);       % +0.5 → 0
    V3 = linspace(0.0, -0.7, seg_len);      % 0 → -0.5
    V4 = linspace(-0.7, 0.0, seg_len);      % -0.5 → 0

    V_waveform = [V1, V2, V3, V4, V1, V2, V3, V4]; % Repeat 2x
    t_vec = (0:length(V_waveform)-1) * dt;
else
    % --- Pulse Train
    pulse_amplitude = 0.4;
    pulse_width = 2e-4;
    pulse_period = 4e-4;
    num_pulses = 100;

    pulse_steps = round(pulse_width / dt);
    period_steps = round(pulse_period / dt);
    V_waveform = zeros(num_pulses * period_steps, 1);

    for n = 1:num_pulses
        idx_start = (n-1)*period_steps + 1;
        idx_end = idx_start + pulse_steps - 1;
        V_waveform(idx_start:idx_end) = pulse_amplitude;
    end

    t_vec = (0:length(V_waveform)-1) * dt;
end

%% V waveform from t_vec
%t_vec = (0:length(V_waveform)-1) * dt;
V_func = @(t) interp1(t_vec, V_waveform, t, 'previous', 0); % zero outside range

%% Initial State Definition
x0 = [5e-9; 300; 300; 0]; % [phi0, Thot0, Tcold0, dose0]

%% Solve with ode15s
%t_out = t_vec(:);
opts = odeset(MaxStep=dt*2); % tighter tolerances optional 'RelTol',1e-9,'AbsTol',1e-9,
[t_out, x_out] = ode23tb(@(t,x) memristorODE(t, x, V_func, params), [0 t_vec(end)], x0, opts);


phi = x_out(:,1);
Thot = x_out(:,2);
Tcold = x_out(:,3);
dose = x_out(:,4);
V = V_func(t_out);

%% Recompute quantities for plotting
Es = 0.8*q;
Area_eff = pi * (min(phi, 0.2e-9)/2).^2;
S = Es ./ (kb * Thot.^2);

% Geometry-based resistance
rx1 = rT - ((rT - phi/2) ./ (L/2)) .* ((L/2) - g);
rx2 = rB - ((rB - phi/2) ./ (L/2)) .* ((L/2) - g);
rcf = 3e-13 .* (1 + (3/4)*(28e-9 ./ phi) .* (1 - 0.5));
rox = 1000 ./ (1 + 20e-9 * abs(V / g));
Roff = (rcf.*(L-g)./(2*pi*rT.*rx1)) + ...
       (rcf.*(L-g)./(2*pi*rB.*rx2)) + ...
       (rox .* g ./ (pi .* rx1 .* rx2));
Ron = (rcf * L ./ (2*pi*rT.*(phi/2))) + ...
      (rcf * L ./ (2*pi*rB.*(phi/2)));
st_f = @(y, y0) 1 ./ (1 + exp(-y ./ y0));
R = st_f(phi - 12e-9, 1e-9).*Ron + ...
    st_f(0.01e-9 - phi, 1e-9).*Roff;

% Currents
A_star = 1.2e6; phi_B = 0.65*q; n_schottky = 1.5;
J0 = A_star .* Thot.^2 .* exp(-phi_B ./ (kb .* Thot));
J_schottky = J0 .* (exp(q .* V ./ (n_schottky * kb .* Thot)) - 1);
I_schottky = J_schottky .* Area_eff;
I_cf = V ./ R;
I_total = I_cf + I_schottky;

deltaT = Thot - Tcold;

%% Plotting (mimics your Python version)
time_ms = t_out * 1e3;

figure('Position', [100, 100, 800, 900]);
tiledlayout(6,1);

% 1. Voltage
nexttile;
plot(time_ms, V);
title("Input Voltage"); ylabel("V");

% 2. Current breakdown
nexttile;
semilogy(time_ms, abs(I_total), 'r', 'DisplayName', 'Total Current'); hold on;
semilogy(time_ms, abs(I_schottky), 'k--', 'DisplayName', 'Schottky Current');
ylabel("A"); title("Current Breakdown"); legend;

% 3. Resistance
nexttile;
semilogy(time_ms, R);
grid on;
ylabel("Ohm"); title("Resistance from CF Geometry");

% 4. Filament diameter
nexttile;
plot(time_ms, phi);
ylabel("m"); title("Effective Filament Diameter");

% 5. Dose
nexttile;
plot(time_ms, dose);
ylabel("arb. units"); title("Dose Accumulation (Plasticity Trace)");

% 6. Temperature diff
nexttile;
plot(time_ms, deltaT);
xlabel("Time (ms)"); ylabel("K"); title("Temperature Difference (ΔT)");

%% Pulse response (abs current)
if use_IV_sweep
    figure;
    semilogy(V, abs(I_total), 'r', 'LineWidth', 2);
    title("Simulated I–V Characteristic (Log Scale)");
    xlabel("Voltage (V)"); ylabel("Current (A)");
    grid on;
else
    figure('Position', [100, 100, 800, 600]);
    plot(time_ms, abs(I_total), 'r-', 'LineWidth', 2);
    title("Pulse Train Response");
    xlabel("Time (ms)"); ylabel("|Current| (A)");
    grid on; set(gca, 'XMinorGrid', 'on', 'YMinorGrid', 'on');
end

figure('Position', [200, 200, 600, 300]);

% Dual-axis plot
yyaxis right
plot(time_ms, abs(I_total), 'r-', 'LineWidth', 1.5);
ylabel('|I_{total}| (A)');
ylim([1e-9, max(abs(I_total))*1.2]); % Adjust if needed
grid on;

yyaxis left
plot(time_ms, dose, 'b-', 'LineWidth', 1.5);
ylabel('Dose (a.u.)');

xlabel('Time (ms)');
title('Adaptive Current Response and Dose Accumulation');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);

legend({'|I_{total}|', 'Dose'}, 'Location', 'northwest');