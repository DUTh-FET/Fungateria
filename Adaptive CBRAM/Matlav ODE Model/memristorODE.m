function dxdt = memristorODE(t, x, V_fun, params)

% Unpack state variables
phi = x(1); Thot = x(2); Tcold = x(3); dose = x(4);

% Voltage at this time
V = V_fun(t);

% Constants
% --- Unpack model parameters from struct
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


% Utilities
st_f = @(y, y0) 1 ./ (1 + exp(-y ./ y0));
soft_barrier = @(y, width) 1 ./ (1 + (y ./ width).^2);

% Compute values
S = Es / (kb * Thot^2);
E = V / g;

rx1 = rT - ((rT - phi/2) / (L/2)) * ((L/2) - g);
rx2 = rB - ((rB - phi/2) / (L/2)) * ((L/2) - g);
rcf = rbulkm * (1 + (3/4)*(el/phi)*(1 - p));
rox = rbulkox / (1 + gamma*abs(E));
Roff = (rcf*(L-g)/(2*pi*rT*rx1)) + ...
       (rcf*(L-g)/(2*pi*rB*rx2)) + ...
       (rox * g / (pi * rx1 * rx2));
Ron = (rcf * L / (2*pi*rT*phi/2)) + ...
      (rcf * L / (2*pi*rB*phi/2));
R = st_f(phi - phi_max, 1e-9) * Ron + ...
    st_f(phi_min - phi, 1e-9) * Roff;

% Currents
phi_limit = 0.2e-9;
Area_eff = pi * (min(phi, phi_limit)/2)^2;
J0 = A_star * Thot^2 * exp(-phi_B / (kb * Thot));
J_schottky = J0 * (exp(q*V / (n_schottky*kb*Thot)) - 1);
I_schottky = J_schottky * Area_eff;
I_cf = V / R;
I_total = I_cf + I_schottky;

% Temperature dynamics
kthcold = m * (2 * rT);
kthhot = m * phi;
dTcold = (abs(V * I_total) / kthcold + Tamb / Tau_th - Tcold / Tau_th);
dThot  = (abs(V * I_total) / kthhot + Tamb / Tau_th - Thot / Tau_th);
deltaT = Thot - Tcold;
T = Thot;

% φ dynamics
drift = AP * exp(-Aexp * (Edrift - alphaP * q * V) / (kb * T)) * st_f(V, 0.2) * soft_barrier(phi - phi_max, 1e-2);
diff = BP * (1/phi) * exp(-Ediff / (kb*T)) * (1 + gamma_dose*dose) * exp(-phi / phi_sat);
thermal = CP * S * (1/phi) * deltaT / (L/2) * (1 + lambda_plasticity * dose) * exp(phi_min/phi);
thermal = thermal * (1 + (0.5 + 0.5*tanh((deltaT - T_trigger) / T_slope)));
neg_drift = AP * exp(-Aexp * (Edrift - alphaP*q*abs(V)) / (kb*T)) * ...
            st_f(-V, 0.2) * st_f(phi - phi_max, 1e-9);

dphi = st_f(phi_max - phi, 1e-10) * (drift + diff - thermal) - ...
       st_f(phi - phi_min, 1e-10) * (diff + neg_drift + thermal);

% Dose dynamics
absorbed_energy = abs(V * I_total);
temperature_factor = max((T - Tamb), 0) / Tamb;
dose_rate = (Vmax * absorbed_energy * temperature_factor) / ...
            (Km + absorbed_energy * temperature_factor);
healing_effect = exp(-(Tcold - Tamb) / 10);
ddose = dose_rate - healing_rate * healing_effect * dose;

% Output
dxdt = [dphi; dThot; dTcold; ddose];
end


