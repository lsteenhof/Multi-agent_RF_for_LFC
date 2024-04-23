# Loek Steenhoff
# 22-04-2024
# Main file for the three area power system

import numpy as np
import matplotlib.pyplot as plt


# Time parameters
t0 = 0
N = 100
tau = 2.5
t = np.linspace(t0, N*tau-tau, N)


# Initialize system states
countries = ['nl', 'be', 'de']
system_states = ['angle', 'freq', 'storage']
input_variables = ['P_disp', 'P_ESS_c', 'P_ESS_d']

states = {f"{state}_{country}": np.zeros((N, 1)) for country in countries for state in system_states}
inputs = {f"{input_var}_{country}": np.zeros((N, 1)) for country in countries for input_var in input_variables}


# Set initial conditions for angles and frequencies
initial_conditions = {
    'angle': 30,
    'freq': 50,
}
for country in countries:
    states[f'angle_{country}'][0, 0] = initial_conditions['angle']
    states[f'freq_{country}'][0, 0] = initial_conditions['freq']


# System external inputs
external_inputs = {
    **{f"P_load_nl": np.minimum(((0.12 / np.sqrt(100)) * np.sqrt(t)),0.12)},
    **{f"P_load_be": np.minimum(((0.10 / np.sqrt(100)) * np.sqrt(t)),0.10)},
    **{f"P_load_de": np.minimum(((0.20 / np.sqrt(100)) * np.sqrt(t)),0.20)},
    **{f"tie_lines_{country}": np.zeros((N, 1)) for country in countries},
    **{f"P_ren_{country}": np.zeros((N, 1)) for country in countries}
}

# System parameters
Tp = 25  # Time constant [s]
Kp = 1  # Gain
T_nl_be = 1/(1.75*10**3)  # Tie line length nl - be [km]
T_nl_de = 1/(4.98*10**3)  # Tie line length nl - de [km]
T_be_de = 1/(5.77*10**3)  # Tie line length be - de [km]

mu_c_nl = 0.1  # Charging efficiency of ESS in country nl
mu_c_be = 0.1  # Charging efficiency of ESS in country be
mu_c_de = 0.1  # Charging efficiency of ESS in country de

mu_d_nl = 0.1  # Discharging efficiency of ESS in country nl
mu_d_be = 0.1  # Discharging efficiency of ESS in country be
mu_d_de = 0.1  # Discharging efficiency of ESS in country de

efficiencies = {
    'mu_c': {'nl': mu_c_nl, 'be': mu_c_be, 'de': mu_c_de},
    'mu_d': {'nl': mu_d_nl, 'be': mu_d_be, 'de': mu_d_de},
}

# System dynamics
for i in range(N - 1):  # Loop through each time step, except the last one
    for country in countries:
        # Update angle
        states[f'angle_{country}'][i + 1] = (
                states[f'angle_{country}'][i] + tau * 2 * np.pi * states[f'freq_{country}'][i]
        )

        # Calculate the power balance
        power_balance = (
                inputs[f'P_disp_{country}'][i]
                - external_inputs[f'P_load_{country}'][i]
                + external_inputs[f'P_ren_{country}'][i]
                - external_inputs[f'tie_lines_{country}'][i]
                - inputs[f'P_ESS_c_{country}'][i]
                + inputs[f'P_ESS_d_{country}'][i]
        )

        # Update frequency
        freq_factor = (1 - tau / Tp)
        freq_adjustment = tau * (Kp / Tp) * power_balance
        states[f'freq_{country}'][i + 1] = (
                states[f'freq_{country}'][i] * freq_factor + freq_adjustment
        )

        # Update storage
        storage_charge = tau * efficiencies['mu_c'][country] * inputs[f'P_ESS_c_{country}'][i]
        storage_discharge = tau * efficiencies['mu_d'][country] * inputs[f'P_ESS_d_{country}'][i]
        states[f'storage_{country}'][i + 1] = (
                states[f'storage_{country}'][i] + storage_charge - storage_discharge
        )

    # Update tie line
    external_inputs['tie_lines_nl'][i + 1] = (T_nl_be * (states['angle_nl'][i] - states['angle_be'][i])
                                                 + T_nl_de * (states['angle_nl'][i] - states['angle_de'][i]))
    external_inputs['tie_lines_be'][i + 1] = (T_nl_be * (states['angle_be'][i] - states['angle_nl'][i])
                                                 + T_be_de * (states['angle_be'][i] - states['angle_de'][i]))
    external_inputs['tie_lines_de'][i + 1] = (T_nl_de * (states['angle_de'][i] - states['angle_nl'][i])
                                                 + T_be_de * (states['angle_de'][i] - states['angle_be'][i]))



"""
Section: Plot results
"""
for country in countries:
    plt.plot(t, states[f'angle_{country}'], label=f'Angle {country.upper()}')
plt.title('Angle in each country')
plt.xlabel('Time [s]')
plt.ylabel('Angle [Degrees]')
plt.legend()
plt.grid(True)
plt.show()

# Plot frequencies for each country
for country in countries:
    plt.plot(t, states[f'freq_{country}'], label=f'Frequency {country.upper()}')
plt.title('Frequency in each country')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.legend()
plt.grid(True)
plt.show()

# Plot storage for each country
for country in countries:
    plt.plot(t, states[f'storage_{country}'], label=f'Storage {country.upper()}')
plt.title('Storage in each country')
plt.xlabel('Time [s]')
plt.ylabel('Storage [MWh]' if np.max(states['storage_nl']) > 1 else 'Storage [kWh]')
plt.legend()
plt.grid(True)
plt.show()

for country in countries:
    plt.plot(t, external_inputs[f'P_load_{country}'], label=f'Load {country.upper()}')
plt.title('Load in each country')
plt.xlabel('Time [s]')
plt.ylabel('Load [MW]')
plt.legend()
plt.grid(True)
plt.show()

