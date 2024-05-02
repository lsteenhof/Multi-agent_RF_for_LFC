# Loek Steenhoff
# 02-05-2024
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
Kp_rotating_mass = 1  # Gain
T_nl_be = 1/(1.75*10**3)  # Tie line length nl - be [km]
T_nl_de = 1/(4.98*10**3)  # Tie line length nl - de [km]
T_be_de = 1/(5.77*10**3)  # Tie line length be - de [km]


efficiencies = {
    'mu_c': {'nl': 0.9, 'be': 0.9, 'de': 0.9},
    'mu_d': {'nl': 0.9, 'be': 0.9, 'de': 0.9},
}

# PID parameters
pid_params = {
    'nl': {'Kp': 0.010, 'Ki': 0.001, 'Kd': 0.005},
    'be': {'Kp': 0.012, 'Ki': 0.002, 'Kd': 0.006},
    'de': {'Kp': 0.011, 'Ki': 0.0015, 'Kd': 0.007}
}

# Initialize integral term and previous frequency deviation
integral = {country: 0 for country in countries}
prev_freq_deviation = {country: 50 - states[f'freq_{country}'][0] for country in countries}


def update_dispatch(i, country):
    current_freq = states[f'freq_{country}'][i]
    freq_deviation = 50 - current_freq  # Calculate deviation from nominal frequency

    # Calculate derivative
    derivative = (freq_deviation - prev_freq_deviation[country]) / tau

    # Update integral
    integral[country] += freq_deviation * tau

    # Fetch PID parameters for the country
    Kp = pid_params[country]['Kp']
    Ki = pid_params[country]['Ki']
    Kd = pid_params[country]['Kd']

    # Calculate PID adjustment
    adjustment = (Kp * freq_deviation +
                  Ki * integral[country] +
                  Kd * derivative)

    # Update previous deviation
    prev_freq_deviation[country] = freq_deviation

    # Update dispatch power based on controller output
    inputs[f'P_disp_{country}'][i + 1] = inputs[f'P_disp_{country}'][i] + adjustment


# System dynamics
for i in range(N - 1):  # Loop through each time step, except the last one
    for country in countries:
        update_dispatch(i, country)

    """
    Section: Netherlands (nl)
    """
    # Update angle for nl
    states['angle_nl'][i + 1] = (
            states['angle_nl'][i] + tau * 2 * np.pi * states['freq_nl'][i]
    )

    # Calculate the power balance for nl
    power_balance_nl = (
            inputs['P_disp_nl'][i]
            - external_inputs['P_load_nl'][i]
            + external_inputs['P_ren_nl'][i]
            - external_inputs['tie_lines_nl'][i]
            - inputs['P_ESS_c_nl'][i]
            + inputs['P_ESS_d_nl'][i]
    )

    # Update frequency for nl
    freq_factor_nl = (1 - tau / Tp)
    freq_adjustment_nl = tau * (Kp_rotating_mass / Tp) * power_balance_nl
    states['freq_nl'][i + 1] = (
            states['freq_nl'][i] * freq_factor_nl + freq_adjustment_nl
    )

    # Update storage for nl
    storage_charge_nl = tau * efficiencies['mu_c']['nl'] * inputs['P_ESS_c_nl'][i]
    storage_discharge_nl = tau * efficiencies['mu_d']['nl'] * inputs['P_ESS_d_nl'][i]
    states['storage_nl'][i + 1] = (
            states['storage_nl'][i] + storage_charge_nl - storage_discharge_nl
    )

    """ 
    Section: Belgium (be)
    """
    # Update angle for be
    states['angle_be'][i + 1] = (
            states['angle_be'][i] + tau * 2 * np.pi * states['freq_be'][i]
    )

    # Calculate the power balance for be
    power_balance_be = (
            inputs['P_disp_be'][i]
            - external_inputs['P_load_be'][i]
            + external_inputs['P_ren_be'][i]
            - external_inputs['tie_lines_be'][i]
            - inputs['P_ESS_c_be'][i]
            + inputs['P_ESS_d_be'][i]
    )

    # Update frequency for be
    freq_factor_be = (1 - tau / Tp)
    freq_adjustment_be = tau * (Kp_rotating_mass / Tp) * power_balance_be
    states['freq_be'][i + 1] = (
            states['freq_be'][i] * freq_factor_be + freq_adjustment_be
    )

    # Update storage for be
    storage_charge_be = tau * efficiencies['mu_c']['be'] * inputs['P_ESS_c_be'][i]
    storage_discharge_be = tau * efficiencies['mu_d']['be'] * inputs['P_ESS_d_be'][i]
    states['storage_be'][i + 1] = (
            states['storage_be'][i] + storage_charge_be - storage_discharge_be
    )

    """ 
    Section: Germany (de)
    """
    # Update angle for de
    states['angle_de'][i + 1] = (
            states['angle_de'][i] + tau * 2 * np.pi * states['freq_de'][i]
    )

    # Calculate the power balance for de
    power_balance_de = (
            inputs['P_disp_de'][i]
            - external_inputs['P_load_de'][i]
            + external_inputs['P_ren_de'][i]
            - external_inputs['tie_lines_de'][i]
            - inputs['P_ESS_c_de'][i]
            + inputs['P_ESS_d_de'][i]
    )

    # Update frequency for de
    freq_factor_de = (1 - tau / Tp)
    freq_adjustment_de = tau * (Kp_rotating_mass / Tp) * power_balance_de
    states['freq_de'][i + 1] = (
            states['freq_de'][i] * freq_factor_de + freq_adjustment_de
    )

    # Update storage for de
    storage_charge_de = tau * efficiencies['mu_c']['de'] * inputs['P_ESS_c_de'][i]
    storage_discharge_de = tau * efficiencies['mu_d']['de'] * inputs['P_ESS_d_de'][i]
    states['storage_de'][i + 1] = (
            states['storage_de'][i] + storage_charge_de - storage_discharge_de
    )

    """
    Section: Update tie line
    """
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

