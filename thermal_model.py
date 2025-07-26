# Módulo com a implementação do modelo térmico CIGRE
import numpy as np
from scipy.optimize import fsolve

class CigreModeloTermico:
    def __init__(self, parametros_condutor):
        self.diametro = parametros_condutor['diametro']
        self.resistencia_ac_25 = parametros_condutor['resistencia_ac_25']
        self.resistencia_ac_75 = parametros_condutor['resistencia_ac_75']
        self.emissividade = parametros_condutor['emissividade']
        self.absortividade = parametros_condutor['absortividade']

    def _calculate_resistance(self, conductor_temp):
        # Interpolação linear da resistência AC com a temperatura do condutor
        # Assumindo que a resistência varia linearmente entre 25C e 75C
        # R_ac(T_c) = R_ac_25 + (T_c - 25) * (R_ac_75 - R_ac_25) / (75 - 25)
        return self.resistencia_ac_25 + (conductor_temp - 25) * \
               (self.resistencia_ac_75 - self.resistencia_ac_25) / 50

    def _calculate_joule_heating(self, current, conductor_temp):
        # P_J = I^2 * R_ac(T_c)
        r_ac = self._calculate_resistance(conductor_temp)
        return current**2 * r_ac

    def _calculate_solar_heating(self, solar_rad, line_azimuth, conductor_temp):
        # P_S = alpha * D * Q_s * sin(theta)
        # Simplificado por enquanto, theta é o ângulo de incidência solar no condutor
        # Para simplificar, vamos assumir um valor fixo ou uma função simples de azimute
        # O modelo CIGRE 601 detalha o cálculo de theta, que depende da hora, dia, latitude, azimute da linha, etc.
        # Por enquanto, usaremos uma simplificação: Q_s é a radiação solar global, e o fator de forma é 1 (incidência direta)
        # Isso precisará ser refinado com o modelo completo do CIGRE
        return self.absortividade * self.diametro * solar_rad

    def _calculate_convective_cooling(self, wind_speed, wind_angle, air_temp, conductor_temp):
        # P_c = f(wind_speed, wind_angle, air_temp, conductor_temp, D)
        # Este é o componente mais complexo do modelo CIGRE. Requer cálculos de Reynolds, Nusselt, etc.
        # Por enquanto, uma simplificação: convecção natural ou forçada simples
        # Exemplo muito simplificado (apenas para ter um valor inicial)
        h_c = 1.15 * ((conductor_temp - air_temp) / self.diametro)**0.25 # Convecção natural (simplificado)
        if wind_speed > 0.1: # Se houver vento, convecção forçada (simplificado)
            h_c = 0.026 * (wind_speed / self.diametro)**0.6 * (conductor_temp - air_temp)**0.4 # Exemplo
        return np.pi * self.diametro * h_c * (conductor_temp - air_temp)

    def _calculate_radiative_cooling(self, air_temp, conductor_temp):
        # P_r = epsilon * sigma * pi * D * (T_c^4 - T_a^4)
        sigma = 5.67e-8  # Constante de Stefan-Boltzmann
        return self.emissividade * sigma * np.pi * self.diametro * \
               ((conductor_temp + 273.15)**4 - (air_temp + 273.15)**4) # Convertendo para Kelvin

    def _thermal_balance_equation(self, conductor_temp, current, solar_rad, line_azimuth, wind_speed, wind_angle, air_temp):
        p_j = self._calculate_joule_heating(current, conductor_temp)
        p_s = self._calculate_solar_heating(solar_rad, line_azimuth, conductor_temp)
        p_c = self._calculate_convective_cooling(wind_speed, wind_angle, air_temp, conductor_temp)
        p_r = self._calculate_radiative_cooling(air_temp, conductor_temp)
        return p_j + p_s - p_c - p_r

    def solve_conductor_temperature(self, initial_guess, current, solar_rad, line_azimuth, wind_speed, wind_angle, air_temp):
        # Usa fsolve para encontrar a temperatura do condutor que zera a equação de balanço térmico
        temp_condutor = fsolve(self._thermal_balance_equation, initial_guess,
                               args=(current, solar_rad, line_azimuth, wind_speed, wind_angle, air_temp))
        return temp_condutor[0]