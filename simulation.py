# Módulo para a Simulação de Monte Carlo
import numpy as np
from thermal_model import CigreModeloTermico

class MonteCarloSimulator:
    def __init__(self, modelo_termico):
        self.modelo_termico = modelo_termico

    def run_simulation(self, environmental_means, environmental_stds, line_azimuth, current, num_iterations):
        temperaturas_condutor = []

        for _ in range(num_iterations):
            # Amostrar valores aleatórios para Ta, Qs, U, V a partir de distribuições normais
            ta_amostrado = np.random.normal(environmental_means['temperatura_ar'], environmental_stds['temperatura_ar'])
            qs_amostrado = np.random.normal(environmental_means['radiacao_global'], environmental_stds['radiacao_global'])
            u_amostrado = np.random.normal(environmental_means['vento_u'], environmental_stds['vento_u'])
            v_amostrado = np.random.normal(environmental_means['vento_v'], environmental_stds['vento_v'])

            # Reconstruir velocidade e direção do vento
            vento_velocidade_amostrado = np.sqrt(u_amostrado**2 + v_amostrado**2)
            vento_direcao_amostrado = np.degrees(np.arctan2(v_amostrado, u_amostrado))
            # Ajustar a direção para estar entre 0 e 360 graus
            vento_direcao_amostrado = (vento_direcao_amostrado + 360) % 360

            # Calcular o ângulo de ataque do vento em relação ao condutor
            # Simplificação: assumir que o ângulo de ataque é a diferença entre a direção do vento e o azimute da linha
            # O modelo CIGRE 601 tem um cálculo mais detalhado para isso.
            wind_angle = abs(vento_direcao_amostrado - line_azimuth)
            if wind_angle > 180: # Garantir que o ângulo esteja entre 0 e 180
                wind_angle = 360 - wind_angle

            # Chamar o solver numérico para encontrar a temperatura do condutor
            try:
                temp_condutor = self.modelo_termico.solve_conductor_temperature(
                    initial_guess=50, # Pode ser um valor mais inteligente, como a temperatura do ar
                    current=current,
                    solar_rad=qs_amostrado,
                    line_azimuth=line_azimuth,
                    wind_speed=vento_velocidade_amostrado,
                    wind_angle=wind_angle,
                    air_temp=ta_amostrado
                )
                temperaturas_condutor.append(temp_condutor)
            except Exception as e:
                # Lidar com casos onde o solver não converge
                print(f"Aviso: Solver não convergiu para uma iteração. Erro: {e}")
                continue

        return np.array(temperaturas_condutor)