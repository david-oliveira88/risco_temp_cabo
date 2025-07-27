# Módulo com a implementação do modelo térmico CIGRE 601
import numpy as np
import math
from scipy.optimize import fsolve, brentq
import logging
import config

logger = logging.getLogger(__name__)

class CigreModeloTermico:
    """
    Implementação do modelo térmico CIGRE 601 para cálculo da temperatura do condutor.
    """
    
    def __init__(self, parametros_condutor):
        """
        Inicializa o modelo térmico com os parâmetros do condutor.
        
        Args:
            parametros_condutor (dict): Dicionário com parâmetros do cabo
        """
        # Parâmetros do condutor
        self.diametro = parametros_condutor['diametro']  # metros
        self.resistencia_ac_25 = parametros_condutor['resistencia_ac_25']  # ohm/m
        self.resistencia_ac_75 = parametros_condutor['resistencia_ac_75']  # ohm/m
        self.emissividade = parametros_condutor['emissividade']  # adimensional
        self.absortividade = parametros_condutor['absortividade']  # adimensional
        
        # Constantes físicas
        self.sigma = config.STEFAN_BOLTZMANN  # W/m²K⁴
        self.g = config.GRAVIDADE  # m/s²
        
        # Propriedades do ar (dependentes da temperatura)
        self.rho_ar_ref = config.DENSIDADE_AR_PADRAO  # kg/m³ a 20°C
        self.nu_ar_ref = config.VISCOSIDADE_CINEMATICA_AR  # m²/s a 20°C
        self.k_ar_ref = config.CONDUTIVIDADE_TERMICA_AR  # W/m·K a 20°C
        
        logger.info(f"Modelo CIGRE inicializado - Diâmetro: {self.diametro:.4f}m")

    def calcular_resistencia_ac(self, temperatura_condutor):
        """
        Calcula a resistência AC do condutor em função da temperatura.
        
        Args:
            temperatura_condutor (float): Temperatura do condutor em °C
            
        Returns:
            float: Resistência AC em ohm/m
        """
        # Interpolação linear entre 25°C e 75°C
        if temperatura_condutor <= 25:
            return self.resistencia_ac_25
        elif temperatura_condutor >= 75:
            return self.resistencia_ac_75
        else:
            # Interpolação linear
            fator = (temperatura_condutor - 25) / (75 - 25)
            return self.resistencia_ac_25 + fator * (self.resistencia_ac_75 - self.resistencia_ac_25)

    def calcular_aquecimento_joule(self, corrente, temperatura_condutor):
        """
        Calcula o aquecimento Joule por unidade de comprimento.
        
        Args:
            corrente (float): Corrente elétrica em Ampères
            temperatura_condutor (float): Temperatura do condutor em °C
            
        Returns:
            float: Potência de aquecimento Joule em W/m
        """
        r_ac = self.calcular_resistencia_ac(temperatura_condutor)
        return corrente**2 * r_ac

    def calcular_aquecimento_solar(self, radiacao_solar, azimute_linha, 
                                  latitude=None, dia_ano=None, hora_dia=None):
        """
        Calcula o aquecimento solar no condutor.
        
        Args:
            radiacao_solar (float): Radiação solar global em W/m²
            azimute_linha (float): Azimute da linha em graus
            latitude (float): Latitude em graus (opcional para cálculo detalhado)
            dia_ano (int): Dia do ano (opcional)
            hora_dia (float): Hora do dia (opcional)
            
        Returns:
            float: Potência de aquecimento solar em W/m
        """
        # Simplificação: assumir que a radiação solar incide perpendicularmente
        # O modelo completo CIGRE 601 inclui cálculos detalhados de posição solar
        # e fatores de forma, que requerem latitude, data e hora
        
        if latitude is not None and dia_ano is not None and hora_dia is not None:
            # Cálculo detalhado (implementação simplificada)
            fator_forma = self._calcular_fator_forma_solar(
                latitude, dia_ano, hora_dia, azimute_linha
            )
        else:
            # Simplificação: fator de forma constante
            fator_forma = 0.5  # Valor típico para linha horizontal
        
        return self.absortividade * self.diametro * radiacao_solar * fator_forma

    def _calcular_fator_forma_solar(self, latitude, dia_ano, hora_dia, azimute_linha):
        """
        Calcula o fator de forma para radiação solar (simplificado).
        
        Returns:
            float: Fator de forma (0 a 1)
        """
        # Implementação simplificada do cálculo de posição solar
        # Para uma implementação completa, seria necessário usar bibliotecas
        # especializadas como pysolar ou pvlib
        
        # Declinação solar (aproximação)
        declinacao = 23.45 * math.sin(math.radians(360 * (284 + dia_ano) / 365))
        
        # Ângulo horário
        angulo_horario = 15 * (hora_dia - 12)  # graus
        
        # Elevação solar (simplificada)
        lat_rad = math.radians(latitude)
        dec_rad = math.radians(declinacao)
        ha_rad = math.radians(angulo_horario)
        
        sin_elevacao = (math.sin(lat_rad) * math.sin(dec_rad) + 
                       math.cos(lat_rad) * math.cos(dec_rad) * math.cos(ha_rad))
        
        elevacao = math.degrees(math.asin(max(-1, min(1, sin_elevacao))))
        
        # Azimute solar (simplificado)
        cos_azimute = ((math.sin(dec_rad) - math.sin(lat_rad) * sin_elevacao) / 
                      (math.cos(lat_rad) * math.cos(math.radians(elevacao))))
        
        azimute_solar = math.degrees(math.acos(max(-1, min(1, cos_azimute))))
        if angulo_horario > 0:
            azimute_solar = 360 - azimute_solar
        
        # Ângulo de incidência no condutor
        angulo_incidencia = abs(azimute_solar - azimute_linha)
        if angulo_incidencia > 90:
            angulo_incidencia = 180 - angulo_incidencia
        
        # Fator de forma (simplificado)
        if elevacao > 0:
            fator_forma = math.sin(math.radians(elevacao)) * math.cos(math.radians(angulo_incidencia))
            return max(0, fator_forma)
        else:
            return 0

    def calcular_resfriamento_convectivo(self, velocidade_vento, angulo_vento,
                                       temperatura_ar, temperatura_condutor):
        """
        Calcula o resfriamento convectivo usando correlações CIGRE.
        
        Args:
            velocidade_vento (float): Velocidade do vento em m/s
            angulo_vento (float): Ângulo do vento relativo ao condutor em graus
            temperatura_ar (float): Temperatura do ar em °C
            temperatura_condutor (float): Temperatura do condutor em °C
            
        Returns:
            float: Potência de resfriamento convectivo em W/m
        """
        # Propriedades do ar na temperatura filme
        temp_filme = (temperatura_ar + temperatura_condutor) / 2 + 273.15  # K
        
        # Propriedades do ar corrigidas pela temperatura
        rho_ar = self._densidade_ar(temp_filme)
        nu_ar = self._viscosidade_cinematica_ar(temp_filme)
        k_ar = self._condutividade_termica_ar(temp_filme)
        
        # Temperatura absoluta
        T_ar_abs = temperatura_ar + 273.15
        T_c_abs = temperatura_condutor + 273.15
        
        # Componente do vento perpendicular ao condutor
        v_perp = velocidade_vento * math.sin(math.radians(angulo_vento))
        
        # Número de Reynolds
        Re = max(1e-6, v_perp * self.diametro / nu_ar)
        
        # Número de Grashof (convecção natural)
        beta = 1 / temp_filme  # Coeficiente de expansão térmica
        Gr = (self.g * beta * abs(T_c_abs - T_ar_abs) * self.diametro**3) / nu_ar**2
        
        # Número de Prandtl (para ar ≈ 0.7)
        Pr = 0.7
        
        # Determinar regime de convecção e calcular Nusselt
        if v_perp < 0.1:  # Convecção natural dominante
            Nu = self._nusselt_conveccao_natural(Gr, Pr)
        else:  # Convecção forçada
            Nu = self._nusselt_conveccao_forcada(Re, Pr)
            
            # Considerar convecção mista se necessário
            Nu_nat = self._nusselt_conveccao_natural(Gr, Pr)
            Nu = max(Nu, Nu_nat)
        
        # Coeficiente de transferência de calor
        h_c = Nu * k_ar / self.diametro
        
        # Potência de resfriamento convectivo
        return math.pi * self.diametro * h_c * (T_c_abs - T_ar_abs)

    def _nusselt_conveccao_natural(self, Gr, Pr):
        """Calcula número de Nusselt para convecção natural em cilindro."""
        Ra = Gr * Pr
        
        if Ra < 1e-5:
            return 0.4
        elif Ra < 1e3:
            return 0.675 * Ra**0.058
        elif Ra < 1e9:
            return 1.02 * Ra**0.148
        else:
            return 0.85 * Ra**0.188

    def _nusselt_conveccao_forcada(self, Re, Pr):
        """Calcula número de Nusselt para convecção forçada em cilindro."""
        if Re < 0.4:
            return 0.8
        elif Re < 4:
            return 0.821 * Re**0.385
        elif Re < 40:
            return 0.615 * Re**0.466
        elif Re < 4000:
            return 0.174 * Re**0.618
        elif Re < 40000:
            return 0.0239 * Re**0.805
        else:
            return 0.0239 * Re**0.805

    def calcular_resfriamento_radiativo(self, temperatura_ar, temperatura_condutor):
        """
        Calcula o resfriamento radiativo usando lei de Stefan-Boltzmann.
        
        Args:
            temperatura_ar (float): Temperatura do ar em °C
            temperatura_condutor (float): Temperatura do condutor em °C
            
        Returns:
            float: Potência de resfriamento radiativo em W/m
        """
        T_ar_abs = temperatura_ar + 273.15  # K
        T_c_abs = temperatura_condutor + 273.15  # K
        
        return (self.emissividade * self.sigma * math.pi * self.diametro * 
                (T_c_abs**4 - T_ar_abs**4))

    def _densidade_ar(self, temperatura_abs):
        """Calcula densidade do ar em função da temperatura."""
        # Aproximação: rho = rho_ref * (T_ref / T)
        T_ref = 293.15  # 20°C em K
        return self.rho_ar_ref * (T_ref / temperatura_abs)

    def _viscosidade_cinematica_ar(self, temperatura_abs):
        """Calcula viscosidade cinemática do ar em função da temperatura."""
        # Aproximação usando lei de Sutherland
        T_ref = 293.15  # 20°C em K
        S = 110.4  # Constante de Sutherland para ar (K)
        
        return self.nu_ar_ref * ((temperatura_abs / T_ref)**1.5) * ((T_ref + S) / (temperatura_abs + S))

    def _condutividade_termica_ar(self, temperatura_abs):
        """Calcula condutividade térmica do ar em função da temperatura."""
        # Aproximação linear
        T_ref = 293.15  # 20°C em K
        return self.k_ar_ref * (temperatura_abs / T_ref)**0.8

    def equacao_balanco_termico(self, temperatura_condutor, corrente, radiacao_solar,
                               azimute_linha, velocidade_vento, angulo_vento, temperatura_ar):
        """
        Equação de balanço térmico: P_Joule + P_Solar - P_Convectivo - P_Radiativo = 0
        
        Args:
            temperatura_condutor (float): Temperatura do condutor em °C
            corrente (float): Corrente elétrica em A
            radiacao_solar (float): Radiação solar em W/m²
            azimute_linha (float): Azimute da linha em graus
            velocidade_vento (float): Velocidade do vento em m/s
            angulo_vento (float): Ângulo do vento em graus
            temperatura_ar (float): Temperatura do ar em °C
            
        Returns:
            float: Diferença de potência (deve ser zero na solução)
        """
        try:
            # Ganhos de calor
            P_joule = self.calcular_aquecimento_joule(corrente, temperatura_condutor)
            P_solar = self.calcular_aquecimento_solar(radiacao_solar, azimute_linha)
            
            # Perdas de calor
            P_convectivo = self.calcular_resfriamento_convectivo(
                velocidade_vento, angulo_vento, temperatura_ar, temperatura_condutor
            )
            P_radiativo = self.calcular_resfriamento_radiativo(
                temperatura_ar, temperatura_condutor
            )
            
            return P_joule + P_solar - P_convectivo - P_radiativo
            
        except Exception as e:
            logger.error(f"Erro na equação de balanço térmico: {e}")
            return float('inf')

    def resolver_temperatura_condutor(self, estimativa_inicial, corrente, radiacao_solar,
                                    azimute_linha, velocidade_vento, angulo_vento, 
                                    temperatura_ar, metodo='brentq'):
        """
        Resolve a equação de balanço térmico para encontrar a temperatura do condutor.
        
        Args:
            estimativa_inicial (float): Estimativa inicial da temperatura em °C
            corrente (float): Corrente elétrica em A
            radiacao_solar (float): Radiação solar em W/m²
            azimute_linha (float): Azimute da linha em graus
            velocidade_vento (float): Velocidade do vento em m/s
            angulo_vento (float): Ângulo do vento em graus
            temperatura_ar (float): Temperatura do ar em °C
            metodo (str): Método numérico ('brentq', 'fsolve')
            
        Returns:
            float: Temperatura do condutor em °C
        """
        def funcao_objetivo(T_c):
            return self.equacao_balanco_termico(
                T_c, corrente, radiacao_solar, azimute_linha,
                velocidade_vento, angulo_vento, temperatura_ar
            )
        
        try:
            if metodo == 'brentq':
                # Método Brent (mais robusto, requer bracket)
                T_min = temperatura_ar  # Mínimo físico
                T_max = temperatura_ar + 200  # Máximo razoável
                
                # Verificar se há mudança de sinal
                f_min = funcao_objetivo(T_min)
                f_max = funcao_objetivo(T_max)
                
                if f_min * f_max > 0:
                    # Ajustar limites se não há mudança de sinal
                    if f_min > 0:
                        T_min = temperatura_ar - 10
                    else:
                        T_max = temperatura_ar + 300
                
                T_condutor = brentq(funcao_objetivo, T_min, T_max, xtol=0.01)
                
            else:
                # Método fsolve
                resultado = fsolve(funcao_objetivo, estimativa_inicial, xtol=0.01)
                T_condutor = resultado[0]
            
            # Validar resultado
            if T_condutor < temperatura_ar - 10 or T_condutor > temperatura_ar + 300:
                logger.warning(f"Temperatura calculada fora da faixa esperada: {T_condutor:.1f}°C")
            
            return T_condutor
            
        except Exception as e:
            logger.error(f"Erro ao resolver temperatura do condutor: {e}")
            # Retornar estimativa conservadora em caso de erro
            return temperatura_ar + 50

    def calcular_ampacidade(self, temperatura_maxima, radiacao_solar, azimute_linha,
                           velocidade_vento, angulo_vento, temperatura_ar):
        """
        Calcula a ampacidade (corrente máxima) para uma temperatura limite.
        
        Args:
            temperatura_maxima (float): Temperatura máxima permitida em °C
            radiacao_solar (float): Radiação solar em W/m²
            azimute_linha (float): Azimute da linha em graus
            velocidade_vento (float): Velocidade do vento em m/s
            angulo_vento (float): Ângulo do vento em graus
            temperatura_ar (float): Temperatura do ar em °C
            
        Returns:
            float: Ampacidade em A
        """
        # Perdas de calor na temperatura máxima
        P_convectivo = self.calcular_resfriamento_convectivo(
            velocidade_vento, angulo_vento, temperatura_ar, temperatura_maxima
        )
        P_radiativo = self.calcular_resfriamento_radiativo(temperatura_ar, temperatura_maxima)
        
        # Ganho solar
        P_solar = self.calcular_aquecimento_solar(radiacao_solar, azimute_linha)
        
        # Potência Joule disponível
        P_joule_max = P_convectivo + P_radiativo - P_solar
        
        if P_joule_max <= 0:
            logger.warning("Condições ambientais não permitem operação segura")
            return 0
        
        # Calcular corrente correspondente
        R_ac = self.calcular_resistencia_ac(temperatura_maxima)
        ampacidade = math.sqrt(P_joule_max / R_ac)
        
        return ampacidade