# Módulo para a Simulação de Monte Carlo
import numpy as np
import logging
import config
from thermal_model import CigreModeloTermico
import warnings
from scipy import stats

logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    """
    Classe responsável pela execução da Simulação de Monte Carlo para 
    propagação de incertezas na temperatura do condutor.
    """
    
    def __init__(self, modelo_termico):
        """
        Inicializa o simulador Monte Carlo.
        
        Args:
            modelo_termico (CigreModeloTermico): Instância do modelo térmico
        """
        if not isinstance(modelo_termico, CigreModeloTermico):
            raise TypeError("modelo_termico deve ser uma instância de CigreModeloTermico")
        
        self.modelo_termico = modelo_termico
        self.num_iteracoes_padrao = config.NUM_ITERACOES_MC
        
        logger.info(f"Simulador Monte Carlo inicializado com {self.num_iteracoes_padrao} iterações padrão")

    def executar_simulacao(self, medias_ambientais, desvios_ambientais, azimute_linha,
                          corrente, num_iteracoes=None, metodo_amostragem='normal',
                          semente_aleatoria=None):
        """
        Executa a simulação de Monte Carlo.
        
        Args:
            medias_ambientais (dict): Médias das variáveis ambientais
            desvios_ambientais (dict): Desvios padrão das variáveis ambientais
            azimute_linha (float): Azimute da linha em graus
            corrente (float): Corrente elétrica em A
            num_iteracoes (int): Número de iterações (opcional)
            metodo_amostragem (str): Método de amostragem ('normal', 'lognormal', 'triangular')
            semente_aleatoria (int): Semente para reprodutibilidade (opcional)
            
        Returns:
            dict: Resultados da simulação
        """
        if num_iteracoes is None:
            num_iteracoes = self.num_iteracoes_padrao
        
        if semente_aleatoria is not None:
            np.random.seed(semente_aleatoria)
        
        logger.info(f"Iniciando simulação Monte Carlo com {num_iteracoes} iterações")
        
        # Validar dados de entrada
        self._validar_dados_entrada(medias_ambientais, desvios_ambientais)
        
        # Executar simulação
        resultados = self._executar_loop_simulacao(
            medias_ambientais, desvios_ambientais, azimute_linha, corrente,
            num_iteracoes, metodo_amostragem
        )
        
        # Calcular estatísticas
        estatisticas = self._calcular_estatisticas(resultados['temperaturas'])
        
        # Compilar resultados finais
        resultado_final = {
            'temperaturas': resultados['temperaturas'],
            'estatisticas': estatisticas,
            'iteracoes_validas': resultados['iteracoes_validas'],
            'iteracoes_com_erro': resultados['iteracoes_com_erro'],
            'taxa_sucesso': resultados['iteracoes_validas'] / num_iteracoes,
            'parametros': {
                'num_iteracoes': num_iteracoes,
                'corrente': corrente,
                'azimute_linha': azimute_linha,
                'metodo_amostragem': metodo_amostragem
            }
        }
        
        logger.info(f"Simulação concluída: {resultado_final['iteracoes_validas']}/{num_iteracoes} iterações válidas")
        logger.info(f"Temperatura média: {estatisticas['media']:.2f}°C, "
                   f"P90: {estatisticas['percentil_90']:.2f}°C")
        
        return resultado_final

    def _validar_dados_entrada(self, medias_ambientais, desvios_ambientais):
        """Valida os dados de entrada da simulação."""
        variaveis_obrigatorias = ['temperatura_ar', 'radiacao_global', 'vento_u', 'vento_v']
        
        for var in variaveis_obrigatorias:
            if var not in medias_ambientais:
                raise ValueError(f"Variável '{var}' não encontrada nas médias ambientais")
            if var not in desvios_ambientais:
                raise ValueError(f"Variável '{var}' não encontrada nos desvios ambientais")
            
            # Verificar se valores são numéricos e válidos
            if not np.isfinite(medias_ambientais[var]):
                raise ValueError(f"Média de '{var}' inválida: {medias_ambientais[var]}")
            if not np.isfinite(desvios_ambientais[var]) or desvios_ambientais[var] < 0:
                raise ValueError(f"Desvio padrão de '{var}' inválido: {desvios_ambientais[var]}")

    def _executar_loop_simulacao(self, medias_ambientais, desvios_ambientais, 
                                azimute_linha, corrente, num_iteracoes, metodo_amostragem):
        """Executa o loop principal da simulação Monte Carlo."""
        temperaturas_condutor = []
        iteracoes_com_erro = 0
        
        # Progresso a cada 10% das iterações
        progresso_intervalo = max(1, num_iteracoes // 10)
        
        for i in range(num_iteracoes):
            if i % progresso_intervalo == 0 and i > 0:
                logger.debug(f"Progresso: {i}/{num_iteracoes} ({100*i/num_iteracoes:.0f}%)")
            
            try:
                # Amostrar variáveis ambientais
                variaveis_amostradas = self._amostrar_variaveis_ambientais(
                    medias_ambientais, desvios_ambientais, metodo_amostragem
                )
                
                # Reconstruir velocidade e direção do vento
                vento_info = self._reconstruir_vento(
                    variaveis_amostradas['vento_u'], 
                    variaveis_amostradas['vento_v']
                )
                
                # Calcular ângulo de ataque do vento
                angulo_vento = self._calcular_angulo_vento(vento_info['direcao'], azimute_linha)
                
                # Resolver temperatura do condutor
                temp_condutor = self.modelo_termico.resolver_temperatura_condutor(
                    estimativa_inicial=variaveis_amostradas['temperatura_ar'] + 20,
                    corrente=corrente,
                    radiacao_solar=variaveis_amostradas['radiacao_global'],
                    azimute_linha=azimute_linha,
                    velocidade_vento=vento_info['velocidade'],
                    angulo_vento=angulo_vento,
                    temperatura_ar=variaveis_amostradas['temperatura_ar']
                )
                
                # Validar resultado
                if self._validar_temperatura_resultado(temp_condutor, variaveis_amostradas['temperatura_ar']):
                    temperaturas_condutor.append(temp_condutor)
                else:
                    iteracoes_com_erro += 1
                    
            except Exception as e:
                logger.debug(f"Erro na iteração {i}: {e}")
                iteracoes_com_erro += 1
                continue
        
        return {
            'temperaturas': np.array(temperaturas_condutor),
            'iteracoes_validas': len(temperaturas_condutor),
            'iteracoes_com_erro': iteracoes_com_erro
        }

    def _amostrar_variaveis_ambientais(self, medias, desvios, metodo):
        """
        Amostra valores das variáveis ambientais usando o método especificado.
        
        Args:
            medias (dict): Médias das variáveis
            desvios (dict): Desvios padrão das variáveis
            metodo (str): Método de amostragem
            
        Returns:
            dict: Variáveis amostradas
        """
        variaveis_amostradas = {}
        
        for variavel in ['temperatura_ar', 'radiacao_global', 'vento_u', 'vento_v']:
            media = medias[variavel]
            desvio = desvios[variavel]
            
            if metodo == 'normal':
                valor = np.random.normal(media, desvio)
            elif metodo == 'lognormal':
                # Para radiação solar (sempre positiva)
                if variavel == 'radiacao_global':
                    if media > 0 and desvio > 0:
                        mu = np.log(media**2 / np.sqrt(media**2 + desvio**2))
                        sigma = np.sqrt(np.log(1 + (desvio/media)**2))
                        valor = np.random.lognormal(mu, sigma)
                    else:
                        valor = max(0, np.random.normal(media, desvio))
                else:
                    valor = np.random.normal(media, desvio)
            elif metodo == 'triangular':
                # Distribuição triangular simétrica
                a = media - np.sqrt(6) * desvio
                b = media + np.sqrt(6) * desvio
                valor = np.random.triangular(a, media, b)
            else:
                raise ValueError(f"Método de amostragem desconhecido: {metodo}")
            
            # Aplicar limites físicos
            valor = self._aplicar_limites_fisicos(variavel, valor)
            variaveis_amostradas[variavel] = valor
        
        return variaveis_amostradas

    def _aplicar_limites_fisicos(self, variavel, valor):
        """Aplica limites físicos às variáveis amostradas."""
        if variavel == 'temperatura_ar':
            return np.clip(valor, config.TEMP_AR_MIN, config.TEMP_AR_MAX)
        elif variavel == 'radiacao_global':
            return np.clip(valor, config.RADIACAO_MIN, config.RADIACAO_MAX)
        elif variavel in ['vento_u', 'vento_v']:
            return np.clip(valor, -config.VENTO_VEL_MAX, config.VENTO_VEL_MAX)
        else:
            return valor

    def _reconstruir_vento(self, u, v):
        """
        Reconstrói velocidade e direção do vento a partir das componentes U e V.
        
        Args:
            u (float): Componente zonal do vento (m/s)
            v (float): Componente meridional do vento (m/s)
            
        Returns:
            dict: Velocidade e direção do vento
        """
        velocidade = np.sqrt(u**2 + v**2)
        direcao = np.degrees(np.arctan2(v, u))
        
        # Normalizar direção para 0-360°
        if direcao < 0:
            direcao += 360
        
        # Aplicar limite máximo de velocidade
        velocidade = min(velocidade, config.VENTO_VEL_MAX)
        
        return {
            'velocidade': velocidade,
            'direcao': direcao
        }

    def _calcular_angulo_vento(self, direcao_vento, azimute_linha):
        """
        Calcula o ângulo de ataque do vento em relação ao condutor.
        
        Args:
            direcao_vento (float): Direção do vento em graus (0-360)
            azimute_linha (float): Azimute da linha em graus
            
        Returns:
            float: Ângulo de ataque em graus (0-90)
        """
        angulo = abs(direcao_vento - azimute_linha)
        
        # Garantir que o ângulo esteja entre 0 e 180°
        if angulo > 180:
            angulo = 360 - angulo
        
        # Para o modelo CIGRE, usar o ângulo de 0 a 90°
        if angulo > 90:
            angulo = 180 - angulo
        
        return angulo

    def _validar_temperatura_resultado(self, temperatura, temperatura_ar):
        """
        Valida se a temperatura calculada é fisicamente razoável.
        
        Args:
            temperatura (float): Temperatura do condutor calculada
            temperatura_ar (float): Temperatura do ar
            
        Returns:
            bool: True se válida, False caso contrário
        """
        # Verificar valores numéricos
        if not np.isfinite(temperatura):
            return False
        
        # Temperatura do condutor deve ser >= temperatura do ar
        if temperatura < temperatura_ar - 5:  # Tolerância de 5°C
            return False
        
        # Limite superior razoável
        if temperatura > temperatura_ar + 200:
            return False
        
        return True

    def _calcular_estatisticas(self, temperaturas):
        """
        Calcula estatísticas descritivas das temperaturas simuladas.
        
        Args:
            temperaturas (np.array): Array de temperaturas
            
        Returns:
            dict: Estatísticas calculadas
        """
        if len(temperaturas) == 0:
            logger.warning("Nenhuma temperatura válida para cálculo de estatísticas")
            return {
                'media': np.nan,
                'mediana': np.nan,
                'desvio_padrao': np.nan,
                'minimo': np.nan,
                'maximo': np.nan,
                'percentil_5': np.nan,
                'percentil_10': np.nan,
                'percentil_90': np.nan,
                'percentil_95': np.nan,
                'percentil_99': np.nan
            }
        
        return {
            'media': np.mean(temperaturas),
            'mediana': np.median(temperaturas),
            'desvio_padrao': np.std(temperaturas),
            'minimo': np.min(temperaturas),
            'maximo': np.max(temperaturas),
            'percentil_5': np.percentile(temperaturas, 5),
            'percentil_10': np.percentile(temperaturas, 10),
            'percentil_90': np.percentile(temperaturas, config.PERCENTIL_CONFIANCA),
            'percentil_95': np.percentile(temperaturas, 95),
            'percentil_99': np.percentile(temperaturas, 99)
        }

    def analisar_sensibilidade(self, medias_ambientais, desvios_ambientais, azimute_linha,
                              corrente, num_iteracoes_sensibilidade=1000):
        """
        Realiza análise de sensibilidade das variáveis ambientais.
        
        Args:
            medias_ambientais (dict): Médias das variáveis ambientais
            desvios_ambientais (dict): Desvios padrão das variáveis ambientais
            azimute_linha (float): Azimute da linha
            corrente (float): Corrente elétrica
            num_iteracoes_sensibilidade (int): Número de iterações para análise
            
        Returns:
            dict: Resultados da análise de sensibilidade
        """
        logger.info("Iniciando análise de sensibilidade...")
        
        # Resultado base
        resultado_base = self.executar_simulacao(
            medias_ambientais, desvios_ambientais, azimute_linha, corrente, 
            num_iteracoes_sensibilidade, semente_aleatoria=42
        )
        
        temp_base = resultado_base['estatisticas']['percentil_90']
        
        sensibilidades = {}
        variacoes = [0.9, 1.1]  # ±10%
        
        for variavel in ['temperatura_ar', 'radiacao_global', 'vento_u', 'vento_v']:
            sensibilidades[variavel] = []
            
            for fator in variacoes:
                medias_modificadas = medias_ambientais.copy()
                medias_modificadas[variavel] *= fator
                
                try:
                    resultado_mod = self.executar_simulacao(
                        medias_modificadas, desvios_ambientais, azimute_linha, corrente,
                        num_iteracoes_sensibilidade, semente_aleatoria=42
                    )
                    
                    temp_mod = resultado_mod['estatisticas']['percentil_90']
                    sensibilidade = (temp_mod - temp_base) / (temp_base * (fator - 1))
                    sensibilidades[variavel].append(sensibilidade)
                    
                except Exception as e:
                    logger.warning(f"Erro na análise de sensibilidade para {variavel}: {e}")
                    sensibilidades[variavel].append(np.nan)
        
        # Calcular sensibilidade média
        for variavel in sensibilidades:
            valores_validos = [s for s in sensibilidades[variavel] if np.isfinite(s)]
            if valores_validos:
                sensibilidades[variavel] = np.mean(np.abs(valores_validos))
            else:
                sensibilidades[variavel] = np.nan
        
        logger.info("Análise de sensibilidade concluída")
        return {
            'temperatura_base': temp_base,
            'sensibilidades': sensibilidades,
            'variavel_mais_sensivel': max(sensibilidades, key=lambda k: sensibilidades[k] if np.isfinite(sensibilidades[k]) else 0)
        }