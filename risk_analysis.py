# Módulo para analisar os resultados da simulação e calcular as métricas finais
import numpy as np
import pandas as pd
import logging
from scipy import stats
import config

logger = logging.getLogger(__name__)

class RiskAnalyzer:
    """
    Classe responsável pela análise de risco térmico conforme ABNT NBR 5422.
    """
    
    def __init__(self):
        """Inicializa o analisador de risco."""
        self.criterios_nbr_5422 = self._definir_criterios_nbr_5422()
        logger.info("Analisador de risco inicializado conforme NBR 5422")

    def _definir_criterios_nbr_5422(self):
        """
        Define os critérios de risco térmico conforme ABNT NBR 5422.
        
        Returns:
            dict: Critérios de classificação de risco
        """
        return {
            'risco_baixo': {'limite': 0.01, 'descricao': 'Risco baixo (< 1%)'},
            'risco_moderado': {'limite': 0.05, 'descricao': 'Risco moderado (1-5%)'},
            'risco_alto': {'limite': 0.10, 'descricao': 'Risco alto (5-10%)'},
            'risco_critico': {'limite': float('inf'), 'descricao': 'Risco crítico (> 10%)'}
        }

    def calcular_temperatura_confianca(self, temperaturas_distribuicao, percentil=None):
        """
        Calcula a temperatura do condutor correspondente a um percentil da distribuição.
        
        Args:
            temperaturas_distribuicao (np.array): Array com temperaturas da simulação Monte Carlo
            percentil (int): Percentil desejado (padrão: config.PERCENTIL_CONFIANCA)
            
        Returns:
            float: Temperatura no percentil especificado
        """
        if percentil is None:
            percentil = config.PERCENTIL_CONFIANCA
        
        if len(temperaturas_distribuicao) == 0:
            logger.warning("Array de temperaturas vazio")
            return np.nan
        
        # Remover valores inválidos
        temperaturas_validas = temperaturas_distribuicao[np.isfinite(temperaturas_distribuicao)]
        
        if len(temperaturas_validas) == 0:
            logger.warning("Nenhuma temperatura válida encontrada")
            return np.nan
        
        return np.percentile(temperaturas_validas, percentil)

    def calcular_risco_termico(self, temperaturas_distribuicao, temperatura_max_projeto):
        """
        Calcula a probabilidade de exceder uma temperatura máxima de projeto.
        
        Args:
            temperaturas_distribuicao (np.array): Array com temperaturas da simulação
            temperatura_max_projeto (float): Temperatura máxima de projeto (T_max)
            
        Returns:
            float: Probabilidade (0-1) de exceder T_max
        """
        if len(temperaturas_distribuicao) == 0:
            logger.warning("Array de temperaturas vazio")
            return np.nan
        
        # Remover valores inválidos
        temperaturas_validas = temperaturas_distribuicao[np.isfinite(temperaturas_distribuicao)]
        
        if len(temperaturas_validas) == 0:
            logger.warning("Nenhuma temperatura válida encontrada")
            return np.nan
        
        # Contar excedências
        excedencias = np.sum(temperaturas_validas > temperatura_max_projeto)
        probabilidade_excedencia = excedencias / len(temperaturas_validas)
        
        return probabilidade_excedencia

    def classificar_risco_nbr_5422(self, probabilidade_excedencia):
        """
        Classifica o risco térmico conforme NBR 5422.
        
        Args:
            probabilidade_excedencia (float): Probabilidade de exceder temperatura limite
            
        Returns:
            dict: Classificação do risco
        """
        if np.isnan(probabilidade_excedencia):
            return {
                'categoria': 'indefinido',
                'descricao': 'Risco indefinido (dados insuficientes)',
                'cor': 'gray',
                'acao_recomendada': 'Revisar dados de entrada'
            }
        
        if probabilidade_excedencia < self.criterios_nbr_5422['risco_baixo']['limite']:
            return {
                'categoria': 'baixo',
                'descricao': self.criterios_nbr_5422['risco_baixo']['descricao'],
                'cor': 'green',
                'acao_recomendada': 'Operação normal'
            }
        elif probabilidade_excedencia < self.criterios_nbr_5422['risco_moderado']['limite']:
            return {
                'categoria': 'moderado',
                'descricao': self.criterios_nbr_5422['risco_moderado']['descricao'],
                'cor': 'yellow',
                'acao_recomendada': 'Monitoramento reforçado'
            }
        elif probabilidade_excedencia < self.criterios_nbr_5422['risco_alto']['limite']:
            return {
                'categoria': 'alto',
                'descricao': self.criterios_nbr_5422['risco_alto']['descricao'],
                'cor': 'orange',
                'acao_recomendada': 'Revisão dos limites operacionais'
            }
        else:
            return {
                'categoria': 'critico',
                'descricao': self.criterios_nbr_5422['risco_critico']['descricao'],
                'cor': 'red',
                'acao_recomendada': 'Intervenção imediata necessária'
            }

    def calcular_intervalo_confianca(self, temperaturas_distribuicao, nivel_confianca=0.95):
        """
        Calcula intervalo de confiança para a temperatura do condutor.
        
        Args:
            temperaturas_distribuicao (np.array): Array de temperaturas
            nivel_confianca (float): Nível de confiança (0-1)
            
        Returns:
            dict: Intervalo de confiança
        """
        if len(temperaturas_distribuicao) == 0:
            return {'limite_inferior': np.nan, 'limite_superior': np.nan}
        
        temperaturas_validas = temperaturas_distribuicao[np.isfinite(temperaturas_distribuicao)]
        
        if len(temperaturas_validas) < 2:
            return {'limite_inferior': np.nan, 'limite_superior': np.nan}
        
        alpha = 1 - nivel_confianca
        p_inferior = (alpha / 2) * 100
        p_superior = (1 - alpha / 2) * 100
        
        return {
            'limite_inferior': np.percentile(temperaturas_validas, p_inferior),
            'limite_superior': np.percentile(temperaturas_validas, p_superior),
            'nivel_confianca': nivel_confianca
        }

    def testar_normalidade(self, temperaturas_distribuicao, alpha=0.05):
        """
        Testa se a distribuição de temperaturas segue uma distribuição normal.
        
        Args:
            temperaturas_distribuicao (np.array): Array de temperaturas
            alpha (float): Nível de significância
            
        Returns:
            dict: Resultados do teste de normalidade
        """
        if len(temperaturas_distribuicao) < 3:
            return {
                'eh_normal': False,
                'p_valor': np.nan,
                'estatistica': np.nan,
                'teste': 'Dados insuficientes'
            }
        
        temperaturas_validas = temperaturas_distribuicao[np.isfinite(temperaturas_distribuicao)]
        
        if len(temperaturas_validas) < 3:
            return {
                'eh_normal': False,
                'p_valor': np.nan,
                'estatistica': np.nan,
                'teste': 'Dados válidos insuficientes'
            }
        
        try:
            # Teste de Shapiro-Wilk para amostras pequenas (< 5000)
            if len(temperaturas_validas) < 5000:
                estatistica, p_valor = stats.shapiro(temperaturas_validas)
                teste_usado = 'Shapiro-Wilk'
            else:
                # Teste de Kolmogorov-Smirnov para amostras grandes
                estatistica, p_valor = stats.kstest(temperaturas_validas, 'norm',
                    args=(np.mean(temperaturas_validas), np.std(temperaturas_validas)))
                teste_usado = 'Kolmogorov-Smirnov'
            
            return {
                'eh_normal': p_valor > alpha,
                'p_valor': p_valor,
                'estatistica': estatistica,
                'teste': teste_usado,
                'interpretacao': 'Normal' if p_valor > alpha else 'Não-normal'
            }
            
        except Exception as e:
            logger.warning(f"Erro no teste de normalidade: {e}")
            return {
                'eh_normal': False,
                'p_valor': np.nan,
                'estatistica': np.nan,
                'teste': f'Erro: {e}'
            }

    def calcular_vida_util_estimada(self, temperaturas_distribuicao, temperatura_nominal,
                                   fator_arrhenius=15000):
        """
        Estima a redução da vida útil baseada na temperatura de operação.
        
        Args:
            temperaturas_distribuicao (np.array): Array de temperaturas
            temperatura_nominal (float): Temperatura nominal de projeto (°C)
            fator_arrhenius (float): Fator de Arrhenius para o material
            
        Returns:
            dict: Estimativa de vida útil
        """
        if len(temperaturas_distribuicao) == 0:
            return {'fator_reducao': np.nan, 'vida_util_relativa': np.nan}
        
        temperaturas_validas = temperaturas_distribuicao[np.isfinite(temperaturas_distribuicao)]
        
        if len(temperaturas_validas) == 0:
            return {'fator_reducao': np.nan, 'vida_util_relativa': np.nan}
        
        # Converter para Kelvin
        T_nominal_K = temperatura_nominal + 273.15
        T_operacao_K = np.mean(temperaturas_validas) + 273.15
        
        # Cálculo baseado na equação de Arrhenius
        fator_reducao = np.exp(fator_arrhenius * (1/T_nominal_K - 1/T_operacao_K))
        
        return {
            'fator_reducao': fator_reducao,
            'vida_util_relativa': 1.0 / fator_reducao,
            'temperatura_media_operacao': np.mean(temperaturas_validas),
            'temperatura_nominal': temperatura_nominal
        }

    def analisar_tendencias_temporais(self, resultados_temporais):
        """
        Analisa tendências temporais nos resultados de risco.
        
        Args:
            resultados_temporais (pd.DataFrame): DataFrame com resultados por hora
            
        Returns:
            dict: Análise de tendências
        """
        if resultados_temporais.empty:
            return {'erro': 'DataFrame vazio'}
        
        try:
            # Agrupar por hora do dia
            resultados_temporais['hora_dia'] = pd.to_datetime(
                resultados_temporais['hora']
            ).dt.hour
            
            tendencias_horarias = resultados_temporais.groupby('hora_dia').agg({
                'temperatura_condutor_p90': ['mean', 'std', 'max'],
                'risco_termico': ['mean', 'max'],
                'ampacidade_calculada': ['mean', 'min']
            }).round(3)
            
            # Encontrar horários críticos
            idx_temp_max = resultados_temporais['temperatura_condutor_p90'].idxmax()
            hora_temp_max = resultados_temporais.loc[idx_temp_max, 'hora_dia']
            
            idx_risco_max = resultados_temporais['risco_termico'].idxmax()
            hora_risco_max = resultados_temporais.loc[idx_risco_max, 'hora_dia']
            
            return {
                'tendencias_horarias': tendencias_horarias,
                'hora_temperatura_maxima': hora_temp_max,
                'hora_risco_maximo': hora_risco_max,
                'temperatura_maxima_absoluta': resultados_temporais['temperatura_condutor_p90'].max(),
                'risco_maximo_absoluto': resultados_temporais['risco_termico'].max()
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de tendências: {e}")
            return {'erro': str(e)}

    def gerar_relatorio_risco(self, temperaturas_distribuicao, temperatura_max_projeto,
                             nome_ponto="Ponto não especificado"):
        """
        Gera relatório completo de análise de risco.
        
        Args:
            temperaturas_distribuicao (np.array): Array de temperaturas
            temperatura_max_projeto (float): Temperatura máxima de projeto
            nome_ponto (str): Nome/identificador do ponto
            
        Returns:
            dict: Relatório completo de risco
        """
        if len(temperaturas_distribuicao) == 0:
            return {
                'erro': 'Dados insuficientes',
                'ponto': nome_ponto
            }
        
        # Cálculos básicos
        temp_confianca = self.calcular_temperatura_confianca(temperaturas_distribuicao)
        risco_termico = self.calcular_risco_termico(temperaturas_distribuicao, temperatura_max_projeto)
        classificacao_risco = self.classificar_risco_nbr_5422(risco_termico)
        
        # Estatísticas da distribuição
        temperaturas_validas = temperaturas_distribuicao[np.isfinite(temperaturas_distribuicao)]
        
        estatisticas = {
            'media': np.mean(temperaturas_validas),
            'mediana': np.median(temperaturas_validas),
            'desvio_padrao': np.std(temperaturas_validas),
            'minimo': np.min(temperaturas_validas),
            'maximo': np.max(temperaturas_validas),
            'percentil_5': np.percentile(temperaturas_validas, 5),
            'percentil_95': np.percentile(temperaturas_validas, 95),
            'coeficiente_variacao': np.std(temperaturas_validas) / np.mean(temperaturas_validas) if np.mean(temperaturas_validas) > 0 else np.nan
        }
        
        # Testes estatísticos
        teste_normalidade = self.testar_normalidade(temperaturas_distribuicao)
        intervalo_confianca = self.calcular_intervalo_confianca(temperaturas_distribuicao)
        
        # Estimativa de vida útil
        vida_util = self.calcular_vida_util_estimada(temperaturas_distribuicao, 75)  # 75°C como nominal
        
        # Compilar relatório
        relatorio = {
            'identificacao': {
                'ponto': nome_ponto,
                'data_analise': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'temperatura_limite': temperatura_max_projeto
            },
            'resultados_principais': {
                'temperatura_confianca_p90': temp_confianca,
                'probabilidade_excedencia': risco_termico,
                'classificacao_risco': classificacao_risco
            },
            'estatisticas_distribuicao': estatisticas,
            'testes_estatisticos': {
                'normalidade': teste_normalidade,
                'intervalo_confianca_95': intervalo_confianca
            },
            'estimativa_vida_util': vida_util,
            'interpretacao': {
                'conclusao_principal': self._interpretar_resultado_principal(classificacao_risco, temp_confianca, temperatura_max_projeto),
                'recomendacoes': self._gerar_recomendacoes(classificacao_risco, vida_util)
            }
        }
        
        return relatorio

    def _interpretar_resultado_principal(self, classificacao_risco, temp_confianca, temp_limite):
        """Gera interpretação principal dos resultados."""
        categoria = classificacao_risco['categoria']
        
        if categoria == 'baixo':
            return f"Operação segura. Temperatura P90 ({temp_confianca:.1f}°C) bem abaixo do limite ({temp_limite}°C)."
        elif categoria == 'moderado':
            return f"Operação aceitável com monitoramento. Temperatura P90 ({temp_confianca:.1f}°C) próxima ao limite."
        elif categoria == 'alto':
            return f"Atenção requerida. Alta probabilidade de exceder limite operacional."
        else:  # crítico
            return f"Situação crítica. Temperatura P90 ({temp_confianca:.1f}°C) excede significativamente o limite."

    def _gerar_recomendacoes(self, classificacao_risco, vida_util):
        """Gera recomendações baseadas na análise."""
        recomendacoes = []
        
        categoria = classificacao_risco['categoria']
        
        if categoria == 'baixo':
            recomendacoes.append("Manter operação normal")
            recomendacoes.append("Revisão periódica conforme programação")
        elif categoria == 'moderado':
            recomendacoes.append("Intensificar monitoramento da temperatura")
            recomendacoes.append("Avaliar condições ambientais durante picos de carga")
        elif categoria == 'alto':
            recomendacoes.append("Revisar limites operacionais de corrente")
            recomendacoes.append("Considerar melhorias no sistema de resfriamento")
            recomendacoes.append("Implementar monitoramento contínuo")
        else:  # crítico
            recomendacoes.append("Reduzir imediatamente a carga operacional")
            recomendacoes.append("Investigar causas da sobrecarga térmica")
            recomendacoes.append("Considerar substituição ou upgrade do condutor")
        
        # Recomendações baseadas na vida útil
        if vida_util.get('vida_util_relativa', 1) < 0.8:
            recomendacoes.append("Vida útil pode estar comprometida - avaliar substituição")
        
        return recomendacoes

    def exportar_relatorio_csv(self, relatorio, caminho_arquivo):
        """
        Exporta relatório para arquivo CSV.
        
        Args:
            relatorio (dict): Relatório gerado
            caminho_arquivo (str): Caminho para salvar o arquivo
        """
        try:
            # Flatten do dicionário para CSV
            dados_csv = []
            
            for secao, conteudo in relatorio.items():
                if isinstance(conteudo, dict):
                    for chave, valor in conteudo.items():
                        if isinstance(valor, dict):
                            for sub_chave, sub_valor in valor.items():
                                dados_csv.append({
                                    'secao': secao,
                                    'categoria': chave,
                                    'parametro': sub_chave,
                                    'valor': sub_valor
                                })
                        else:
                            dados_csv.append({
                                'secao': secao,
                                'categoria': chave,
                                'parametro': '',
                                'valor': valor
                            })
                else:
                    dados_csv.append({
                        'secao': secao,
                        'categoria': '',
                        'parametro': '',
                        'valor': conteudo
                    })
            
            df_relatorio = pd.DataFrame(dados_csv)
            df_relatorio.to_csv(caminho_arquivo, index=False, encoding='utf-8')
            logger.info(f"Relatório exportado para: {caminho_arquivo}")
            
        except Exception as e:
            logger.error(f"Erro ao exportar relatório: {e}")
            raise