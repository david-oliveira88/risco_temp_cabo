# Módulo para analisar os resultados da simulação e calcular as métricas finais.
import numpy as np

def calcular_temperatura_confianca(temperaturas_distribuicao, percentil=90):
    """
    Calcula a temperatura do condutor correspondente a um dado percentil da distribuição.

    Args:
        temperaturas_distribuicao (np.array): Array com as temperaturas do condutor da simulação Monte Carlo.
        percentil (int): O percentil desejado (ex: 90 para o percentil 90).

    Returns:
        float: A temperatura no percentil especificado.
    """
    return np.percentile(temperaturas_distribuicao, percentil)

def calcular_risco_termico(temperaturas_distribuicao, temperatura_max_projeto):
    """
    Calcula a probabilidade de exceder uma temperatura máxima de projeto.

    Args:
        temperaturas_distribuicao (np.array): Array com as temperaturas do condutor da simulação Monte Carlo.
        temperatura_max_projeto (float): A temperatura máxima de projeto (T_max).

    Returns:
        float: A probabilidade (entre 0 e 1) de a temperatura exceder T_max.
    """
    excedencias = temperaturas_distribuicao[temperaturas_distribuicao > temperatura_max_projeto]
    probabilidade_excedencia = len(excedencias) / len(temperaturas_distribuicao)
    return probabilidade_excedencia