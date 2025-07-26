# Orquestrador principal da execução
from data_loader import carregar_parametros_cabo, carregar_dados_linha, carregar_dados_estacoes
from geoprocessing import converter_coordenadas, discretizar_linha, krigagem_horaria
from thermal_model import CigreModeloTermico
from simulation import MonteCarloSimulator
from risk_analysis import calcular_temperatura_confianca, calcular_risco_termico
import pandas as pd
import numpy as np

def main():
    # 1. Carregar configurações e dados
    parametros_cabo = carregar_parametros_cabo('C:/Users/gpate/OneDrive/Documentos/_Codigo/Capacidade_Operativa/Artigo/analisador_risco_termico/entrada/parametros_cabo.json')
    
    # Create a valid trassado_linha.xlsx for testing
    dados_linha_df = pd.DataFrame({
        'Progressiva': [0, 1, 2],
        'azimute': [0, 0, 0],
        'latitude': [-23.5505, -23.5515, -23.5525],
        'longitude': [-46.6333, -46.6343, -46.6353]
    })
    dados_linha_df.to_excel('C:/Users/gpate/OneDrive/Documentos/_Codigo/Capacidade_Operativa/Artigo/analisador_risco_termico/entrada/trassado_linha.xlsx', index=False)

    dados_linha = carregar_dados_linha('C:/Users/gpate/OneDrive/Documentos/_Codigo/Capacidade_Operativa/Artigo/analisador_risco_termico/entrada/trassado_linha.xlsx')
    dados_estacoes = carregar_dados_estacoes('C:/Users/gpate/OneDrive/Documentos/_Codigo/Capacidade_Operativa/Artigo/dados') # Use the actual data folder

    print(f"Dados estações carregados: {dados_estacoes.keys()}")
    if dados_estacoes:
        first_station_key = list(dados_estacoes.keys())[0]
        print(f"Colunas da primeira estação ({first_station_key}): {dados_estacoes[first_station_key].columns.tolist()}")

    # 2. Processar dados geoespaciais e executar Krigagem
    pontos_linha = discretizar_linha(dados_linha)
    resultados_krigagem = krigagem_horaria(dados_estacoes, pontos_linha)

    # 3. Inicializar modelos
    modelo_termico = CigreModeloTermico(parametros_cabo)
    simulador_mc = MonteCarloSimulator(modelo_termico)

    resultados_finais = []

    # 4. Iterar sobre cada ponto da linha e cada hora do período de análise
    for idx_ponto, ponto in pontos_linha.iterrows():
        for hora, dados_hora_krigagem in resultados_krigagem.items():
            # Obter a corrente elétrica para a hora atual (assumindo constante por enquanto)
            current = 500  # Exemplo: 500 Amperes

            # Obter médias e desvios-padrão das variáveis ambientais para o ponto/hora atual
            environmental_means = {
                'temperatura_ar': dados_hora_krigagem['temperatura_ar']['media'][idx_ponto],
                'radiacao_global': dados_hora_krigagem['radiacao_global']['media'][idx_ponto],
                'vento_u': dados_hora_krigagem['vento_u']['media'][idx_ponto],
                'vento_v': dados_hora_krigagem['vento_v']['media'][idx_ponto]
            }
            environmental_stds = {
                'temperatura_ar': np.sqrt(dados_hora_krigagem['temperatura_ar']['variancia'][idx_ponto]),
                'radiacao_global': np.sqrt(dados_hora_krigagem['radiacao_global']['variancia'][idx_ponto]),
                'vento_u': np.sqrt(dados_hora_krigagem['vento_u']['variancia'][idx_ponto]),
                'vento_v': np.sqrt(dados_hora_krigagem['vento_v']['variancia'][idx_ponto])
            }

            # 5. Executar a simulação de Monte Carlo
            num_iterations = 1000
            temperaturas_mc = simulador_mc.run_simulation(
                environmental_means,
                environmental_stds,
                ponto['azimute'],  # Azimute da linha para o cálculo do ângulo do vento
                current,
                num_iterations
            )

            # 6. Executar a análise de risco
            temp_confianca_90 = calcular_temperatura_confianca(temperaturas_mc, 90)
            temperatura_max_projeto = 75  # Exemplo de temperatura máxima de projeto
            risco_termico = calcular_risco_termico(temperaturas_mc, temperatura_max_projeto)

            # Armazenar os resultados
            resultados_finais.append({
                'hora': hora,
                'ponto_idx': idx_ponto,
                'latitude': ponto['latitude'],
                'longitude': ponto['longitude'],
                'temperatura_condutor_p90': temp_confianca_90,
                'risco_termico': risco_termico
            })

    # 7. Salvar os resultados finais
    df_resultados = pd.DataFrame(resultados_finais)
    df_resultados.to_csv('C:/Users/gpate/OneDrive/Documentos/_Codigo/Capacidade_Operativa/Artigo/analisador_risco_termico/saida/resultado_horario.csv', index=False)
    print("Simulação concluída e resultados salvos em resultado_horario.csv")

if __name__ == "__main__":
    main()
