# Módulo para conversão de coordenadas e Krigagem
from pyproj import CRS, Transformer
import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging

def converter_coordenadas(latitude, longitude, crs_origem='EPSG:4674', crs_destino='EPSG:5880'):
    """
    Converte coordenadas de um sistema de referência para outro.

    Args:
        latitude (float): Latitude no sistema de origem.
        longitude (float): Longitude no sistema de origem.
        crs_origem (str): Código EPSG do sistema de coordenadas de origem.
        crs_destino (str): Código EPSG do sistema de coordenadas de destino.

    Returns:
        tuple: Uma tupla (x, y) com as coordenadas convertidas em metros.
    """
    transformer = Transformer.from_crs(crs_origem, crs_destino, always_xy=True)
    x, y = transformer.transform(longitude, latitude)
    return x, y

def discretizar_linha(dados_linha, distancia_entre_pontos=10):
    """
    Discretiza o traçado da linha em pontos equidistantes.

    Args:
        dados_linha (pd.DataFrame): DataFrame com as colunas 'Progressiva', 'azimute', 'latitude', 'longitude'.
        distancia_entre_pontos (int): Distância em metros entre os pontos discretizados.

    Returns:
        pd.DataFrame: DataFrame com os pontos discretizados, suas coordenadas projetadas e azimutes.
    """
    pontos_discretizados = []

    for i in range(len(dados_linha) - 1):
        ponto_inicial = dados_linha.iloc[i]
        ponto_final = dados_linha.iloc[i+1]

        lat_inicial, lon_inicial = ponto_inicial['latitude'], ponto_inicial['longitude']
        lat_final, lon_final = ponto_final['latitude'], ponto_final['longitude']

        x_inicial, y_inicial = converter_coordenadas(lat_inicial, lon_inicial)
        x_final, y_final = converter_coordenadas(lat_final, lon_final)

        distancia_segmento = np.sqrt((x_final - x_inicial)**2 + (y_final - y_inicial)**2)
        num_pontos_segmento = int(distancia_segmento / distancia_entre_pontos)

        for j in range(num_pontos_segmento):
            fracao = j / num_pontos_segmento
            lat_interp = lat_inicial + fracao * (lat_final - lat_inicial)
            lon_interp = lon_inicial + fracao * (lon_final - lon_inicial)
            azimute_interp = ponto_inicial['azimute'] # Assumindo azimute constante no segmento

            x_interp, y_interp = converter_coordenadas(lat_interp, lon_interp)

            pontos_discretizados.append({
                'latitude': lat_interp,
                'longitude': lon_interp,
                'x': x_interp,
                'y': y_interp,
                'azimute': azimute_interp
            })

    return pd.DataFrame(pontos_discretizados)

def krigagem_horaria(dados_estacoes, pontos_linha, variaveis_ambientais=['temperatura_ar', 'radiacao_global', 'vento_u', 'vento_v']):
    """
    Realiza a Krigagem Ordinária para cada hora e variável ambiental.

    Args:
        dados_estacoes (dict): Dicionário de DataFrames com os dados das estações.
        pontos_linha (pd.DataFrame): DataFrame com os pontos discretizados da linha.
        variaveis_ambientais (list): Lista de variáveis ambientais para krigagem.

    Returns:
        dict: Um dicionário com os resultados da krigagem (média e variância) para cada ponto da linha e hora.
    """
    resultados_krigagem = {}

    # Consolidar dados de todas as estações em um único DataFrame para facilitar a iteração por hora
    df_consolidado = pd.concat(dados_estacoes.values())
    df_consolidado.set_index('data_medicao', inplace=True)
    df_consolidado.sort_index(inplace=True)

    horas_unicas = df_consolidado.index.unique()

    for hora in horas_unicas:
        dados_hora = df_consolidado.loc[[hora]]
        
        # Coordenadas das estações para a hora atual
        estacao_coords = np.array([[converter_coordenadas(row['latitude'], row['longitude'])[0], 
                                    converter_coordenadas(row['latitude'], row['longitude'])[1]] 
                                   for index, row in dados_hora.iterrows()])

        # Coordenadas dos pontos da linha
        linha_coords = pontos_linha[['x', 'y']].values

        resultados_krigagem[hora] = {}
        for var in variaveis_ambientais:
            valores_estacoes = dados_hora[var].values
            
            # Remover NaNs dos valores das estações e suas coordenadas correspondentes
            valid_indices = ~np.isnan(valores_estacoes)
            valores_estacoes_validos = valores_estacoes[valid_indices]
            estacao_coords_validas = estacao_coords[valid_indices]

            if len(valores_estacoes_validos) > 1: # Krigagem requer pelo menos 2 pontos
                OK = OrdinaryKriging(
                    estacao_coords_validas[:, 0], 
                    estacao_coords_validas[:, 1], 
                    valores_estacoes_validos,
                    variogram_model='linear',
                    verbose=False,
                    enable_plotting=False
                )
                z, ss = OK.execute('points', linha_coords[:, 0], linha_coords[:, 1])
                resultados_krigagem[hora][var] = {'media': z.data, 'variancia': ss.data}
            else:
                # Se não há dados suficientes para krigagem, preencher com NaN ou estratégia alternativa
                resultados_krigagem[hora][var] = {'media': np.full(len(linha_coords), np.nan), 'variancia': np.full(len(linha_coords), np.nan)}

    return resultados_krigagem
