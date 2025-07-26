# Módulo para carregar e validar dados de entrada
import json
import pandas as pd
import os
import numpy as np
import re
import traceback

def carregar_parametros_cabo(file_path):
    """
    Carrega os parâmetros do condutor de um arquivo JSON.

    Args:
        file_path (str): O caminho para o arquivo JSON.

    Returns:
        dict: Um dicionário com os parâmetros do condutor.
    """
    with open(file_path, 'r') as f:
        parametros = json.load(f)
    
    # Validação
    required_keys = ["diametro", "resistencia_ac_25", "resistencia_ac_75", "emissividade", "absortividade"]
    for key in required_keys:
        if key not in parametros:
            raise ValueError(f"A chave '{key}' está faltando no arquivo de parâmetros do cabo.")
            
    return parametros

def carregar_dados_linha(file_path):
    """
    Carrega os dados do traçado da linha de um arquivo Excel.

    Args:
        file_path (str): O caminho para o arquivo Excel.

    Returns:
        pandas.DataFrame: Um DataFrame com os dados do traçado da linha.
    """
    dados_linha = pd.read_excel(file_path, engine='openpyxl')

    # Validação
    required_columns = ["Progressiva", "azimute", "latitude", "longitude"]
    for col in required_columns:
        if col not in dados_linha.columns:
            raise ValueError(f"A coluna '{col}' está faltando no arquivo de traçado da linha.")

    return dados_linha

def carregar_dados_estacoes(folder_path):
    """
    Carrega os dados de todas as estações de uma pasta.

    Args:
        folder_path (str): O caminho para a pasta com os arquivos CSV das estações.

    Returns:
        dict: Um dicionário de DataFrames, onde cada chave é o nome da estação.
    """
    dados_estacoes = {}
    all_columns = set()
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            print(f"[DEBUG] Tentando processar o arquivo: {filename}")
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
                    # Latitude na linha 3, Longitude na linha 4
                    lat = float(lines[2].split(':')[1].strip().replace(',', '.'))
                    lon = float(lines[3].split(':')[1].strip().replace(',', '.'))

                # Dados começam na linha 11 (skiprows=10)
                df = pd.read_csv(file_path, sep=';', skiprows=10, decimal=',', encoding='latin-1')
                
                print(f"[DEBUG] {filename}: Colunas após read_csv: {df.columns.tolist()}")

                # Limpar nomes das colunas: remover espaços extras e caracteres especiais
                cleaned_columns = []
                for col in df.columns:
                    cleaned_col = re.sub(r'[^a-zA-Z0-9_]', '', col.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('°', '').replace('²', '').replace('/', ''))
                    cleaned_columns.append(cleaned_col)
                df.columns = cleaned_columns

                print(f"[DEBUG] {filename}: Colunas após limpeza: {df.columns.tolist()}")
                
                if df.empty:
                    print(f"[DEBUG] {filename}: DataFrame está vazio após read_csv.")
                    continue

                df['latitude'] = lat
                df['longitude'] = lon

                # Renomear colunas para nomes padronizados
                df.rename(columns={
                    'DataMedicao': 'data_medicao',
                    'HoraMedicao': 'hora_medicao',
                    'TEMPERATURA_DO_AR__BULBO_SECO_HORARIAC': 'temperatura_ar',
                    'RADIACAO_GLOBALKjm': 'radiacao_global',
                    'VENTO_VELOCIDADE_HORARIAms': 'vento_velocidade',
                    'VENTO_DIRECAO_HORARIA_gr__gr': 'vento_direcao'
                }, inplace=True)
                print(f"[DEBUG] {filename}: Colunas após rename: {df.columns.tolist()}")

                # Converter data e hora
                df['data_hora'] = pd.to_datetime(df['data_medicao'] + df['hora_medicao'].astype(str).str.zfill(4), format='%Y-%m-%d%H%M')
                df.drop(columns=['data_medicao', 'hora_medicao'], inplace=True)
                df.set_index('data_hora', inplace=True)

                # Tratar dados faltantes
                df.replace(-9999, np.nan, inplace=True)
                df.dropna(subset=['temperatura_ar', 'radiacao_global', 'vento_velocidade', 'vento_direcao'], inplace=True)
                print(f"[DEBUG] {filename}: Shape após dropna: {df.shape}")

                # Decompor vento
                df['vento_u'] = df['vento_velocidade'] * np.cos(np.deg2rad(df['vento_direcao']))
                df['vento_v'] = df['vento_velocidade'] * np.sin(np.deg2rad(df['vento_direcao']))

                station_name = os.path.splitext(filename)[0]
                dados_estacoes[station_name] = df
            except Exception as e:
                print(f"Erro ao processar o arquivo {filename}: {e}")
                traceback.print_exc()

    print("Todas as colunas únicas encontradas:")
    for col in sorted(list(all_columns)):
        print(f"- {col}")

    return dados_estacoes
