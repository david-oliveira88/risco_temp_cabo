# Módulo para carregar e validar dados de entrada
import json
import pandas as pd
import os
import numpy as np
import re
import traceback
import logging
from datetime import datetime
import config

# Configurar logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class DataLoader:
    """Classe responsável por carregar e validar todos os dados de entrada."""
    
    def __init__(self):
        self.parametros_cabo = None
        self.dados_linha = None
        self.dados_estacoes = {}
        self.dados_sincronizados = None
        
    def carregar_todos_dados(self):
        """Carrega todos os dados necessários para a análise."""
        logger.info("Iniciando carregamento de dados...")
        
        try:
            self.parametros_cabo = self.carregar_parametros_cabo()
            self.dados_linha = self.carregar_dados_linha()
            self.dados_estacoes = self.carregar_dados_estacoes()
            self.dados_sincronizados = self.sincronizar_dados_estacoes()
            
            logger.info("Todos os dados carregados com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            raise

    def carregar_parametros_cabo(self):
        """
        Carrega os parâmetros do condutor de um arquivo JSON.
        
        Returns:
            dict: Dicionário com os parâmetros do condutor.
        """
        logger.info(f"Carregando parâmetros do cabo: {config.ARQUIVO_PARAMETROS_CABO}")
        
        try:
            with open(config.ARQUIVO_PARAMETROS_CABO, 'r', encoding='utf-8') as f:
                parametros = json.load(f)
            
            # Validação dos parâmetros obrigatórios
            for key in config.PARAMETROS_CABO_OBRIGATORIOS:
                if key not in parametros:
                    raise ValueError(f"Parâmetro obrigatório '{key}' não encontrado no arquivo")
                
                # Validação de tipos e valores
                valor = parametros[key]
                if not isinstance(valor, (int, float)) or valor <= 0:
                    raise ValueError(f"Parâmetro '{key}' deve ser um número positivo")
            
            logger.info("Parâmetros do cabo carregados e validados")
            return parametros
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo de parâmetros não encontrado: {config.ARQUIVO_PARAMETROS_CABO}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Erro ao decodificar JSON: {e}")

    def carregar_dados_linha(self):
        """
        Carrega os dados do traçado da linha de um arquivo Excel.
        
        Returns:
            pandas.DataFrame: DataFrame com os dados do traçado da linha.
        """
        logger.info(f"Carregando dados da linha: {config.ARQUIVO_TRASSADO_LINHA}")
        
        try:
            dados_linha = pd.read_excel(config.ARQUIVO_TRASSADO_LINHA, engine='openpyxl')
            
            # Validação das colunas obrigatórias
            for col in config.COLUNAS_TRASSADO_OBRIGATORIAS:
                if col not in dados_linha.columns:
                    raise ValueError(f"Coluna obrigatória '{col}' não encontrada")
            
            # Validação de dados
            if dados_linha.empty:
                raise ValueError("Arquivo de traçado está vazio")
            
            # Validação de coordenadas
            lat_validas = (-90 <= dados_linha['latitude']) & (dados_linha['latitude'] <= 90)
            lon_validas = (-180 <= dados_linha['longitude']) & (dados_linha['longitude'] <= 180)
            
            if not lat_validas.all():
                raise ValueError("Latitudes inválidas encontradas")
            if not lon_validas.all():
                raise ValueError("Longitudes inválidas encontradas")
            
            logger.info(f"Dados da linha carregados: {len(dados_linha)} pontos")
            return dados_linha
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo de traçado não encontrado: {config.ARQUIVO_TRASSADO_LINHA}")

    def carregar_dados_estacoes(self):
        """
        Carrega os dados de todas as estações meteorológicas.
        
        Returns:
            dict: Dicionário com DataFrames das estações.
        """
        logger.info(f"Carregando dados das estações: {config.DADOS_DIR}")
        
        dados_estacoes = {}
        arquivos_processados = 0
        arquivos_com_erro = 0
        
        for filename in os.listdir(config.DADOS_DIR):
            if not filename.endswith(".csv"):
                continue
                
            file_path = os.path.join(config.DADOS_DIR, filename)
            station_name = os.path.splitext(filename)[0]
            
            try:
                df_estacao = self._processar_arquivo_estacao(file_path, station_name)
                if df_estacao is not None and not df_estacao.empty:
                    dados_estacoes[station_name] = df_estacao
                    arquivos_processados += 1
                    logger.info(f"Estação {station_name}: {len(df_estacao)} registros válidos")
                else:
                    logger.warning(f"Estação {station_name}: nenhum registro válido")
                    arquivos_com_erro += 1
                    
            except Exception as e:
                logger.error(f"Erro ao processar {filename}: {e}")
                arquivos_com_erro += 1
        
        if not dados_estacoes:
            raise ValueError("Nenhuma estação foi carregada com sucesso")
        
        logger.info(f"Resumo: {arquivos_processados} estações carregadas, {arquivos_com_erro} com erro")
        return dados_estacoes

    def _processar_arquivo_estacao(self, file_path, station_name):
        """
        Processa um arquivo individual de estação meteorológica.
        
        Args:
            file_path (str): Caminho para o arquivo CSV
            station_name (str): Nome da estação
            
        Returns:
            pandas.DataFrame: DataFrame processado da estação
        """
        try:
            # Ler coordenadas do cabeçalho
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
                
            lat_line = next((line for line in lines if 'Latitude' in line), None)
            lon_line = next((line for line in lines if 'Longitude' in line), None)
            
            if not lat_line or not lon_line:
                raise ValueError("Coordenadas não encontradas no cabeçalho")
            
            lat = float(lat_line.split(':')[1].strip().replace(',', '.'))
            lon = float(lon_line.split(':')[1].strip().replace(',', '.'))
            
            # Ler dados (assumindo que começam na linha 11)
            df = pd.read_csv(file_path, sep=';', skiprows=10, decimal=',', encoding='latin-1')
            
            if df.empty:
                return None
            
            # Limpar nomes das colunas
            df.columns = [self._limpar_nome_coluna(col) for col in df.columns]
            
            # Mapear colunas para nomes padronizados
            mapeamento_colunas = self._obter_mapeamento_colunas()
            df.rename(columns=mapeamento_colunas, inplace=True)
            
            # Verificar se todas as colunas necessárias estão presentes
            colunas_necessarias = ['data_medicao', 'hora_medicao', 'temperatura_ar', 
                                 'radiacao_global', 'vento_velocidade', 'vento_direcao']
            
            for col in colunas_necessarias:
                if col not in df.columns:
                    raise ValueError(f"Coluna necessária '{col}' não encontrada")
            
            # Adicionar coordenadas
            df['latitude'] = lat
            df['longitude'] = lon
            
            # Converter data e hora
            df['data_hora'] = pd.to_datetime(
                df['data_medicao'].astype(str) + df['hora_medicao'].astype(str).str.zfill(4), 
                format='%Y-%m-%d%H%M', 
                errors='coerce'
            )
            
            # Remover registros com data/hora inválida
            df = df.dropna(subset=['data_hora'])
            df.set_index('data_hora', inplace=True)
            df.drop(columns=['data_medicao', 'hora_medicao'], inplace=True)
            
            # Tratar valores inválidos
            df = self._tratar_valores_invalidos(df)
            
            # Decompor vento em componentes U e V
            df = self._decompor_vento(df)
            
            # Validar limites físicos
            df = self._validar_limites_fisicos(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao processar {station_name}: {e}")
            return None

    def _limpar_nome_coluna(self, col_name):
        """Limpa e padroniza nomes de colunas."""
        # Remover caracteres especiais e espaços extras
        cleaned = re.sub(r'[^\w\s]', '', str(col_name).strip())
        cleaned = re.sub(r'\s+', '_', cleaned)
        return cleaned.lower()

    def _obter_mapeamento_colunas(self):
        """Retorna mapeamento de nomes de colunas para padrão."""
        return {
            'datamedicao': 'data_medicao',
            'horamedicao': 'hora_medicao',
            'temperatura_do_ar__bulbo_seco_horariac': 'temperatura_ar',
            'radiacao_globalkjm': 'radiacao_global',
            'vento_velocidade_horariams': 'vento_velocidade',
            'vento_direcao_horaria_gr__gr': 'vento_direcao'
        }

    def _tratar_valores_invalidos(self, df):
        """Trata valores inválidos nos dados meteorológicos."""
        # Substituir códigos de dados faltantes
        codigos_invalidos = [-9999, -999, 999999, np.inf, -np.inf]
        for codigo in codigos_invalidos:
            df.replace(codigo, np.nan, inplace=True)
        
        # Remover registros com valores faltantes nas variáveis críticas
        colunas_criticas = ['temperatura_ar', 'radiacao_global', 'vento_velocidade', 'vento_direcao']
        df_inicial = len(df)
        df = df.dropna(subset=colunas_criticas)
        
        registros_removidos = df_inicial - len(df)
        if registros_removidos > 0:
            logger.info(f"Removidos {registros_removidos} registros com dados faltantes")
        
        return df

    def _decompor_vento(self, df):
        """Decompõe velocidade e direção do vento em componentes U e V."""
        # Converter direção de graus para radianos
        direcao_rad = np.deg2rad(df['vento_direcao'])
        
        # Calcular componentes U (zonal) e V (meridional)
        df['vento_u'] = df['vento_velocidade'] * np.cos(direcao_rad)
        df['vento_v'] = df['vento_velocidade'] * np.sin(direcao_rad)
        
        return df

    def _validar_limites_fisicos(self, df):
        """Valida se os valores estão dentro de limites físicos aceitáveis."""
        filtros = {
            'temperatura_ar': (df['temperatura_ar'] >= config.TEMP_AR_MIN) & 
                             (df['temperatura_ar'] <= config.TEMP_AR_MAX),
            'radiacao_global': (df['radiacao_global'] >= config.RADIACAO_MIN) & 
                              (df['radiacao_global'] <= config.RADIACAO_MAX),
            'vento_velocidade': (df['vento_velocidade'] >= config.VENTO_VEL_MIN) & 
                               (df['vento_velocidade'] <= config.VENTO_VEL_MAX),
            'vento_direcao': (df['vento_direcao'] >= config.VENTO_DIR_MIN) & 
                            (df['vento_direcao'] <= config.VENTO_DIR_MAX)
        }
        
        # Aplicar todos os filtros
        filtro_combinado = pd.Series(True, index=df.index)
        for variavel, filtro in filtros.items():
            filtro_combinado &= filtro
            registros_invalidos = (~filtro).sum()
            if registros_invalidos > 0:
                logger.warning(f"{registros_invalidos} registros com {variavel} fora dos limites")
        
        df_filtrado = df[filtro_combinado]
        registros_removidos = len(df) - len(df_filtrado)
        
        if registros_removidos > 0:
            logger.info(f"Removidos {registros_removidos} registros fora dos limites físicos")
        
        return df_filtrado

    def sincronizar_dados_estacoes(self):
        """
        Sincroniza dados de todas as estações por timestamp, removendo horas 
        onde qualquer estação não tenha dados válidos.
        
        Returns:
            pandas.DataFrame: DataFrame consolidado com dados sincronizados
        """
        logger.info("Sincronizando dados entre estações...")
        
        if not self.dados_estacoes:
            raise ValueError("Nenhum dado de estação disponível para sincronização")
        
        # Encontrar timestamps comuns a todas as estações
        timestamps_comuns = None
        
        for nome_estacao, df_estacao in self.dados_estacoes.items():
            if timestamps_comuns is None:
                timestamps_comuns = set(df_estacao.index)
            else:
                timestamps_comuns &= set(df_estacao.index)
        
        if not timestamps_comuns:
            raise ValueError("Nenhum timestamp comum encontrado entre as estações")
        
        timestamps_comuns = sorted(list(timestamps_comuns))
        logger.info(f"Encontrados {len(timestamps_comuns)} timestamps comuns")
        
        # Consolidar dados das estações nos timestamps comuns
        dados_consolidados = []
        
        for timestamp in timestamps_comuns:
            for nome_estacao, df_estacao in self.dados_estacoes.items():
                if timestamp in df_estacao.index:
                    registro = df_estacao.loc[timestamp].copy()
                    registro['estacao'] = nome_estacao
                    registro['data_hora'] = timestamp
                    dados_consolidados.append(registro)
        
        df_consolidado = pd.DataFrame(dados_consolidados)
        df_consolidado.set_index('data_hora', inplace=True)
        
        logger.info(f"Dados sincronizados: {len(df_consolidado)} registros totais")
        
        # Estatísticas por estação
        for estacao in df_consolidado['estacao'].unique():
            count = (df_consolidado['estacao'] == estacao).sum()
            logger.info(f"Estação {estacao}: {count} registros sincronizados")
        
        return df_consolidado

    def obter_resumo_dados(self):
        """Retorna um resumo dos dados carregados."""
        resumo = {
            'parametros_cabo': self.parametros_cabo is not None,
            'dados_linha': len(self.dados_linha) if self.dados_linha is not None else 0,
            'num_estacoes': len(self.dados_estacoes),
            'estacoes': list(self.dados_estacoes.keys()),
            'dados_sincronizados': len(self.dados_sincronizados) if self.dados_sincronizados is not None else 0
        }
        
        if self.dados_sincronizados is not None:
            resumo['periodo_dados'] = {
                'inicio': self.dados_sincronizados.index.min(),
                'fim': self.dados_sincronizados.index.max()
            }
        
        return resumo