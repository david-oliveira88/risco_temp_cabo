# Módulo para conversão de coordenadas e Krigagem
from pyproj import CRS, Transformer
import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
import logging
import config
from sklearn.metrics import mean_squared_error
import warnings

# Configurar logging
logger = logging.getLogger(__name__)

class GeoProcessor:
    """Classe responsável por todo o processamento geoespacial."""
    
    def __init__(self):
        self.transformer = None
        self._inicializar_transformer()
    
    def _inicializar_transformer(self):
        """Inicializa o transformador de coordenadas."""
        try:
            self.transformer = Transformer.from_crs(
                config.CRS_ORIGEM, 
                config.CRS_DESTINO, 
                always_xy=True
            )
            logger.info(f"Transformer inicializado: {config.CRS_ORIGEM} -> {config.CRS_DESTINO}")
        except Exception as e:
            logger.error(f"Erro ao inicializar transformer: {e}")
            raise

    def converter_coordenadas(self, latitude, longitude):
        """
        Converte coordenadas de um sistema de referência para outro.
        
        Args:
            latitude (float): Latitude no sistema de origem
            longitude (float): Longitude no sistema de origem
            
        Returns:
            tuple: (x, y) coordenadas convertidas em metros
        """
        try:
            x, y = self.transformer.transform(longitude, latitude)
            return x, y
        except Exception as e:
            logger.error(f"Erro na conversão de coordenadas: {e}")
            raise

    def converter_coordenadas_lote(self, latitudes, longitudes):
        """
        Converte múltiplas coordenadas de uma vez.
        
        Args:
            latitudes (array-like): Array de latitudes
            longitudes (array-like): Array de longitudes
            
        Returns:
            tuple: (x_array, y_array) coordenadas convertidas
        """
        try:
            x_coords, y_coords = self.transformer.transform(longitudes, latitudes)
            return x_coords, y_coords
        except Exception as e:
            logger.error(f"Erro na conversão em lote: {e}")
            raise

    def discretizar_linha(self, dados_linha, distancia_entre_pontos=None):
        """
        Discretiza o traçado da linha em pontos equidistantes.
        
        Args:
            dados_linha (pd.DataFrame): DataFrame com colunas 'Progressiva', 'azimute', 'latitude', 'longitude'
            distancia_entre_pontos (int): Distância em metros entre pontos (padrão: config.DISTANCIA_DISCRETIZACAO)
            
        Returns:
            pd.DataFrame: DataFrame com pontos discretizados
        """
        if distancia_entre_pontos is None:
            distancia_entre_pontos = config.DISTANCIA_DISCRETIZACAO
            
        logger.info(f"Discretizando linha com pontos a cada {distancia_entre_pontos}m")
        
        pontos_discretizados = []
        
        for i in range(len(dados_linha) - 1):
            ponto_inicial = dados_linha.iloc[i]
            ponto_final = dados_linha.iloc[i + 1]
            
            # Converter coordenadas
            x_inicial, y_inicial = self.converter_coordenadas(
                ponto_inicial['latitude'], ponto_inicial['longitude']
            )
            x_final, y_final = self.converter_coordenadas(
                ponto_final['latitude'], ponto_final['longitude']
            )
            
            # Calcular distância do segmento
            distancia_segmento = np.sqrt(
                (x_final - x_inicial)**2 + (y_final - y_inicial)**2
            )
            
            # Número de pontos no segmento
            num_pontos_segmento = max(1, int(distancia_segmento / distancia_entre_pontos))
            
            # Interpolar pontos ao longo do segmento
            for j in range(num_pontos_segmento):
                fracao = j / num_pontos_segmento if num_pontos_segmento > 1 else 0
                
                # Interpolação linear
                lat_interp = ponto_inicial['latitude'] + fracao * (
                    ponto_final['latitude'] - ponto_inicial['latitude']
                )
                lon_interp = ponto_inicial['longitude'] + fracao * (
                    ponto_final['longitude'] - ponto_inicial['longitude']
                )
                
                # Azimute interpolado (assumindo constante no segmento)
                azimute_interp = ponto_inicial['azimute']
                
                # Converter coordenadas interpoladas
                x_interp, y_interp = self.converter_coordenadas(lat_interp, lon_interp)
                
                pontos_discretizados.append({
                    'ponto_id': len(pontos_discretizados),
                    'segmento': i,
                    'fracao_segmento': fracao,
                    'latitude': lat_interp,
                    'longitude': lon_interp,
                    'x': x_interp,
                    'y': y_interp,
                    'azimute': azimute_interp,
                    'progressiva_aprox': ponto_inicial['Progressiva'] + fracao * (
                        ponto_final['Progressiva'] - ponto_inicial['Progressiva']
                    )
                })
        
        # Adicionar último ponto
        ultimo_ponto = dados_linha.iloc[-1]
        x_ultimo, y_ultimo = self.converter_coordenadas(
            ultimo_ponto['latitude'], ultimo_ponto['longitude']
        )
        
        pontos_discretizados.append({
            'ponto_id': len(pontos_discretizados),
            'segmento': len(dados_linha) - 1,
            'fracao_segmento': 1.0,
            'latitude': ultimo_ponto['latitude'],
            'longitude': ultimo_ponto['longitude'],
            'x': x_ultimo,
            'y': y_ultimo,
            'azimute': ultimo_ponto['azimute'],
            'progressiva_aprox': ultimo_ponto['Progressiva']
        })
        
        df_pontos = pd.DataFrame(pontos_discretizados)
        logger.info(f"Linha discretizada em {len(df_pontos)} pontos")
        
        return df_pontos

    def executar_krigagem_horaria(self, dados_sincronizados, pontos_linha, 
                                 variaveis_ambientais=None):
        """
        Realiza Krigagem Ordinária para cada hora e variável ambiental.
        
        Args:
            dados_sincronizados (pd.DataFrame): Dados sincronizados das estações
            pontos_linha (pd.DataFrame): Pontos discretizados da linha
            variaveis_ambientais (list): Lista de variáveis para krigagem
            
        Returns:
            dict: Resultados da krigagem para cada hora/variável
        """
        if variaveis_ambientais is None:
            variaveis_ambientais = config.VARIAVEIS_AMBIENTAIS
            
        logger.info("Iniciando krigagem horária...")
        logger.info(f"Variáveis: {variaveis_ambientais}")
        
        resultados_krigagem = {}
        
        # Obter timestamps únicos
        horas_unicas = dados_sincronizados.index.unique().sort_values()
        logger.info(f"Processando {len(horas_unicas)} timestamps")
        
        # Coordenadas dos pontos da linha
        coords_linha = pontos_linha[['x', 'y']].values
        
        # Processar cada timestamp
        for i, hora in enumerate(horas_unicas):
            if i % 100 == 0:  # Log a cada 100 horas
                logger.info(f"Processando hora {i+1}/{len(horas_unicas)}: {hora}")
            
            try:
                resultados_hora = self._processar_hora_krigagem(
                    dados_sincronizados, hora, coords_linha, variaveis_ambientais
                )
                resultados_krigagem[hora] = resultados_hora
                
            except Exception as e:
                logger.warning(f"Erro na krigagem para {hora}: {e}")
                # Preencher com NaN em caso de erro
                resultados_krigagem[hora] = self._criar_resultado_nan(
                    variaveis_ambientais, len(coords_linha)
                )
        
        logger.info("Krigagem horária concluída")
        return resultados_krigagem

    def _processar_hora_krigagem(self, dados_sincronizados, hora, coords_linha, 
                                variaveis_ambientais):
        """
        Processa krigagem para uma hora específica.
        
        Args:
            dados_sincronizados (pd.DataFrame): Dados das estações
            hora (datetime): Timestamp para processar
            coords_linha (np.array): Coordenadas dos pontos da linha
            variaveis_ambientais (list): Variáveis para interpolar
            
        Returns:
            dict: Resultados da krigagem para a hora
        """
        # Filtrar dados para a hora específica
        dados_hora = dados_sincronizados.loc[dados_sincronizados.index == hora]
        
        if dados_hora.empty:
            raise ValueError(f"Nenhum dado encontrado para {hora}")
        
        # Preparar coordenadas das estações
        coords_estacoes = []
        for _, row in dados_hora.iterrows():
            x, y = self.converter_coordenadas(row['latitude'], row['longitude'])
            coords_estacoes.append([x, y])
        
        coords_estacoes = np.array(coords_estacoes)
        
        if len(coords_estacoes) < 2:
            raise ValueError(f"Insuficientes estações ({len(coords_estacoes)}) para krigagem")
        
        resultados_hora = {}
        
        # Processar cada variável ambiental
        for variavel in variaveis_ambientais:
            try:
                if variavel not in dados_hora.columns:
                    logger.warning(f"Variável {variavel} não encontrada nos dados")
                    resultados_hora[variavel] = self._criar_resultado_nan_variavel(len(coords_linha))
                    continue
                
                valores_estacoes = dados_hora[variavel].values
                
                # Verificar e remover NaNs
                indices_validos = ~np.isnan(valores_estacoes)
                valores_validos = valores_estacoes[indices_validos]
                coords_validas = coords_estacoes[indices_validos]
                
                if len(valores_validos) < 2:
                    logger.warning(f"Dados insuficientes para {variavel} em {hora}")
                    resultados_hora[variavel] = self._criar_resultado_nan_variavel(len(coords_linha))
                    continue
                
                # Executar krigagem
                resultado_krigagem = self._executar_krigagem_variavel(
                    coords_validas, valores_validos, coords_linha, variavel
                )
                
                resultados_hora[variavel] = resultado_krigagem
                
            except Exception as e:
                logger.warning(f"Erro na krigagem de {variavel} em {hora}: {e}")
                resultados_hora[variavel] = self._criar_resultado_nan_variavel(len(coords_linha))
        
        return resultados_hora

    def _executar_krigagem_variavel(self, coords_estacoes, valores_estacoes, 
                                   coords_linha, variavel):
        """
        Executa krigagem para uma variável específica.
        
        Args:
            coords_estacoes (np.array): Coordenadas das estações
            valores_estacoes (np.array): Valores da variável nas estações
            coords_linha (np.array): Coordenadas dos pontos da linha
            variavel (str): Nome da variável
            
        Returns:
            dict: Resultados da krigagem (média e variância)
        """
        try:
            # Suprimir warnings da pykrige
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Ajustar variograma automaticamente
                melhor_modelo = self._ajustar_variograma_automatico(
                    coords_estacoes, valores_estacoes, variavel
                )
                
                # Criar objeto de krigagem
                OK = OrdinaryKriging(
                    coords_estacoes[:, 0],  # X coordinates
                    coords_estacoes[:, 1],  # Y coordinates
                    valores_estacoes,       # Values
                    variogram_model=melhor_modelo,
                    verbose=False,
                    enable_plotting=False,
                    coordinates_type='euclidean'
                )
                
                # Executar krigagem
                z_pred, ss_pred = OK.execute(
                    'points',
                    coords_linha[:, 0],
                    coords_linha[:, 1]
                )
                
                # Verificar qualidade dos resultados
                if hasattr(z_pred, 'data'):
                    z_pred = z_pred.data
                if hasattr(ss_pred, 'data'):
                    ss_pred = ss_pred.data
                
                # Validar resultados
                if np.any(np.isnan(z_pred)) or np.any(np.isnan(ss_pred)):
                    logger.warning(f"NaN encontrados nos resultados de {variavel}")
                
                # Garantir que variâncias sejam positivas
                ss_pred = np.maximum(ss_pred, 0)
                
                # Validar valores físicos por variável
                if variavel == 'radiacao_global':
                    z_pred = self._validar_radiacao_krigagem(z_pred)
                elif variavel == 'temperatura_ar':
                    z_pred = self._validar_temperatura_krigagem(z_pred)
                elif variavel in ['vento_u', 'vento_v', 'vento_velocidade']:
                    z_pred = self._validar_vento_krigagem(z_pred, variavel)
                
                return {
                    'media': z_pred,
                    'variancia': ss_pred,
                    'desvio_padrao': np.sqrt(ss_pred)
                }
                
        except Exception as e:
            logger.error(f"Erro na krigagem de {variavel}: {e}")
            raise
    
    def _ajustar_variograma_automatico(self, coords_estacoes, valores_estacoes, variavel):
        """
        Ajusta automaticamente o modelo de variograma testando diferentes opções.
        
        Args:
            coords_estacoes (np.array): Coordenadas das estações
            valores_estacoes (np.array): Valores da variável
            variavel (str): Nome da variável
            
        Returns:
            str: Melhor modelo de variograma
        """
        # Modelos específicos por variável - radiação tem comportamento especial
        if variavel == 'radiacao_global':
            # Para radiação, preferir modelos mais suaves que reduzem variância
            modelos_testar = ['gaussian', 'spherical', 'exponential', 'linear']
        else:
            modelos_testar = ['linear', 'power', 'gaussian', 'spherical', 'exponential']
            
        melhor_modelo = config.MODELO_VARIOGRAMA  # Default
        melhor_score = float('inf')
        melhor_variancia_media = float('inf')
        
        # Se temos muitas poucas estações, usar modelo linear
        if len(coords_estacoes) < 4:
            return 'linear'
        
        # Calcular variância dos dados para normalização
        variancia_dados = np.var(valores_estacoes)
        
        # Dividir dados em treino e validação (cross-validation simples)
        n_estacoes = len(coords_estacoes)
        indices = np.arange(n_estacoes)
        np.random.shuffle(indices)
        
        # Usar 80% para treino, 20% para validação
        n_treino = max(2, int(0.8 * n_estacoes))
        indices_treino = indices[:n_treino]
        indices_validacao = indices[n_treino:]
        
        if len(indices_validacao) == 0:
            return melhor_modelo
        
        coords_treino = coords_estacoes[indices_treino]
        valores_treino = valores_estacoes[indices_treino]
        coords_validacao = coords_estacoes[indices_validacao]
        valores_validacao = valores_estacoes[indices_validacao]
        
        # Testar cada modelo
        for modelo in modelos_testar:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Criar krigagem com dados de treino
                    OK = OrdinaryKriging(
                        coords_treino[:, 0],
                        coords_treino[:, 1],
                        valores_treino,
                        variogram_model=modelo,
                        verbose=False,
                        enable_plotting=False,
                        coordinates_type='euclidean'
                    )
                    
                    # Predizer nos pontos de validação
                    pred_validacao, var_validacao = OK.execute(
                        'points',
                        coords_validacao[:, 0],
                        coords_validacao[:, 1]
                    )
                    
                    if hasattr(pred_validacao, 'data'):
                        pred_validacao = pred_validacao.data
                    if hasattr(var_validacao, 'data'):
                        var_validacao = var_validacao.data
                    
                    # Calcular métricas
                    if not np.any(np.isnan(pred_validacao)) and not np.any(np.isnan(var_validacao)):
                        rmse = np.sqrt(mean_squared_error(valores_validacao, pred_validacao))
                        variancia_media = np.mean(var_validacao)
                        
                        # Para radiação, penalizar modelos com variância muito alta
                        if variavel == 'radiacao_global':
                            # Score combinado: RMSE + penalidade por variância alta
                            penalidade_variancia = min(variancia_media / variancia_dados, 10.0)
                            score_combinado = rmse * (1 + 0.1 * penalidade_variancia)
                        else:
                            score_combinado = rmse
                        
                        if score_combinado < melhor_score:
                            melhor_score = score_combinado
                            melhor_modelo = modelo
                            melhor_variancia_media = variancia_media
                            
            except Exception:
                continue
        
        # Log apenas para primeira vez por variável (evitar spam)
        if not hasattr(self, '_variograma_logged'):
            self._variograma_logged = set()
        
        if variavel not in self._variograma_logged:
            if variavel == 'radiacao_global':
                logger.info(f"Melhor modelo de variograma para {variavel}: {melhor_modelo} "
                           f"(RMSE: {melhor_score:.3f}, Var.média: {melhor_variancia_media:.1f})")
            else:
                logger.info(f"Melhor modelo de variograma para {variavel}: {melhor_modelo} (RMSE: {melhor_score:.3f})")
            self._variograma_logged.add(variavel)
        
        return melhor_modelo

    def _validar_radiacao_krigagem(self, valores_radiacao):
        """
        Valida valores de radiação da krigagem e anula valores fisicamente impossíveis.
        
        Args:
            valores_radiacao (np.array): Valores de radiação interpolados
            
        Returns:
            np.array: Valores validados (valores inválidos substituídos por NaN)
        """
        valores_validados = valores_radiacao.copy()
        
        # Limites físicos para radiação solar (W/m²)
        # Radiação máxima teórica ao nível do mar é ~1400 W/m²
        # Radiação mínima é 0 (não pode ser negativa)
        limite_min = 0.0
        limite_max = 1400.0
        
        # Contar valores inválidos antes da correção
        mask_invalidos = (valores_validados < limite_min) | (valores_validados > limite_max)
        num_invalidos = np.sum(mask_invalidos)
        
        if num_invalidos > 0:
            # Log dos valores problemáticos para análise
            valores_problematicos = valores_validados[mask_invalidos]
            logger.warning(f"Krigagem de radiação: {num_invalidos} valores inválidos detectados")
            logger.warning(f"Valores problemáticos: min={valores_problematicos.min():.1f}, "
                          f"max={valores_problematicos.max():.1f} W/m²")
            
            # Substituir valores inválidos por NaN
            valores_validados[mask_invalidos] = np.nan
            
            logger.info(f"Valores de radiação fora do intervalo [{limite_min}, {limite_max}] W/m² "
                       f"foram anulados ({num_invalidos} pontos)")
        
        return valores_validados

    def _validar_temperatura_krigagem(self, valores_temperatura):
        """
        Valida valores de temperatura da krigagem e anula valores fisicamente impossíveis.
        
        Args:
            valores_temperatura (np.array): Valores de temperatura interpolados
            
        Returns:
            np.array: Valores validados (valores inválidos substituídos por NaN)
        """
        valores_validados = valores_temperatura.copy()
        
        # Limites físicos para temperatura do ar (°C)
        # Considerando faixa realista para condições meteorológicas no Brasil
        limite_min = -10.0  # Temperatura mínima extrema
        limite_max = 55.0   # Temperatura máxima extrema
        
        # Contar valores inválidos antes da correção
        mask_invalidos = (valores_validados < limite_min) | (valores_validados > limite_max)
        num_invalidos = np.sum(mask_invalidos)
        
        if num_invalidos > 0:
            # Log dos valores problemáticos para análise
            valores_problematicos = valores_validados[mask_invalidos]
            logger.warning(f"Krigagem de temperatura: {num_invalidos} valores inválidos detectados")
            logger.warning(f"Valores problemáticos: min={valores_problematicos.min():.1f}, "
                          f"max={valores_problematicos.max():.1f} °C")
            
            # Substituir valores inválidos por NaN
            valores_validados[mask_invalidos] = np.nan
            
            logger.info(f"Valores de temperatura fora do intervalo [{limite_min}, {limite_max}] °C "
                       f"foram anulados ({num_invalidos} pontos)")
        
        return valores_validados

    def _validar_vento_krigagem(self, valores_vento, variavel):
        """
        Valida valores de vento da krigagem e anula valores fisicamente impossíveis.
        
        Args:
            valores_vento (np.array): Valores de vento interpolados
            variavel (str): Nome da variável (vento_u, vento_v, vento_velocidade)
            
        Returns:
            np.array: Valores validados (valores inválidos substituídos por NaN)
        """
        valores_validados = valores_vento.copy()
        
        # Limites físicos para vento
        if variavel == 'vento_velocidade':
            # Velocidade do vento não pode ser negativa e raramente excede 50 m/s
            limite_min = 0.0
            limite_max = 50.0
        else:  # vento_u, vento_v (componentes)
            # Componentes podem ser negativas, mas limitadas por velocidade máxima
            limite_min = -50.0
            limite_max = 50.0
        
        # Contar valores inválidos antes da correção
        mask_invalidos = (valores_validados < limite_min) | (valores_validados > limite_max)
        num_invalidos = np.sum(mask_invalidos)
        
        if num_invalidos > 0:
            # Log dos valores problemáticos para análise
            valores_problematicos = valores_validados[mask_invalidos]
            logger.warning(f"Krigagem de {variavel}: {num_invalidos} valores inválidos detectados")
            logger.warning(f"Valores problemáticos: min={valores_problematicos.min():.1f}, "
                          f"max={valores_problematicos.max():.1f} m/s")
            
            # Substituir valores inválidos por NaN
            valores_validados[mask_invalidos] = np.nan
            
            logger.info(f"Valores de {variavel} fora do intervalo [{limite_min}, {limite_max}] m/s "
                       f"foram anulados ({num_invalidos} pontos)")
        
        return valores_validados

    def _criar_resultado_nan(self, variaveis_ambientais, num_pontos):
        """Cria resultado preenchido com NaN para todas as variáveis."""
        resultado = {}
        for variavel in variaveis_ambientais:
            resultado[variavel] = self._criar_resultado_nan_variavel(num_pontos)
        return resultado

    def _criar_resultado_nan_variavel(self, num_pontos):
        """Cria resultado preenchido com NaN para uma variável."""
        return {
            'media': np.full(num_pontos, np.nan),
            'variancia': np.full(num_pontos, np.nan),
            'desvio_padrao': np.full(num_pontos, np.nan)
        }

    def validar_resultados_krigagem(self, resultados_krigagem, pontos_linha):
        """
        Valida os resultados da krigagem e fornece estatísticas.
        
        Args:
            resultados_krigagem (dict): Resultados da krigagem
            pontos_linha (pd.DataFrame): Pontos da linha
            
        Returns:
            dict: Estatísticas de validação
        """
        logger.info("Validando resultados da krigagem...")
        
        estatisticas = {
            'total_horas': len(resultados_krigagem),
            'total_pontos': len(pontos_linha),
            'variaveis': {},
            'horas_com_erro': 0,
            'qualidade_geral': 'boa'
        }
        
        horas_com_nan = 0
        
        for hora, resultados_hora in resultados_krigagem.items():
            hora_tem_nan = False
            
            for variavel, resultado_var in resultados_hora.items():
                if variavel not in estatisticas['variaveis']:
                    estatisticas['variaveis'][variavel] = {
                        'media_min': np.inf,
                        'media_max': -np.inf,
                        'variancia_media': 0,
                        'pontos_com_nan': 0
                    }
                
                media = resultado_var['media']
                variancia = resultado_var['variancia']
                
                # Contar NaNs
                nan_count = np.sum(np.isnan(media))
                if nan_count > 0:
                    hora_tem_nan = True
                    estatisticas['variaveis'][variavel]['pontos_com_nan'] += nan_count
                
                # Estatísticas dos valores válidos
                valores_validos = media[~np.isnan(media)]
                if len(valores_validos) > 0:
                    estatisticas['variaveis'][variavel]['media_min'] = min(
                        estatisticas['variaveis'][variavel]['media_min'],
                        np.min(valores_validos)
                    )
                    estatisticas['variaveis'][variavel]['media_max'] = max(
                        estatisticas['variaveis'][variavel]['media_max'],
                        np.max(valores_validos)
                    )
                
                # Variância média
                var_validas = variancia[~np.isnan(variancia)]
                if len(var_validas) > 0:
                    estatisticas['variaveis'][variavel]['variancia_media'] += np.mean(var_validas)
            
            if hora_tem_nan:
                horas_com_nan += 1
        
        estatisticas['horas_com_erro'] = horas_com_nan
        
        # Finalizar cálculo de variância média
        for variavel in estatisticas['variaveis']:
            estatisticas['variaveis'][variavel]['variancia_media'] /= len(resultados_krigagem)
        
        # Determinar qualidade geral
        taxa_erro = horas_com_nan / len(resultados_krigagem)
        if taxa_erro > 0.1:
            estatisticas['qualidade_geral'] = 'ruim'
        elif taxa_erro > 0.05:
            estatisticas['qualidade_geral'] = 'regular'
        
        logger.info(f"Validação concluída: {taxa_erro:.1%} horas com problemas")
        logger.info(f"Qualidade geral: {estatisticas['qualidade_geral']}")
        
        return estatisticas