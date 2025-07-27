# Módulo para validação robusta de dados
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
import config

logger = logging.getLogger(__name__)

class DataValidator:
    """Classe para validação robusta de dados meteorológicos e técnicos."""
    
    def __init__(self):
        self.limites_fisicos = self._definir_limites_fisicos()
        self.tolerancias = self._definir_tolerancias()
        
    def _definir_limites_fisicos(self) -> Dict[str, Dict[str, float]]:
        """Define limites físicos realistas para as variáveis."""
        return {
            'temperatura_ar': {
                'min_absoluto': -60.0,
                'max_absoluto': 60.0,
                'min_tipico': -40.0,
                'max_tipico': 50.0
            },
            'radiacao_global': {
                'min_absoluto': 0.0,
                'max_absoluto': 1500.0,  # W/m²
                'min_tipico': 0.0,
                'max_tipico': 1200.0
            },
            'vento_velocidade': {
                'min_absoluto': 0.0,
                'max_absoluto': 60.0,
                'min_tipico': 0.0,
                'max_tipico': 25.0
            },
            'vento_direcao': {
                'min_absoluto': 0.0,
                'max_absoluto': 360.0,
                'min_tipico': 0.0,
                'max_tipico': 360.0
            }
        }
    
    def _definir_tolerancias(self) -> Dict[str, float]:
        """Define tolerâncias para detecção de outliers."""
        return {
            'temperatura_ar': 3.0,  # desvios padrão
            'radiacao_global': 3.0,
            'vento_velocidade': 3.0,
            'vento_direcao': None  # Não aplicável para direção
        }
    
    def validar_dados_meteorologicos(self, df: pd.DataFrame, nome_estacao: str = "") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Valida dados meteorológicos com relatório detalhado.
        
        Args:
            df: DataFrame com dados meteorológicos
            nome_estacao: Nome da estação para logs
            
        Returns:
            Tuple[DataFrame limpo, relatório de validação]
        """
        relatorio = {
            'estacao': nome_estacao,
            'registros_originais': len(df),
            'registros_validos': 0,
            'problemas_encontrados': {},
            'estatisticas_limpeza': {}
        }
        
        if df.empty:
            relatorio['erro'] = 'DataFrame vazio'
            return df, relatorio
            
        df_limpo = df.copy()
        
        # 1. Validar limites físicos
        df_limpo, problemas_limites = self._validar_limites_fisicos(df_limpo)
        relatorio['problemas_encontrados']['limites_fisicos'] = problemas_limites
        
        # 2. Detectar e tratar outliers
        df_limpo, problemas_outliers = self._detectar_outliers(df_limpo)
        relatorio['problemas_encontrados']['outliers'] = problemas_outliers
        
        # 3. Validar consistência temporal
        df_limpo, problemas_temporais = self._validar_consistencia_temporal(df_limpo)
        relatorio['problemas_encontrados']['consistencia_temporal'] = problemas_temporais
        
        # 4. Validar completude dos dados
        completude = self._analisar_completude(df_limpo)
        relatorio['completude'] = completude
        
        relatorio['registros_validos'] = len(df_limpo)
        relatorio['taxa_aproveitamento'] = len(df_limpo) / len(df) if len(df) > 0 else 0
        
        # Log do resumo
        if nome_estacao:
            logger.info(f"Estação {nome_estacao}: {relatorio['registros_validos']}/{relatorio['registros_originais']} "
                       f"registros válidos ({relatorio['taxa_aproveitamento']:.1%})")
        
        return df_limpo, relatorio
    
    def _validar_limites_fisicos(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Valida limites físicos das variáveis."""
        problemas = {}
        df_result = df.copy()
        
        for variavel, limites in self.limites_fisicos.items():
            if variavel not in df_result.columns:
                continue
                
            # Contar problemas antes da limpeza
            mask_min = df_result[variavel] < limites['min_absoluto']
            mask_max = df_result[variavel] > limites['max_absoluto']
            
            problemas[f'{variavel}_abaixo_minimo'] = mask_min.sum()
            problemas[f'{variavel}_acima_maximo'] = mask_max.sum()
            
            # Remover registros fora dos limites
            df_result = df_result[~(mask_min | mask_max)]
        
        return df_result, problemas
    
    def _detectar_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Detecta e remove outliers usando método IQR e Z-score."""
        problemas = {}
        df_result = df.copy()
        
        for variavel, tolerancia in self.tolerancias.items():
            if variavel not in df_result.columns or tolerancia is None:
                continue
                
            valores = df_result[variavel].dropna()
            if len(valores) < 10:  # Dados insuficientes para análise
                continue
            
            # Método IQR
            Q1 = valores.quantile(0.25)
            Q3 = valores.quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            
            # Método Z-score
            z_scores = np.abs((valores - valores.mean()) / valores.std())
            
            # Combinação dos métodos
            mask_iqr = (df_result[variavel] < limite_inferior) | (df_result[variavel] > limite_superior)
            mask_zscore = z_scores > tolerancia
            mask_outliers = mask_iqr & mask_zscore  # Ambos os métodos devem detectar
            
            problemas[f'{variavel}_outliers'] = mask_outliers.sum()
            
            # Remover outliers
            df_result = df_result[~mask_outliers]
        
        return df_result, problemas
    
    def _validar_consistencia_temporal(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Valida consistência temporal dos dados."""
        problemas = {}
        df_result = df.copy()
        
        if not isinstance(df_result.index, pd.DatetimeIndex):
            problemas['indice_nao_temporal'] = 1
            return df_result, problemas
        
        # Verificar gaps temporais excessivos
        if len(df_result) > 1:
            intervalos = df_result.index[1:] - df_result.index[:-1]
            intervalo_mediano = intervalos.median()
            
            # Gaps maiores que 3x o intervalo mediano
            gaps_grandes = intervalos > (3 * intervalo_mediano)
            problemas['gaps_temporais_grandes'] = gaps_grandes.sum()
        
        # Verificar duplicatas temporais
        duplicatas = df_result.index.duplicated()
        problemas['timestamps_duplicados'] = duplicatas.sum()
        
        if duplicatas.any():
            df_result = df_result[~duplicatas]
        
        return df_result, problemas
    
    def _analisar_completude(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analisa completude dos dados por variável."""
        if df.empty:
            return {}
        
        completude = {}
        variaveis_importantes = ['temperatura_ar', 'radiacao_global', 'vento_velocidade', 'vento_direcao']
        
        for variavel in variaveis_importantes:
            if variavel in df.columns:
                valores_validos = df[variavel].notna().sum()
                completude[variavel] = valores_validos / len(df)
            else:
                completude[variavel] = 0.0
        
        return completude
    
    def validar_parametros_condutor(self, parametros: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valida parâmetros do condutor."""
        erros = []
        
        # Parâmetros obrigatórios
        obrigatorios = config.PARAMETROS_CABO_OBRIGATORIOS
        for param in obrigatorios:
            if param not in parametros:
                erros.append(f"Parâmetro obrigatório '{param}' ausente")
                continue
            
            valor = parametros[param]
            if not isinstance(valor, (int, float)) or valor <= 0:
                erros.append(f"Parâmetro '{param}' deve ser um número positivo, recebido: {valor}")
        
        # Validações específicas
        if 'diametro' in parametros:
            diametro = parametros['diametro']
            if diametro < 0.005 or diametro > 0.1:  # 5mm a 100mm
                erros.append(f"Diâmetro fora da faixa esperada (5-100mm): {diametro*1000:.1f}mm")
        
        if 'resistencia_ac_25' in parametros and 'resistencia_ac_75' in parametros:
            r25 = parametros['resistencia_ac_25']
            r75 = parametros['resistencia_ac_75']
            if r75 <= r25:
                erros.append("Resistência a 75°C deve ser maior que a 25°C")
        
        if 'emissividade' in parametros:
            emiss = parametros['emissividade']
            if emiss < 0.1 or emiss > 1.0:
                erros.append(f"Emissividade fora da faixa física (0.1-1.0): {emiss}")
        
        if 'absortividade' in parametros:
            absort = parametros['absortividade']
            if absort < 0.1 or absort > 1.0:
                erros.append(f"Absortividade fora da faixa física (0.1-1.0): {absort}")
        
        return len(erros) == 0, erros
    
    def validar_dados_linha(self, dados_linha: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Valida dados do traçado da linha."""
        erros = []
        
        if dados_linha.empty:
            erros.append("Dados da linha estão vazios")
            return False, erros
        
        # Verificar colunas obrigatórias
        colunas_obrigatorias = config.COLUNAS_TRASSADO_OBRIGATORIAS
        for coluna in colunas_obrigatorias:
            if coluna not in dados_linha.columns:
                erros.append(f"Coluna obrigatória '{coluna}' ausente")
        
        if erros:
            return False, erros
        
        # Validar coordenadas
        lats = dados_linha['latitude']
        lons = dados_linha['longitude']
        
        if not lats.between(-90, 90).all():
            erros.append("Latitudes fora da faixa válida (-90 a 90)")
        
        if not lons.between(-180, 180).all():
            erros.append("Longitudes fora da faixa válida (-180 a 180)")
        
        # Validar azimutes
        if 'azimute' in dados_linha.columns:
            azimutes = dados_linha['azimute']
            if not azimutes.between(0, 360).all():
                erros.append("Azimutes fora da faixa válida (0 a 360 graus)")
        
        # Verificar sequência lógica da progressiva
        if 'Progressiva' in dados_linha.columns:
            progressivas = dados_linha['Progressiva']
            if not progressivas.is_monotonic_increasing:
                erros.append("Progressivas não estão em ordem crescente")
        
        return len(erros) == 0, erros
    
    def gerar_relatorio_qualidade(self, relatorios_estacoes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Gera relatório consolidado de qualidade dos dados."""
        if not relatorios_estacoes:
            return {'erro': 'Nenhum relatório de estação fornecido'}
        
        # Estatísticas consolidadas
        total_registros_originais = sum(r.get('registros_originais', 0) for r in relatorios_estacoes)
        total_registros_validos = sum(r.get('registros_validos', 0) for r in relatorios_estacoes)
        
        taxa_geral = total_registros_validos / total_registros_originais if total_registros_originais > 0 else 0
        
        # Problemas mais comuns
        tipos_problemas = {}
        for relatorio in relatorios_estacoes:
            for categoria, problemas in relatorio.get('problemas_encontrados', {}).items():
                if categoria not in tipos_problemas:
                    tipos_problemas[categoria] = 0
                if isinstance(problemas, dict):
                    tipos_problemas[categoria] += sum(problemas.values())
                else:
                    tipos_problemas[categoria] += problemas
        
        # Completude média por variável
        completude_media = {}
        variaveis = ['temperatura_ar', 'radiacao_global', 'vento_velocidade', 'vento_direcao']
        
        for variavel in variaveis:
            valores = [r.get('completude', {}).get(variavel, 0) for r in relatorios_estacoes if 'completude' in r]
            completude_media[variavel] = np.mean(valores) if valores else 0.0
        
        # Classificação da qualidade geral
        if taxa_geral >= 0.9:
            qualidade_geral = "Excelente"
        elif taxa_geral >= 0.8:
            qualidade_geral = "Boa"
        elif taxa_geral >= 0.7:
            qualidade_geral = "Regular"
        else:
            qualidade_geral = "Ruim"
        
        return {
            'resumo_geral': {
                'num_estacoes': len(relatorios_estacoes),
                'registros_originais': total_registros_originais,
                'registros_validos': total_registros_validos,
                'taxa_aproveitamento': taxa_geral,
                'qualidade_geral': qualidade_geral
            },
            'problemas_por_tipo': tipos_problemas,
            'completude_por_variavel': completude_media,
            'estacoes_problematicas': [
                r['estacao'] for r in relatorios_estacoes 
                if r.get('taxa_aproveitamento', 0) < 0.5
            ],
            'recomendacoes': self._gerar_recomendacoes_qualidade(taxa_geral, tipos_problemas, completude_media)
        }
    
    def _gerar_recomendacoes_qualidade(self, taxa_geral: float, problemas: Dict[str, int], 
                                     completude: Dict[str, float]) -> List[str]:
        """Gera recomendações baseadas na qualidade dos dados."""
        recomendacoes = []
        
        if taxa_geral < 0.7:
            recomendacoes.append("Taxa de aproveitamento baixa - revisar procedimentos de coleta")
        
        # Problemas específicos
        if problemas.get('outliers', {}) or sum(v for k, v in problemas.items() if 'outliers' in k):
            recomendacoes.append("Muitos outliers detectados - verificar calibração dos sensores")
        
        if problemas.get('limites_fisicos', {}) or sum(v for k, v in problemas.items() if 'limite' in k):
            recomendacoes.append("Valores fora de limites físicos - verificar funcionamento dos equipamentos")
        
        # Completude
        for variavel, compl in completude.items():
            if compl < 0.8:
                recomendacoes.append(f"Completude baixa para {variavel} ({compl:.1%}) - verificar sensor")
        
        if not recomendacoes:
            recomendacoes.append("Qualidade dos dados está adequada para análise")
        
        return recomendacoes