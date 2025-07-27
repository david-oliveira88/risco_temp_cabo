# Orquestrador principal da execução
import logging
import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

# Importar módulos do framework
import config
from data_loader import DataLoader
from geoprocessing import GeoProcessor
from thermal_model import CigreModeloTermico
from simulation import MonteCarloSimulator
from risk_analysis import RiskAnalyzer

# Configurar logging
def configurar_logging():
    """Configura o sistema de logging."""
    # Criar diretório de logs se não existir
    log_dir = os.path.join(config.BASE_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Nome do arquivo de log com timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'analisador_risco_{timestamp}.log')
    
    # Configurar formatação
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # Handler para arquivo
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    console_handler.setFormatter(formatter)
    
    # Configurar logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

class AnalisadorRiscoTermico:
    """
    Classe principal para executar a análise de risco térmico de cabos elétricos.
    """
    
    def __init__(self):
        """Inicializa o analisador."""
        self.data_loader = None
        self.geo_processor = None
        self.modelo_termico = None
        self.simulador_mc = None
        self.risk_analyzer = None
        
        # Dados carregados
        self.parametros_cabo = None
        self.dados_linha = None
        self.dados_estacoes = None
        self.dados_sincronizados = None
        self.pontos_linha = None
        self.resultados_krigagem = None
        
        # Resultados finais
        self.resultados_finais = []
        
        self.logger = logging.getLogger(__name__)

    def executar_analise_completa(self, salvar_intermediarios=True):
        """
        Executa a análise completa de risco térmico.
        
        Args:
            salvar_intermediarios (bool): Se deve salvar resultados intermediários
            
        Returns:
            bool: True se executou com sucesso, False caso contrário
        """
        self.logger.info("="*80)
        self.logger.info("INICIANDO ANÁLISE DE RISCO TÉRMICO DE CABOS ELÉTRICOS")
        self.logger.info("="*80)
        
        inicio_total = time.time()
        
        try:
            # Etapa 1: Validar configuração
            self.logger.info("Etapa 1: Validando configuração...")
            self._validar_configuracao()
            
            # Etapa 2: Carregar dados
            self.logger.info("Etapa 2: Carregando dados...")
            self._carregar_dados()
            
            # Etapa 3: Processamento geoespacial
            self.logger.info("Etapa 3: Processamento geoespacial...")
            self._processar_dados_geoespaciais()
            
            # Etapa 4: Ejecutar krigagem
            self.logger.info("Etapa 4: Executando krigagem...")
            self._executar_krigagem()
            
            if salvar_intermediarios:
                self._salvar_dados_intermediarios()
            
            # Etapa 5: Inicializar modelos
            self.logger.info("Etapa 5: Inicializando modelos térmicos...")
            self._inicializar_modelos()
            
            # Etapa 6: Simulação Monte Carlo
            self.logger.info("Etapa 6: Executando simulações...")
            self._executar_simulacoes()
            
            # Etapa 7: Salvar resultados
            self.logger.info("Etapa 7: Salvando resultados...")
            self._salvar_resultados_finais()
            
            tempo_total = time.time() - inicio_total
            self.logger.info(f"Análise concluída com sucesso em {tempo_total:.1f} segundos")
            
            # Resumo final
            self._gerar_resumo_final()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro durante a análise: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def _validar_configuracao(self):
        """Valida se todos os arquivos e diretórios necessários existem."""
        try:
            config.validar_configuracao()
            self.logger.info("Configuração validada com sucesso")
        except FileNotFoundError as e:
            self.logger.error(f"Erro na configuração: {e}")
            raise

    def _carregar_dados(self):
        """Carrega todos os dados necessários."""
        self.data_loader = DataLoader()
        
        try:
            # Carregar todos os dados
            self.data_loader.carregar_todos_dados()
            
            # Acessar dados carregados
            self.parametros_cabo = self.data_loader.parametros_cabo
            self.dados_linha = self.data_loader.dados_linha
            self.dados_estacoes = self.data_loader.dados_estacoes
            self.dados_sincronizados = self.data_loader.dados_sincronizados
            
            # Log do resumo
            resumo = self.data_loader.obter_resumo_dados()
            self.logger.info(f"Dados carregados: {resumo['num_estacoes']} estações, "
                           f"{resumo['dados_linha']} pontos da linha, "
                           f"{resumo['dados_sincronizados']} registros sincronizados")
            
            if resumo['periodo_dados']:
                self.logger.info(f"Período dos dados: {resumo['periodo_dados']['inicio']} "
                               f"até {resumo['periodo_dados']['fim']}")
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados: {e}")
            raise

    def _processar_dados_geoespaciais(self):
        """Processa dados geoespaciais e discretiza a linha."""
        self.geo_processor = GeoProcessor()
        
        try:
            # Discretizar linha
            self.pontos_linha = self.geo_processor.discretizar_linha(
                self.dados_linha, 
                config.DISTANCIA_DISCRETIZACAO
            )
            
            self.logger.info(f"Linha discretizada em {len(self.pontos_linha)} pontos")
            
        except Exception as e:
            self.logger.error(f"Erro no processamento geoespacial: {e}")
            raise

    def _executar_krigagem(self):
        """Executa a krigagem para todas as horas e variáveis."""
        try:
            self.resultados_krigagem = self.geo_processor.executar_krigagem_horaria(
                self.dados_sincronizados,
                self.pontos_linha,
                config.VARIAVEIS_AMBIENTAIS
            )
            
            # Validar resultados
            estatisticas_krigagem = self.geo_processor.validar_resultados_krigagem(
                self.resultados_krigagem, 
                self.pontos_linha
            )
            
            self.logger.info(f"Krigagem concluída: {estatisticas_krigagem['total_horas']} horas processadas")
            self.logger.info(f"Qualidade da krigagem: {estatisticas_krigagem['qualidade_geral']}")
            
        except Exception as e:
            self.logger.error(f"Erro na krigagem: {e}")
            raise

    def _salvar_dados_intermediarios(self):
        """Salva dados intermediários para verificação."""
        try:
            # Salvar pontos da linha
            pontos_file = os.path.join(config.SAIDA_DIR, 'pontos_linha.csv')
            self.pontos_linha.to_csv(pontos_file, index=False)
            
            # Salvar estatísticas da krigagem (resumo)
            stats_file = os.path.join(config.SAIDA_DIR, 'estatisticas_krigagem.csv')
            stats_data = []
            
            for hora, resultados_hora in list(self.resultados_krigagem.items())[:10]:  # Apenas primeiras 10 horas
                for variavel, resultado_var in resultados_hora.items():
                    stats_data.append({
                        'hora': hora,
                        'variavel': variavel,
                        'media_min': np.nanmin(resultado_var['media']),
                        'media_max': np.nanmax(resultado_var['media']),
                        'variancia_media': np.nanmean(resultado_var['variancia'])
                    })
            
            pd.DataFrame(stats_data).to_csv(stats_file, index=False)
            self.logger.info("Dados intermediários salvos")
            
        except Exception as e:
            self.logger.warning(f"Erro ao salvar dados intermediários: {e}")

    def _inicializar_modelos(self):
        """Inicializa os modelos térmico e de simulação."""
        try:
            # Modelo térmico CIGRE
            self.modelo_termico = CigreModeloTermico(self.parametros_cabo)
            
            # Simulador Monte Carlo
            self.simulador_mc = MonteCarloSimulator(self.modelo_termico)
            
            # Analisador de risco
            self.risk_analyzer = RiskAnalyzer()
            
            self.logger.info("Modelos inicializados com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar modelos: {e}")
            raise

    def _executar_simulacoes(self):
        """Executa as simulações Monte Carlo para todos os pontos e horas."""
        total_combinacoes = len(self.pontos_linha) * len(self.resultados_krigagem)
        combinacoes_processadas = 0
        
        self.logger.info(f"Iniciando simulações para {total_combinacoes} combinações ponto-hora")
        
        # Configurações da simulação
        corrente_operacao = config.CORRENTE_PADRAO
        temperatura_max_projeto = config.TEMPERATURA_MAX_PROJETO
        
        # Iterar sobre cada ponto da linha
        for idx_ponto, ponto in self.pontos_linha.iterrows():
            if idx_ponto % 10 == 0:  # Log a cada 10 pontos
                self.logger.info(f"Processando ponto {idx_ponto + 1}/{len(self.pontos_linha)}")
            
            # Iterar sobre cada hora
            for hora, dados_hora_krigagem in self.resultados_krigagem.items():
                try:
                    # Verificar se há dados válidos para este ponto/hora
                    if not self._verificar_dados_validos(dados_hora_krigagem, idx_ponto):
                        combinacoes_processadas += 1
                        continue
                    
                    # Extrair médias e desvios para este ponto
                    medias_ambientais, desvios_ambientais = self._extrair_dados_krigagem(
                        dados_hora_krigagem, idx_ponto
                    )
                    
                    # Executar simulação Monte Carlo
                    resultado_mc = self.simulador_mc.executar_simulacao(
                        medias_ambientais=medias_ambientais,
                        desvios_ambientais=desvios_ambientais,
                        azimute_linha=ponto['azimute'],
                        corrente=corrente_operacao,
                        num_iteracoes=config.NUM_ITERACOES_MC
                    )
                    
                    # Análise de risco
                    if resultado_mc['iteracoes_validas'] > 0:
                        temp_p90 = resultado_mc['estatisticas']['percentil_90']
                        risco_termico = self.risk_analyzer.calcular_risco_termico(
                            resultado_mc['temperaturas'], 
                            temperatura_max_projeto
                        )
                        
                        # Calcular ampacidade
                        ampacidade = self.modelo_termico.calcular_ampacidade(
                            temperatura_max_projeto,
                            medias_ambientais['radiacao_global'],
                            ponto['azimute'],
                            medias_ambientais['vento_velocidade'] if 'vento_velocidade' in medias_ambientais else 1.0,
                            0,  # Ângulo do vento (simplificado)
                            medias_ambientais['temperatura_ar']
                        )
                        
                        # Armazenar resultado
                        self.resultados_finais.append({
                            'hora': hora,
                            'ponto_id': idx_ponto,
                            'latitude': ponto['latitude'],
                            'longitude': ponto['longitude'],
                            'progressiva': ponto.get('progressiva_aprox', idx_ponto),
                            'azimute': ponto['azimute'],
                            'corrente_operacao': corrente_operacao,
                            'temperatura_ar_media': medias_ambientais['temperatura_ar'],
                            'radiacao_media': medias_ambientais['radiacao_global'],
                            'vento_u_media': medias_ambientais['vento_u'],
                            'vento_v_media': medias_ambientais['vento_v'],
                            'temperatura_condutor_media': resultado_mc['estatisticas']['media'],
                            'temperatura_condutor_p90': temp_p90,
                            'temperatura_condutor_p95': resultado_mc['estatisticas']['percentil_95'],
                            'risco_termico': risco_termico,
                            'ampacidade_calculada': ampacidade,
                            'iteracoes_validas': resultado_mc['iteracoes_validas'],
                            'taxa_sucesso_mc': resultado_mc['taxa_sucesso']
                        })
                    
                except Exception as e:
                    self.logger.warning(f"Erro na simulação para ponto {idx_ponto}, hora {hora}: {e}")
                
                combinacoes_processadas += 1
                
                # Log de progresso
                if combinacoes_processadas % 1000 == 0:
                    progresso = 100 * combinacoes_processadas / total_combinacoes
                    self.logger.info(f"Progresso geral: {progresso:.1f}% "
                                   f"({combinacoes_processadas}/{total_combinacoes})")
        
        self.logger.info(f"Simulações concluídas: {len(self.resultados_finais)} resultados válidos")

    def _verificar_dados_validos(self, dados_hora_krigagem, idx_ponto):
        """Verifica se há dados válidos para um ponto específico."""
        for variavel in config.VARIAVEIS_AMBIENTAIS:
            if variavel not in dados_hora_krigagem:
                return False
            
            media = dados_hora_krigagem[variavel]['media'][idx_ponto]
            variancia = dados_hora_krigagem[variavel]['variancia'][idx_ponto]
            
            if np.isnan(media) or np.isnan(variancia):
                return False
        
        return True

    def _extrair_dados_krigagem(self, dados_hora_krigagem, idx_ponto):
        """Extrai médias e desvios padrão dos resultados da krigagem."""
        medias_ambientais = {}
        desvios_ambientais = {}
        
        for variavel in config.VARIAVEIS_AMBIENTAIS:
            medias_ambientais[variavel] = dados_hora_krigagem[variavel]['media'][idx_ponto]
            desvios_ambientais[variavel] = np.sqrt(dados_hora_krigagem[variavel]['variancia'][idx_ponto])
        
        # Calcular velocidade do vento para referência
        u = medias_ambientais['vento_u']
        v = medias_ambientais['vento_v']
        medias_ambientais['vento_velocidade'] = np.sqrt(u**2 + v**2)
        
        return medias_ambientais, desvios_ambientais

    def _salvar_resultados_finais(self):
        """Salva os resultados finais em arquivo CSV."""
        if not self.resultados_finais:
            self.logger.warning("Nenhum resultado para salvar")
            return
        
        try:
            # Criar DataFrame com resultados
            df_resultados = pd.DataFrame(self.resultados_finais)
            
            # Salvar arquivo principal
            df_resultados.to_csv(config.ARQUIVO_RESULTADO, index=False)
            self.logger.info(f"Resultados salvos em: {config.ARQUIVO_RESULTADO}")
            
            # Salvar resumo estatístico
            resumo_file = os.path.join(config.SAIDA_DIR, 'resumo_estatistico.csv')
            resumo_stats = df_resultados.groupby('ponto_id').agg({
                'temperatura_condutor_p90': ['mean', 'std', 'min', 'max'],
                'risco_termico': ['mean', 'std', 'min', 'max'],
                'ampacidade_calculada': ['mean', 'std', 'min', 'max']
            }).round(3)
            
            resumo_stats.to_csv(resumo_file)
            
            # Salvar metadata
            metadata_file = os.path.join(config.SAIDA_DIR, 'metadata_analise.txt')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(f"Análise de Risco Térmico - Metadata\n")
                f.write(f"Data/Hora: {datetime.now()}\n")
                f.write(f"Resultados válidos: {len(self.resultados_finais)}\n")
                f.write(f"Pontos da linha: {len(self.pontos_linha)}\n")
                f.write(f"Horas processadas: {len(self.resultados_krigagem)}\n")
                f.write(f"Estações meteorológicas: {len(self.dados_estacoes)}\n")
                f.write(f"Corrente de operação: {config.CORRENTE_PADRAO} A\n")
                f.write(f"Temperatura máxima projeto: {config.TEMPERATURA_MAX_PROJETO} °C\n")
                f.write(f"Iterações Monte Carlo: {config.NUM_ITERACOES_MC}\n")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar resultados: {e}")
            raise

    def _gerar_resumo_final(self):
        """Gera resumo final da análise."""
        if not self.resultados_finais:
            return
        
        df = pd.DataFrame(self.resultados_finais)
        
        self.logger.info("\n" + "="*60)
        self.logger.info("RESUMO FINAL DA ANÁLISE")
        self.logger.info("="*60)
        
        # Estatísticas gerais
        self.logger.info(f"Total de resultados: {len(df)}")
        self.logger.info(f"Pontos únicos: {df['ponto_id'].nunique()}")
        self.logger.info(f"Horas únicas: {df['hora'].nunique()}")
        
        # Temperaturas
        temp_p90_media = df['temperatura_condutor_p90'].mean()
        temp_p90_max = df['temperatura_condutor_p90'].max()
        self.logger.info(f"Temperatura P90 média: {temp_p90_media:.2f}°C")
        self.logger.info(f"Temperatura P90 máxima: {temp_p90_max:.2f}°C")
        
        # Risco térmico
        risco_medio = df['risco_termico'].mean()
        risco_max = df['risco_termico'].max()
        pontos_risco_alto = (df['risco_termico'] > 0.1).sum()
        self.logger.info(f"Risco térmico médio: {risco_medio:.4f}")
        self.logger.info(f"Risco térmico máximo: {risco_max:.4f}")
        self.logger.info(f"Pontos com risco > 10%: {pontos_risco_alto}")
        
        # Ampacidade
        ampacidade_media = df['ampacidade_calculada'].mean()
        ampacidade_min = df['ampacidade_calculada'].min()
        self.logger.info(f"Ampacidade média: {ampacidade_media:.0f} A")
        self.logger.info(f"Ampacidade mínima: {ampacidade_min:.0f} A")
        
        self.logger.info("="*60)

def main():
    """Função principal."""
    # Configurar logging
    log_file = configurar_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Iniciando Analisador de Risco Térmico")
    logger.info(f"Log sendo salvo em: {log_file}")
    
    try:
        # Criar diretórios necessários
        config.criar_diretorios()
        
        # Inicializar e executar analisador
        analisador = AnalisadorRiscoTermico()
        sucesso = analisador.executar_analise_completa()
        
        if sucesso:
            logger.info("Análise concluída com sucesso!")
            sys.exit(0)
        else:
            logger.error("Análise falhou!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Execução interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erro não tratado: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()