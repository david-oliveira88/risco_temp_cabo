#!/usr/bin/env python3
"""
Teste abrangente para validar o funcionamento completo do sistema.
"""
import sys
import os
import numpy as np
import pandas as pd
import logging
import tempfile
import json
from datetime import datetime, timedelta

# Adicionar o diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configurar logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def teste_importacoes():
    """Testa se todos os módulos podem ser importados."""
    logger.info("=== Teste de Importações ===")
    
    try:
        import config
        logger.info("✓ config importado com sucesso")
    except Exception as e:
        logger.error(f"✗ Erro ao importar config: {e}")
        return False
    
    try:
        from data_loader import DataLoader
        logger.info("✓ DataLoader importado com sucesso")
    except Exception as e:
        logger.error(f"✗ Erro ao importar DataLoader: {e}")
        return False
    
    try:
        from thermal_model import CigreModeloTermico
        logger.info("✓ CigreModeloTermico importado com sucesso")
    except Exception as e:
        logger.error(f"✗ Erro ao importar CigreModeloTermico: {e}")
        return False
    
    try:
        from geoprocessing import GeoProcessor
        logger.info("✓ GeoProcessor importado com sucesso")
    except Exception as e:
        logger.error(f"✗ Erro ao importar GeoProcessor: {e}")
        return False
    
    try:
        from simulation import MonteCarloSimulator
        logger.info("✓ MonteCarloSimulator importado com sucesso")
    except Exception as e:
        logger.error(f"✗ Erro ao importar MonteCarloSimulator: {e}")
        return False
    
    try:
        from risk_analysis import RiskAnalyzer
        logger.info("✓ RiskAnalyzer importado com sucesso")
    except Exception as e:
        logger.error(f"✗ Erro ao importar RiskAnalyzer: {e}")
        return False
    
    try:
        from validators import DataValidator
        logger.info("✓ DataValidator importado com sucesso")
    except Exception as e:
        logger.error(f"✗ Erro ao importar DataValidator: {e}")
        return False
    
    try:
        from visualization import VisualizadorResultados
        logger.info("✓ VisualizadorResultados importado com sucesso")
    except Exception as e:
        logger.error(f"✗ Erro ao importar VisualizadorResultados: {e}")
        return False
    
    return True

def teste_configuracao():
    """Testa a configuração básica."""
    logger.info("=== Teste de Configuração ===")
    
    import config
    
    # Testar se as constantes existem
    constantes_obrigatorias = [
        'STEFAN_BOLTZMANN', 'GRAVIDADE', 'NUM_ITERACOES_MC',
        'TEMPERATURA_MAX_PROJETO', 'VARIAVEIS_AMBIENTAIS'
    ]
    
    for constante in constantes_obrigatorias:
        if hasattr(config, constante):
            logger.info(f"✓ Constante {constante} definida: {getattr(config, constante)}")
        else:
            logger.error(f"✗ Constante {constante} não encontrada")
            return False
    
    return True

def teste_modelo_termico():
    """Testa o modelo térmico básico."""
    logger.info("=== Teste do Modelo Térmico ===")
    
    try:
        from thermal_model import CigreModeloTermico
        
        # Parâmetros de teste do condutor
        parametros_teste = {
            'diametro': 0.02814,
            'resistencia_ac_25': 7.28e-5,
            'resistencia_ac_75': 9.09e-5,
            'emissividade': 0.8,
            'absortividade': 0.8
        }
        
        modelo = CigreModeloTermico(parametros_teste)
        logger.info("✓ Modelo térmico inicializado")
        
        # Teste de cálculo de resistência
        r_25 = modelo.calcular_resistencia_ac(25)
        r_75 = modelo.calcular_resistencia_ac(75)
        logger.info(f"✓ Resistência a 25°C: {r_25:.6f} Ω/m")
        logger.info(f"✓ Resistência a 75°C: {r_75:.6f} Ω/m")
        
        # Teste de aquecimento Joule
        p_joule = modelo.calcular_aquecimento_joule(500, 50)
        logger.info(f"✓ Aquecimento Joule (500A, 50°C): {p_joule:.2f} W/m")
        
        # Teste de aquecimento solar
        p_solar = modelo.calcular_aquecimento_solar(800, 90)
        logger.info(f"✓ Aquecimento solar (800 W/m², 90°): {p_solar:.2f} W/m")
        
        # Teste de resfriamento convectivo
        p_conv = modelo.calcular_resfriamento_convectivo(2.0, 90, 25, 50)
        logger.info(f"✓ Resfriamento convectivo: {p_conv:.2f} W/m")
        
        # Teste de resfriamento radiativo
        p_rad = modelo.calcular_resfriamento_radiativo(25, 50)
        logger.info(f"✓ Resfriamento radiativo: {p_rad:.2f} W/m")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro no teste do modelo térmico: {e}")
        return False

def teste_monte_carlo():
    """Testa o simulador Monte Carlo básico."""
    logger.info("=== Teste do Monte Carlo ===")
    
    try:
        from thermal_model import CigreModeloTermico
        from simulation import MonteCarloSimulator
        
        # Inicializar modelo
        parametros_teste = {
            'diametro': 0.02814,
            'resistencia_ac_25': 7.28e-5,
            'resistencia_ac_75': 9.09e-5,
            'emissividade': 0.8,
            'absortividade': 0.8
        }
        
        modelo = CigreModeloTermico(parametros_teste)
        simulador = MonteCarloSimulator(modelo)
        logger.info("✓ Simulador Monte Carlo inicializado")
        
        # Dados ambientais de teste
        medias_ambientais = {
            'temperatura_ar': 30.0,
            'radiacao_global': 400.0,
            'vento_u': 1.0,
            'vento_v': 1.0
        }
        
        desvios_ambientais = {
            'temperatura_ar': 2.0,
            'radiacao_global': 50.0,
            'vento_u': 0.5,
            'vento_v': 0.5
        }
        
        # Executar simulação rápida
        resultado = simulador.executar_simulacao(
            medias_ambientais=medias_ambientais,
            desvios_ambientais=desvios_ambientais,
            azimute_linha=90,
            corrente=400,
            num_iteracoes=100  # Teste rápido
        )
        
        logger.info(f"✓ Simulação concluída: {resultado['iteracoes_validas']} iterações válidas")
        logger.info(f"✓ Temperatura média: {resultado['estatisticas']['media']:.2f}°C")
        logger.info(f"✓ Temperatura P90: {resultado['estatisticas']['percentil_90']:.2f}°C")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro no teste Monte Carlo: {e}")
        return False

def teste_analise_risco():
    """Testa o analisador de risco."""
    logger.info("=== Teste de Análise de Risco ===")
    
    try:
        from risk_analysis import RiskAnalyzer
        
        analyzer = RiskAnalyzer()
        logger.info("✓ Analisador de risco inicializado")
        
        # Dados de teste
        temperaturas_teste = np.random.normal(60, 5, 1000)  # Distribuição normal
        temperatura_limite = 75
        
        # Teste de temperatura de confiança
        temp_p90 = analyzer.calcular_temperatura_confianca(temperaturas_teste, 90)
        logger.info(f"✓ Temperatura P90: {temp_p90:.2f}°C")
        
        # Teste de risco térmico
        risco = analyzer.calcular_risco_termico(temperaturas_teste, temperatura_limite)
        logger.info(f"✓ Risco térmico: {risco:.4f} ({risco*100:.2f}%)")
        
        # Teste de classificação NBR 5422
        classificacao = analyzer.classificar_risco_nbr_5422(risco)
        logger.info(f"✓ Classificação: {classificacao['categoria']} - {classificacao['descricao']}")
        
        # Teste de relatório
        relatorio = analyzer.gerar_relatorio_risco(temperaturas_teste, temperatura_limite, "Teste")
        logger.info("✓ Relatório gerado com sucesso")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro no teste de análise de risco: {e}")
        return False

def teste_validador_dados():
    """Testa o validador de dados."""
    logger.info("=== Teste do Validador de Dados ===")
    
    try:
        from validators import DataValidator
        
        validator = DataValidator()
        logger.info("✓ Validador inicializado")
        
        # Teste 1: Validar parâmetros do condutor
        parametros_validos = {
            'diametro': 0.02814,
            'resistencia_ac_25': 7.28e-5,
            'resistencia_ac_75': 9.09e-5,
            'emissividade': 0.8,
            'absortividade': 0.8
        }
        
        eh_valido, erros = validator.validar_parametros_condutor(parametros_validos)
        if eh_valido:
            logger.info("✓ Validação de parâmetros válidos: PASSOU")
        else:
            logger.error(f"✗ Validação de parâmetros válidos falhou: {erros}")
            return False
        
        # Teste 2: Detectar parâmetros inválidos
        parametros_invalidos = {
            'diametro': -0.01,  # Negativo
            'resistencia_ac_25': 0,  # Zero
            'emissividade': 1.5,  # Fora da faixa
        }
        
        eh_valido, erros = validator.validar_parametros_condutor(parametros_invalidos)
        if not eh_valido and len(erros) > 0:
            logger.info(f"✓ Detectou parâmetros inválidos: {len(erros)} erros")
        else:
            logger.error("✗ Falhou em detectar parâmetros inválidos")
            return False
        
        # Teste 3: Validar dados meteorológicos
        # Criar dados de teste
        datas = pd.date_range('2024-01-01', periods=100, freq='H')
        dados_teste = pd.DataFrame({
            'temperatura_ar': np.random.normal(25, 5, 100),
            'radiacao_global': np.random.uniform(0, 1000, 100),
            'vento_velocidade': np.random.uniform(0, 15, 100),
            'vento_direcao': np.random.uniform(0, 360, 100),
            'latitude': [-15.5] * 100,
            'longitude': [-47.8] * 100
        }, index=datas)
        
        # Adicionar alguns outliers intencionais
        dados_teste.loc[dados_teste.index[10], 'temperatura_ar'] = 100  # Outlier
        dados_teste.loc[dados_teste.index[20], 'radiacao_global'] = -100  # Inválido
        
        dados_limpos, relatorio = validator.validar_dados_meteorologicos(dados_teste, "Teste")
        
        if len(dados_limpos) < len(dados_teste):
            logger.info(f"✓ Validação meteorológica: removeu {len(dados_teste) - len(dados_limpos)} registros inválidos")
        else:
            logger.warning("Validação meteorológica não removeu registros inválidos")
        
        logger.info(f"✓ Taxa de aproveitamento: {relatorio['taxa_aproveitamento']:.1%}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro no teste do validador: {e}")
        return False

def teste_data_loader_melhorado():
    """Testa o DataLoader melhorado."""
    logger.info("=== Teste do DataLoader Melhorado ===")
    
    try:
        from data_loader import DataLoader
        
        # Verificar se o DataLoader tem os novos métodos
        loader = DataLoader()
        
        if hasattr(loader, 'validator'):
            logger.info("✓ DataLoader tem validador integrado")
        else:
            logger.error("✗ DataLoader não tem validador integrado")
            return False
        
        if hasattr(loader, 'obter_relatorio_qualidade_dados'):
            logger.info("✓ DataLoader tem método de relatório de qualidade")
        else:
            logger.error("✗ DataLoader não tem método de relatório de qualidade")
            return False
        
        logger.info("✓ DataLoader melhorado funcionando")
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro no teste do DataLoader: {e}")
        return False

def teste_visualizacao():
    """Testa o módulo de visualização."""
    logger.info("=== Teste de Visualização ===")
    
    try:
        from visualization import VisualizadorResultados
        
        # Criar dados de teste
        n_pontos = 100
        dados_teste = pd.DataFrame({
            'hora': pd.date_range('2024-01-01', periods=n_pontos, freq='H'),
            'temperatura_condutor_p90': np.random.normal(60, 10, n_pontos),
            'temperatura_condutor_media': np.random.normal(55, 8, n_pontos),
            'temperatura_ar_media': np.random.normal(30, 5, n_pontos),
            'risco_termico': np.random.uniform(0, 0.1, n_pontos),
            'ampacidade_calculada': np.random.normal(800, 100, n_pontos),
            'corrente_operacao': [500] * n_pontos,
            'progressiva': np.linspace(0, 10000, n_pontos),
            'ponto_id': range(n_pontos)
        })
        
        # Criar diretório temporário
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizador = VisualizadorResultados(temp_dir)
            logger.info("✓ Visualizador inicializado")
            
            # Teste de distribuição de temperaturas
            fig1 = visualizador.plotar_distribuicao_temperaturas(
                dados_teste['temperatura_condutor_p90'].values,
                salvar=False
            )
            logger.info("✓ Gráfico de distribuição criado")
            
            # Teste de série temporal
            fig2 = visualizador.plotar_serie_temporal_risco(dados_teste, salvar=False)
            logger.info("✓ Gráfico de série temporal criado")
            
            # Teste de dashboard
            fig3 = visualizador.plotar_dashboard_resumo(dados_teste, salvar=False)
            logger.info("✓ Dashboard criado")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro no teste de visualização: {e}")
        return False

def executar_todos_testes():
    """Executa todos os testes básicos."""
    logger.info("=== INICIANDO TESTES BÁSICOS DO SISTEMA ===")
    
    testes = [
        ("Importações", teste_importacoes),
        ("Configuração", teste_configuracao),
        ("Modelo Térmico", teste_modelo_termico),
        ("Monte Carlo", teste_monte_carlo),
        ("Análise de Risco", teste_analise_risco),
        ("Validador de Dados", teste_validador_dados),
        ("DataLoader Melhorado", teste_data_loader_melhorado),
        ("Visualização", teste_visualizacao)
    ]
    
    resultados = {}
    
    for nome_teste, funcao_teste in testes:
        try:
            resultado = funcao_teste()
            resultados[nome_teste] = resultado
            if resultado:
                logger.info(f"✓ {nome_teste}: PASSOU")
            else:
                logger.error(f"✗ {nome_teste}: FALHOU")
        except Exception as e:
            logger.error(f"✗ {nome_teste}: ERRO - {e}")
            resultados[nome_teste] = False
    
    # Resumo final
    logger.info("=== RESUMO DOS TESTES ===")
    aprovados = sum(resultados.values())
    total = len(resultados)
    
    for nome, resultado in resultados.items():
        status = "✓ PASSOU" if resultado else "✗ FALHOU"
        logger.info(f"{nome}: {status}")
    
    logger.info(f"Total: {aprovados}/{total} testes aprovados")
    
    if aprovados == total:
        logger.info("🎉 TODOS OS TESTES BÁSICOS APROVADOS!")
        return True
    else:
        logger.error("❌ ALGUNS TESTES FALHARAM")
        return False

if __name__ == "__main__":
    sucesso = executar_todos_testes()
    sys.exit(0 if sucesso else 1)