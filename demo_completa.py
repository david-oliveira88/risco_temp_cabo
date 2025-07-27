#!/usr/bin/env python3
"""
Demonstração completa do sistema de análise de risco térmico.
Este script mostra como usar todas as funcionalidades implementadas.
"""
import sys
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Adicionar o diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('demo_completa.log')
    ]
)
logger = logging.getLogger(__name__)

def demonstrar_validacao_dados():
    """Demonstra o uso do validador de dados."""
    logger.info("=== DEMONSTRAÇÃO: VALIDAÇÃO DE DADOS ===")
    
    from validators import DataValidator
    
    validator = DataValidator()
    
    # 1. Validação de parâmetros do condutor
    logger.info("1. Validando parâmetros do condutor...")
    
    parametros_teste = {
        'diametro': 0.02814,
        'resistencia_ac_25': 7.28e-5,
        'resistencia_ac_75': 9.09e-5,
        'emissividade': 0.8,
        'absortividade': 0.8
    }
    
    eh_valido, erros = validator.validar_parametros_condutor(parametros_teste)
    if eh_valido:
        logger.info("   ✓ Parâmetros do condutor válidos")
    else:
        logger.error(f"   ✗ Erros encontrados: {erros}")
    
    # 2. Validação de dados meteorológicos
    logger.info("2. Validando dados meteorológicos simulados...")
    
    # Criar dados simulados
    datas = pd.date_range('2024-01-01', periods=1000, freq='H')
    dados_simulados = pd.DataFrame({
        'temperatura_ar': np.random.normal(25, 8, 1000),
        'radiacao_global': np.abs(np.random.normal(400, 200, 1000)),
        'vento_velocidade': np.abs(np.random.normal(3, 2, 1000)),
        'vento_direcao': np.random.uniform(0, 360, 1000),
        'latitude': [-15.5] * 1000,
        'longitude': [-47.8] * 1000
    }, index=datas)
    
    # Adicionar dados problemáticos intencionalmente
    dados_simulados.loc[dados_simulados.index[10:15], 'temperatura_ar'] = np.nan  # Dados faltantes
    dados_simulados.loc[dados_simulados.index[20], 'temperatura_ar'] = 150  # Outlier extremo
    dados_simulados.loc[dados_simulados.index[30], 'radiacao_global'] = -500  # Valor impossível
    
    dados_limpos, relatorio = validator.validar_dados_meteorologicos(dados_simulados, "EstacaoDemo")
    
    logger.info(f"   ✓ Dados originais: {relatorio['registros_originais']}")
    logger.info(f"   ✓ Dados válidos: {relatorio['registros_validos']}")
    logger.info(f"   ✓ Taxa de aproveitamento: {relatorio['taxa_aproveitamento']:.1%}")
    
    # 3. Relatório de qualidade
    relatorio_qualidade = validator.gerar_relatorio_qualidade([relatorio])
    logger.info(f"   ✓ Qualidade geral: {relatorio_qualidade['resumo_geral']['qualidade_geral']}")
    
    return dados_limpos, relatorio_qualidade

def demonstrar_modelo_termico():
    """Demonstra o uso do modelo térmico CIGRE."""
    logger.info("=== DEMONSTRAÇÃO: MODELO TÉRMICO CIGRE ===")
    
    from thermal_model import CigreModeloTermico
    
    # Parâmetros do condutor ACSR 795 Drake
    parametros_condutor = {
        'diametro': 0.02814,
        'resistencia_ac_25': 7.28e-5,
        'resistencia_ac_75': 9.09e-5,
        'emissividade': 0.8,
        'absortividade': 0.8
    }
    
    modelo = CigreModeloTermico(parametros_condutor)
    logger.info("✓ Modelo térmico CIGRE inicializado")
    
    # Cenários de teste
    cenarios = [
        {'nome': 'Condições Normais', 'corrente': 400, 'temp_ar': 25, 'vento': 2.0, 'radiacao': 800},
        {'nome': 'Condições Críticas', 'corrente': 600, 'temp_ar': 40, 'vento': 0.5, 'radiacao': 1000},
        {'nome': 'Condições Favoráveis', 'corrente': 300, 'temp_ar': 15, 'vento': 5.0, 'radiacao': 400},
    ]
    
    for cenario in cenarios:
        logger.info(f"Testando: {cenario['nome']}")
        
        # Calcular temperatura do condutor
        temp_condutor = modelo.resolver_temperatura_condutor(
            estimativa_inicial=cenario['temp_ar'] + 20,
            corrente=cenario['corrente'],
            radiacao_solar=cenario['radiacao'],
            azimute_linha=90,
            velocidade_vento=cenario['vento'],
            angulo_vento=90,
            temperatura_ar=cenario['temp_ar']
        )
        
        # Calcular ampacidade
        ampacidade = modelo.calcular_ampacidade(
            temperatura_maxima=75,
            radiacao_solar=cenario['radiacao'],
            azimute_linha=90,
            velocidade_vento=cenario['vento'],
            angulo_vento=90,
            temperatura_ar=cenario['temp_ar']
        )
        
        logger.info(f"   Corrente: {cenario['corrente']} A → Temperatura: {temp_condutor:.1f}°C")
        logger.info(f"   Ampacidade máxima: {ampacidade:.0f} A")
        
        # Verificar se há risco
        if temp_condutor > 75:
            logger.warning(f"   ⚠️  ATENÇÃO: Temperatura excede limite de 75°C!")
        
        logger.info("")
    
    return modelo

def demonstrar_monte_carlo():
    """Demonstra a simulação Monte Carlo."""
    logger.info("=== DEMONSTRAÇÃO: SIMULAÇÃO MONTE CARLO ===")
    
    from thermal_model import CigreModeloTermico
    from simulation import MonteCarloSimulator
    
    # Inicializar modelo
    parametros_condutor = {
        'diametro': 0.02814,
        'resistencia_ac_25': 7.28e-5,
        'resistencia_ac_75': 9.09e-5,
        'emissividade': 0.8,
        'absortividade': 0.8
    }
    
    modelo = CigreModeloTermico(parametros_condutor)
    simulador = MonteCarloSimulator(modelo)
    
    # Condições ambientais com incertezas
    medias_ambientais = {
        'temperatura_ar': 35.0,  # °C
        'radiacao_global': 900.0,  # W/m²
        'vento_u': 1.5,  # m/s
        'vento_v': 1.0   # m/s
    }
    
    desvios_ambientais = {
        'temperatura_ar': 3.0,   # ±3°C
        'radiacao_global': 100.0,  # ±100 W/m²
        'vento_u': 0.8,   # ±0.8 m/s
        'vento_v': 0.8    # ±0.8 m/s
    }
    
    logger.info("Executando simulação Monte Carlo (10.000 iterações)...")
    
    resultado = simulador.executar_simulacao(
        medias_ambientais=medias_ambientais,
        desvios_ambientais=desvios_ambientais,
        azimute_linha=90,
        corrente=500,  # A
        num_iteracoes=10000
    )
    
    # Mostrar resultados
    stats = resultado['estatisticas']
    logger.info(f"✓ Simulação concluída:")
    logger.info(f"   Iterações válidas: {resultado['iteracoes_validas']}")
    logger.info(f"   Taxa de sucesso: {resultado['taxa_sucesso']:.1%}")
    logger.info(f"   Temperatura média: {stats['media']:.1f}°C")
    logger.info(f"   Temperatura P90: {stats['percentil_90']:.1f}°C")
    logger.info(f"   Temperatura P95: {stats['percentil_95']:.1f}°C")
    logger.info(f"   Desvio padrão: {stats['desvio_padrao']:.1f}°C")
    
    return resultado

def demonstrar_analise_risco():
    """Demonstra a análise de risco térmico."""
    logger.info("=== DEMONSTRAÇÃO: ANÁLISE DE RISCO ===")
    
    from risk_analysis import RiskAnalyzer
    
    analyzer = RiskAnalyzer()
    
    # Simular distribuição de temperaturas
    temperaturas_simuladas = np.random.normal(68, 8, 10000)  # Distribuição normal
    temperatura_limite = 75  # °C
    
    # Calcular métricas de risco
    temp_p90 = analyzer.calcular_temperatura_confianca(temperaturas_simuladas, 90)
    risco_termico = analyzer.calcular_risco_termico(temperaturas_simuladas, temperatura_limite)
    classificacao = analyzer.classificar_risco_nbr_5422(risco_termico)
    
    logger.info(f"✓ Análise de risco concluída:")
    logger.info(f"   Temperatura P90: {temp_p90:.1f}°C")
    logger.info(f"   Probabilidade de excedência: {risco_termico:.4f} ({risco_termico*100:.2f}%)")
    logger.info(f"   Classificação: {classificacao['categoria'].upper()}")
    logger.info(f"   Descrição: {classificacao['descricao']}")
    logger.info(f"   Ação recomendada: {classificacao['acao_recomendada']}")
    
    # Testes estatísticos
    teste_normalidade = analyzer.testar_normalidade(temperaturas_simuladas)
    logger.info(f"   Teste de normalidade: {teste_normalidade['interpretacao']} (p={teste_normalidade['p_valor']:.4f})")
    
    # Intervalo de confiança
    ic = analyzer.calcular_intervalo_confianca(temperaturas_simuladas, 0.95)
    logger.info(f"   IC 95%: [{ic['limite_inferior']:.1f}, {ic['limite_superior']:.1f}]°C")
    
    # Estimativa de vida útil
    vida_util = analyzer.calcular_vida_util_estimada(temperaturas_simuladas, 75)
    logger.info(f"   Fator de redução de vida útil: {vida_util['fator_reducao']:.2f}")
    
    # Gerar relatório completo
    relatorio = analyzer.gerar_relatorio_risco(
        temperaturas_simuladas, 
        temperatura_limite, 
        "Demonstração"
    )
    
    logger.info(f"✓ Relatório completo gerado com {len(relatorio)} seções")
    
    return relatorio

def demonstrar_visualizacao():
    """Demonstra a criação de visualizações."""
    logger.info("=== DEMONSTRAÇÃO: VISUALIZAÇÃO ===")
    
    try:
        from visualization import VisualizadorResultados
        
        # Criar dados simulados para demonstração
        n_pontos = 500
        resultados_demo = pd.DataFrame({
            'hora': pd.date_range('2024-01-01', periods=n_pontos, freq='H'),
            'temperatura_condutor_p90': np.random.normal(65, 10, n_pontos),
            'temperatura_condutor_media': np.random.normal(60, 8, n_pontos),
            'temperatura_ar_media': np.random.normal(28, 6, n_pontos),
            'risco_termico': np.random.uniform(0, 0.08, n_pontos),
            'ampacidade_calculada': np.random.normal(750, 100, n_pontos),
            'corrente_operacao': [500] * n_pontos,
            'progressiva': np.tile(np.linspace(0, 20000, 50), 10),  # 20 km de linha
            'ponto_id': list(range(50)) * 10
        })
        
        # Adicionar algumas variações realistas
        # Efeito diurno na radiação
        horas_dia = resultados_demo['hora'].dt.hour
        fator_radiacao = np.sin(np.pi * (horas_dia - 6) / 12).clip(0)
        resultados_demo['temperatura_condutor_p90'] += 5 * fator_radiacao
        
        # Criar visualizador
        visualizador = VisualizadorResultados("saida/graficos_demo")
        logger.info("✓ Visualizador inicializado")
        
        # 1. Distribuição de temperaturas
        logger.info("Criando gráfico de distribuição de temperaturas...")
        fig1 = visualizador.plotar_distribuicao_temperaturas(
            resultados_demo['temperatura_condutor_p90'].values,
            titulo="Distribuição de Temperaturas - Demonstração"
        )
        
        # 2. Série temporal
        logger.info("Criando gráfico de série temporal...")
        fig2 = visualizador.plotar_serie_temporal_risco(resultados_demo)
        
        # 3. Mapa de calor espacial
        logger.info("Criando mapa de calor espacial...")
        fig3 = visualizador.plotar_mapa_calor_espacial(
            resultados_demo, 
            'temperatura_condutor_p90'
        )
        
        # 4. Dashboard completo
        logger.info("Criando dashboard resumo...")
        relatorio_qualidade_demo = {
            'resumo_geral': {
                'num_estacoes': 5,
                'taxa_aproveitamento': 0.92,
                'qualidade_geral': 'Boa',
                'registros_originais': 5000,
                'registros_validos': 4600
            }
        }
        
        fig4 = visualizador.plotar_dashboard_resumo(
            resultados_demo, 
            relatorio_qualidade_demo
        )
        
        # 5. Análise de sensibilidade
        logger.info("Criando gráfico de análise de sensibilidade...")
        sensibilidades_demo = {
            'temperatura_ar': 0.85,
            'radiacao_global': 0.42,
            'vento_u': -0.38,
            'vento_v': -0.25
        }
        
        fig5 = visualizador.plotar_analise_sensibilidade(sensibilidades_demo)
        
        logger.info("✓ Todas as visualizações foram criadas e salvas em 'saida/graficos_demo'")
        
    except ImportError as e:
        logger.warning(f"Módulo de visualização não disponível: {e}")
        logger.info("Para usar visualizações, instale: pip install matplotlib seaborn")

def main():
    """Função principal da demonstração."""
    logger.info("🚀 INICIANDO DEMONSTRAÇÃO COMPLETA DO SISTEMA")
    logger.info("=" * 80)
    
    try:
        # 1. Validação de dados
        dados_limpos, relatorio_qualidade = demonstrar_validacao_dados()
        logger.info("")
        
        # 2. Modelo térmico
        modelo = demonstrar_modelo_termico()
        logger.info("")
        
        # 3. Monte Carlo
        resultado_mc = demonstrar_monte_carlo()
        logger.info("")
        
        # 4. Análise de risco
        relatorio_risco = demonstrar_analise_risco()
        logger.info("")
        
        # 5. Visualização
        demonstrar_visualizacao()
        logger.info("")
        
        # Resumo final
        logger.info("=" * 80)
        logger.info("🎉 DEMONSTRAÇÃO COMPLETA CONCLUÍDA COM SUCESSO!")
        logger.info("")
        logger.info("RESUMO DOS MÓDULOS DEMONSTRADOS:")
        logger.info("✓ Validação robusta de dados meteorológicos")
        logger.info("✓ Modelo térmico CIGRE 601 completo")
        logger.info("✓ Simulação Monte Carlo para propagação de incertezas")
        logger.info("✓ Análise de risco conforme NBR 5422")
        logger.info("✓ Visualizações avançadas dos resultados")
        logger.info("")
        logger.info("ARQUIVOS GERADOS:")
        logger.info("• demo_completa.log - Log completo da demonstração")
        logger.info("• saida/graficos_demo/ - Gráficos e visualizações")
        logger.info("")
        logger.info("O sistema está pronto para análise de dados reais!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro durante a demonstração: {e}")
        logger.error("Verifique se todas as dependências estão instaladas")
        return False

if __name__ == "__main__":
    import sys
    sucesso = main()
    sys.exit(0 if sucesso else 1)