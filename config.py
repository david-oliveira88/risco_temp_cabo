# Parâmetros de configuração (paths, constantes)
import os

# =============================================================================
# CAMINHOS DOS ARQUIVOS E DIRETÓRIOS
# =============================================================================

# Diretório base do projeto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminhos para dados de entrada
DADOS_DIR = os.path.join(BASE_DIR, 'dados')
ENTRADA_DIR = os.path.join(BASE_DIR, 'entrada')
SAIDA_DIR = os.path.join(BASE_DIR, 'saida')

# Arquivos específicos
ARQUIVO_TRASSADO_LINHA = os.path.join(ENTRADA_DIR, 'trassado_linha.xlsx')
ARQUIVO_PARAMETROS_CABO = os.path.join(ENTRADA_DIR, 'parametros_cabo.json')
ARQUIVO_RESULTADO = os.path.join(SAIDA_DIR, 'resultado_horario.csv')

# =============================================================================
# CONSTANTES FÍSICAS
# =============================================================================

# Constante de Stefan-Boltzmann (W/m²K⁴)
STEFAN_BOLTZMANN = 5.670374419e-8

# Aceleração da gravidade (m/s²)
GRAVIDADE = 9.80665

# Densidade do ar ao nível do mar a 20°C (kg/m³)
DENSIDADE_AR_PADRAO = 1.225

# Viscosidade cinemática do ar a 20°C (m²/s)
VISCOSIDADE_CINEMATICA_AR = 15.06e-6

# Condutividade térmica do ar a 20°C (W/m·K)
CONDUTIVIDADE_TERMICA_AR = 0.0263

# =============================================================================
# PARÂMETROS DE SIMULAÇÃO
# =============================================================================

# Número de iterações para Monte Carlo
NUM_ITERACOES_MC = 10000

# Percentil para cálculo de temperatura de confiança
PERCENTIL_CONFIANCA = 90

# Temperatura máxima de projeto padrão (°C)
TEMPERATURA_MAX_PROJETO = 75

# Distância entre pontos de discretização da linha (metros)
DISTANCIA_DISCRETIZACAO = 1000  # 1 km

# Corrente elétrica padrão (Amperes) - pode ser sobrescrita
CORRENTE_PADRAO = 500

# =============================================================================
# SISTEMAS DE COORDENADAS
# =============================================================================

# Sistema de coordenadas de origem (SIRGAS 2000)
CRS_ORIGEM = 'EPSG:4674'

# Sistema de coordenadas de destino (Brazil Polyconic)
CRS_DESTINO = 'EPSG:5880'

# =============================================================================
# PARÂMETROS DE KRIGAGEM
# =============================================================================

# Modelo de variograma para krigagem
MODELO_VARIOGRAMA = 'linear'

# Variáveis ambientais para interpolação
VARIAVEIS_AMBIENTAIS = ['temperatura_ar', 'radiacao_global', 'vento_u', 'vento_v']

# =============================================================================
# VALIDAÇÃO DE DADOS
# =============================================================================

# Colunas obrigatórias no arquivo de traçado da linha
COLUNAS_TRASSADO_OBRIGATORIAS = ['Progressiva', 'azimute', 'latitude', 'longitude']

# Colunas esperadas nos dados das estações meteorológicas
COLUNAS_ESTACOES_ESPERADAS = [
    'DataMedicao', 'HoraMedicao', 
    'TEMPERATURA_DO_AR__BULBO_SECO_HORARIAC',
    'RADIACAO_GLOBALKjm',
    'VENTO_VELOCIDADE_HORARIAms',
    'VENTO_DIRECAO_HORARIA_gr__gr'
]

# Parâmetros obrigatórios do cabo condutor
PARAMETROS_CABO_OBRIGATORIOS = [
    'diametro', 'resistencia_ac_25', 'resistencia_ac_75', 
    'emissividade', 'absortividade'
]

# =============================================================================
# CONFIGURAÇÕES DE LOGGING
# =============================================================================

# Nível de logging
LOG_LEVEL = 'INFO'

# Formato de log
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# =============================================================================
# LIMITES DE VALIDAÇÃO
# =============================================================================

# Limites de temperatura do ar (°C)
TEMP_AR_MIN = -50
TEMP_AR_MAX = 60

# Limites de radiação solar (kJ/m²)
RADIACAO_MIN = 0
RADIACAO_MAX = 4000

# Limites de velocidade do vento (m/s)
VENTO_VEL_MIN = 0
VENTO_VEL_MAX = 50

# Limites de direção do vento (graus)
VENTO_DIR_MIN = 0
VENTO_DIR_MAX = 360

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def criar_diretorios():
    """Cria os diretórios necessários se não existirem."""
    os.makedirs(DADOS_DIR, exist_ok=True)
    os.makedirs(ENTRADA_DIR, exist_ok=True)
    os.makedirs(SAIDA_DIR, exist_ok=True)

def validar_configuracao():
    """Valida se todos os diretórios e arquivos necessários existem."""
    erros = []
    
    if not os.path.exists(DADOS_DIR):
        erros.append(f"Diretório de dados não encontrado: {DADOS_DIR}")
    
    if not os.path.exists(ENTRADA_DIR):
        erros.append(f"Diretório de entrada não encontrado: {ENTRADA_DIR}")
    
    if not os.path.exists(ARQUIVO_TRASSADO_LINHA):
        erros.append(f"Arquivo de traçado não encontrado: {ARQUIVO_TRASSADO_LINHA}")
    
    if not os.path.exists(ARQUIVO_PARAMETROS_CABO):
        erros.append(f"Arquivo de parâmetros do cabo não encontrado: {ARQUIVO_PARAMETROS_CABO}")
    
    if erros:
        raise FileNotFoundError("\n".join(erros))
    
    return True