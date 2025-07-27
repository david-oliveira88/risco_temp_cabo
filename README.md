# Analisador de Risco Térmico de Cabos Elétricos

Sistema completo para análise de risco térmico de condutores de linhas de transmissão, implementando metodologias conforme CIGRE TB-601 e ABNT NBR 5422.

## 🎯 Características Principais

- **Modelo Térmico CIGRE 601**: Implementação completa do modelo de balanço térmico
- **Simulação Monte Carlo**: Propagação de incertezas meteorológicas
- **Krigagem Ordinária**: Interpolação espacial de dados meteorológicos
- **Análise de Risco NBR 5422**: Classificação de risco térmico
- **Validação Robusta**: Sistema avançado de validação de dados
- **Visualizações Avançadas**: Gráficos e dashboards interativos

## 📁 Estrutura do Projeto

```
analisador_risco_termico/
├── main.py                 # Orquestrador principal
├── config.py               # Configurações e constantes
├── data_loader.py          # Carregamento e validação de dados
├── validators.py           # Validação robusta de dados
├── geoprocessing.py        # Processamento geoespacial e krigagem
├── thermal_model.py        # Modelo térmico CIGRE 601
├── simulation.py           # Simulação Monte Carlo
├── risk_analysis.py        # Análise de risco NBR 5422
├── visualization.py        # Visualizações e gráficos
├── requirements.txt        # Dependências Python
├── test_basic.py          # Testes unitários
├── demo_completa.py       # Demonstração completa
├── /dados/                # Dados meteorológicos (CSV)
├── /entrada/              # Arquivos de configuração
│   ├── parametros_cabo.json
│   └── trassado_linha.xlsx
└── /saida/                # Resultados e relatórios
    ├── resultado_horario.csv
    └── graficos/
```

## 🔧 Instalação

### Pré-requisitos
- Python 3.9 ou superior
- Sistema operacional: Windows, Linux ou macOS

### Dependências
```bash
# Instalar dependências básicas
pip install pandas numpy openpyxl scipy pyproj pykrige matplotlib scikit-learn

# Ou instalar todas as dependências
pip install -r requirements.txt
```

### Dependências Detalhadas
- **pandas**: Manipulação de dados meteorológicos
- **numpy**: Computação numérica
- **scipy**: Solvers para equações térmicas
- **pyproj**: Transformações de coordenadas geográficas
- **pykrige**: Interpolação por krigagem ordinária
- **matplotlib**: Visualização de resultados
- **scikit-learn**: Validação e análise de dados
- **openpyxl**: Leitura de arquivos Excel

## 📊 Dados de Entrada

### 1. Parâmetros do Condutor (`entrada/parametros_cabo.json`)
```json
{
  "nome_condutor": "ACSR 795 Drake",
  "diametro": 0.02814,
  "resistencia_ac_25": 7.28e-5,
  "resistencia_ac_75": 9.09e-5,
  "emissividade": 0.8,
  "absortividade": 0.8,
  "temperatura_maxima_operacao": 75
}
```

### 2. Traçado da Linha (`entrada/trassado_linha.xlsx`)
Colunas obrigatórias:
- `Progressiva`: Posição em metros
- `latitude`: Latitude em graus
- `longitude`: Longitude em graus  
- `azimute`: Azimute em graus

### 3. Dados Meteorológicos (`dados/*.csv`)
Formato padrão INMET com colunas:
- `TEMPERATURA_DO_AR__BULBO_SECO_HORARIAC`
- `RADIACAO_GLOBALKjm`
- `VENTO_VELOCIDADE_HORARIAms`
- `VENTO_DIRECAO_HORARIA_gr__gr`

## 🚀 Uso Rápido

### Execução Completa
```bash
python main.py
```

### Demonstração das Funcionalidades
```bash
python demo_completa.py
```

### Testes do Sistema
```bash
python test_basic.py
```

## 📈 Metodologia

### 1. Modelo Térmico CIGRE 601
Implementa o balanço térmico completo:
- **Aquecimento Joule**: P_J = I²R(T_c)
- **Aquecimento Solar**: P_S = α·D·q_s·sin(β)
- **Resfriamento Convectivo**: P_c = π·D·h_c·(T_c - T_a)
- **Resfriamento Radiativo**: P_r = π·D·ε·σ·(T_c⁴ - T_a⁴)

### 2. Propagação de Incertezas
- **Krigagem Ordinária**: Interpolação espacial com estimativa de variância
- **Monte Carlo**: 10.000 iterações para cada ponto/hora
- **Distribuições**: Normal para variáveis meteorológicas

### 3. Análise de Risco NBR 5422
Classificação baseada na probabilidade de excedência:
- **Baixo**: < 1%
- **Moderado**: 1-5%
- **Alto**: 5-10%
- **Crítico**: > 10%

## 📊 Resultados

### Arquivo Principal (`saida/resultado_horario.csv`)
Contém para cada ponto e hora:
- Temperatura do condutor (média, P90, P95)
- Risco térmico
- Ampacidade calculada
- Condições ambientais interpoladas

### Visualizações Geradas
- Distribuição de temperaturas
- Séries temporais de risco
- Mapas de calor espaciais
- Dashboard resumo
- Análise de sensibilidade

### Relatórios de Qualidade
- Taxa de aproveitamento dos dados
- Problemas identificados por estação
- Completude por variável
- Recomendações de melhoria

## 🔍 Validação de Dados

### Sistema Robusto de Validação
- **Limites Físicos**: Verificação de valores possíveis
- **Detecção de Outliers**: Métodos IQR + Z-score
- **Consistência Temporal**: Identificação de gaps e duplicatas
- **Completude**: Análise de dados faltantes

### Relatório de Qualidade
```python
from data_loader import DataLoader

loader = DataLoader()
loader.carregar_todos_dados()
relatorio = loader.obter_relatorio_qualidade_dados()
print(f"Qualidade geral: {relatorio['resumo_geral']['qualidade_geral']}")
```

## 📐 Exemplo de Uso Programático

```python
# Inicializar sistema
from main import AnalisadorRiscoTermico

analisador = AnalisadorRiscoTermico()
sucesso = analisador.executar_analise_completa()

# Acessar resultados
if sucesso:
    resultados = analisador.resultados_finais
    print(f"Análise concluída: {len(resultados)} resultados")
```

### Uso de Módulos Individuais

```python
# Modelo térmico
from thermal_model import CigreModeloTermico

modelo = CigreModeloTermico(parametros_cabo)
temperatura = modelo.resolver_temperatura_condutor(
    estimativa_inicial=50,
    corrente=500,
    radiacao_solar=800,
    azimute_linha=90,
    velocidade_vento=2.0,
    angulo_vento=90,
    temperatura_ar=30
)

# Simulação Monte Carlo
from simulation import MonteCarloSimulator

simulador = MonteCarloSimulator(modelo)
resultado = simulador.executar_simulacao(
    medias_ambientais=medias,
    desvios_ambientais=desvios,
    azimute_linha=90,
    corrente=500
)

# Análise de risco
from risk_analysis import RiskAnalyzer

analyzer = RiskAnalyzer()
risco = analyzer.calcular_risco_termico(resultado['temperaturas'], 75)
classificacao = analyzer.classificar_risco_nbr_5422(risco)
```

## ⚙️ Configuração Avançada

### Parâmetros Personalizáveis (`config.py`)
```python
# Simulação Monte Carlo
NUM_ITERACOES_MC = 10000
PERCENTIL_CONFIANCA = 90

# Krigagem
MODELO_VARIOGRAMA = 'linear'
DISTANCIA_DISCRETIZACAO = 1000  # metros

# Limites de validação
TEMP_AR_MIN = -50  # °C
TEMP_AR_MAX = 60   # °C
```

### Sistemas de Coordenadas
- **Origem**: SIRGAS 2000 (EPSG:4674)
- **Destino**: Brazil Polyconic (EPSG:5880)

## 🧪 Testes e Validação

### Suite de Testes
```bash
python test_basic.py
```

Testa:
- ✅ Importação de módulos
- ✅ Validação de configurações
- ✅ Modelo térmico CIGRE
- ✅ Simulação Monte Carlo
- ✅ Análise de risco
- ✅ Validação de dados
- ✅ Visualizações

### Dados de Teste
O sistema inclui dados simulados para testes, permitindo validação sem dados reais.

## 📝 Limitações e Considerações

### Limitações Conhecidas
- Modelo solar simplificado (não considera topografia)
- Krigagem assume estacionariedade espacial
- Requer mínimo 2 estações meteorológicas
- Dados horários obrigatórios

### Recomendações de Uso
- Validar dados meteorológicos antes da análise
- Verificar qualidade da interpolação espacial
- Analisar sensibilidade dos resultados
- Documentar premissas assumidas

## 🔬 Base Científica

### Referencias Técnicas
- **CIGRE TB-601**: "Guide for thermal rating calculations of overhead lines"
- **ABNT NBR 5422**: "Projeto de linhas aéreas de transmissão de energia elétrica"
- **IEEE 738**: "Standard for Calculating the Current-Temperature Relationship of Bare Overhead Conductors"

### Validação Científica
O modelo foi validado com:
- Casos de teste do CIGRE
- Dados experimentais de campo
- Comparação com outros softwares comerciais

## 🤝 Contribuição

### Como Contribuir
1. Fork do repositório
2. Criar branch para feature (`git checkout -b feature/nova-feature`)
3. Commit das mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para branch (`git push origin feature/nova-feature`)
5. Abrir Pull Request

### Áreas de Melhoria
- Modelo solar astronômico completo
- Interface gráfica (GUI)
- Integração com bancos de dados
- Análise de múltiplos cabos
- Otimização de corrente operacional

## 📞 Suporte

### Documentação
- Código autodocumentado com docstrings
- Exemplos de uso em `demo_completa.py`
- Logs detalhados de execução

### Troubleshooting
- Verificar instalação das dependências
- Validar formato dos dados de entrada
- Consultar logs de erro
- Executar testes diagnósticos

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.

## 🏆 Reconhecimentos

Desenvolvido com base nas melhores práticas da engenharia de sistemas de potência e metodologias internacionalmente reconhecidas para análise térmica de condutores.

---

**Nota**: Este sistema foi projetado para uso profissional em engenharia de sistemas elétricos. Sempre valide os resultados com especialistas antes de tomar decisões operacionais críticas.