Instruções para Agente de IA: Framework de Cálculo de Risco Térmico de Cabos Elétricos
1. Objetivo Principal
Desenvolver uma aplicação em Python, de forma modular e de fácil manutenção, para calcular a ampacidade e o risco térmico horário de um cabo condutor de uma linha de transmissão (LT), usar termos  e variáveis sempre em português. A metodologia deve seguir as premissas técnicas definidas:

Modelo Térmico: CIGRE (Technical Brochure 601).

Risco Térmico: Definição conforme a norma ABNT NBR 5422.

Modelagem Ambiental: Interpolação espacial horária de dados meteorológicos (Temperatura do ar, Radiação Solar, Vento) utilizando Krigagem Ordinária.

Análise de Incerteza: Propagação das incertezas da interpolação através de Simulação de Monte Carlo para determinar a distribuição de probabilidade da temperatura do condutor.

2. Arquitetura do Código
O código deve ser estruturado em módulos distintos, cada um com uma responsabilidade clara, seguindo os princípios de programação orientada a objetos (POO) sempre que aplicável.

Estrutura de Arquivos e Pastas Sugerida:
/analisador_risco_termico/
|
|-- main.py                 # Orquestrador principal da execução
|-- config.py               # Parâmetros de configuração (paths, constantes)
|-- data_loader.py          # Módulo para carregar e validar dados de entrada
|-- geoprocessing.py        # Módulo para conversão de coordenadas e Krigagem
|-- thermal_model.py        # Módulo com a implementação do modelo térmico CIGRE
|-- simulation.py           # Módulo para a Simulação de Monte Carlo
|-- risk_analysis.py        # Módulo para calcular o risco e a ampacidade
|
|-- /dados/                   # Pasta contendo os CSVs das estações
|   |-- estacao_A.csv
|   `-- estacao_B.csv
|
|-- /entrada/                 # Pasta para outros arquivos de entrada
|   |-- trassado_linha.xlsx
|   `-- parametros_cabo.json
|
|-- /saida/                   # Pasta para os resultados gerados
|   `-- resultado_horario.csv
|
`-- requirements.txt        # Lista de dependências Python

3. Detalhamento dos Módulos
Módulo 1: data_loader.py
Responsabilidade: Carregar, validar e pré-processar todos os dados de entrada.

Funcionalidades:

Carregar Dados da Linha:

Ler o arquivo trassado_linha.xlsx.

Validação: Verificar a existência das colunas Progressiva, azimute, latitude, longitude.

Armazenar em um DataFrame Pandas.

Carregar Dados das Estações:

Ler todos os arquivos .csv da pasta dados/.

Extrair as coordenadas (Latitude, Longitude) do cabeçalho de cada arquivo.

Validação: Verificar se os cabeçalhos das colunas de dados correspondem ao formato esperado (ex: 'Data Medicao', 'TEMPERATURA DO AR...', 'RADIACAO GLOBAL', 'VENTO, VELOCIDADE HORARIA (m/s)', 'VENTO, DIRECCAO HORARIA (gr)'). Existe um arquivo chamada exemplo_estacao.csv para você ler e entender como os dados estão ajustados.

Faça tratamento de erro, retirando linhas com dados invalidos.

Converter a coluna de data/hora para o formato datetime.

Pré-processamento:

Decompor o vetor do vento (velocidade e direção) em componentes U (zonal) e V (meridional) para cada registro horário. A direção está em graus.

Tratar dados faltantes (NaNs ou outros erros) com uma estratégia definida (ex: logar um aviso e remover a linha/hora). Como vai ser feita uma interpolação a acada hora deve-se garantir que para aquela hora todos os conjuntos de dados de todas as estações são válidos, caso alguma hora de alguma substação seja invalido essa hora deverá ser descartada de todas as estações. No fim registrar resumo dos dados utilizados.

Consolidar os dados de todas as estações em um único DataFrame ou em uma estrutura de dados adequada (ex: dicionário de DataFrames).

Carregar Parâmetros do Condutor:

Ler um arquivo de configuração (ex: parametros_cabo.json) contendo as propriedades físicas do cabo (diâmetro, resistência AC a 25°C e 75°C, emissividade, absortividade). Crie um exemplo inicial.

Validação: Garantir que todas as chaves necessárias estão presentes.

Módulo 2: geoprocessing.py
Responsabilidade: Realizar todo o processamento geoespacial e a interpolação por Krigagem.

Funcionalidades:

Conversão de Coordenadas:

Implementar uma função que recebe as coordenadas das estações e do traçado da linha (em SIRGAS 2000, EPSG:4674) e as converte para o sistema projetado Brazil Polyconic (EPSG:5880). As coordenadas convertidas (em metros) serão usadas para os cálculos de distância da Krigagem.

Discretização da Linha:

Implementar uma função que recebe o traçado da linha e o divide em N pontos de cálculo equidistantes (ex: a cada 1 km). Para cada ponto, deve-se ter suas coordenadas projetadas e seu azimute.

Krigagem Horária:

Criar uma classe ou função que, para uma hora específica, receba os dados de todas as estações para uma variável ambiental (Ta, Qs, U ou V).

Executar a Krigagem Ordinária para estimar o valor médio e a variância da krigagem em cada um dos pontos discretizados da linha.

O processo deve ser repetido para cada hora do período de análise e para cada uma das 4 variáveis.

Saída: Uma estrutura de dados que armazene, para cada ponto da linha e para cada hora, a média e o desvio padrão (raiz da variância) de Ta, Qs, U e V.

Módulo 3: thermal_model.py
Responsabilidade: Implementar o modelo de balanço térmico do CIGRE.

Funcionalidades:

Classe CigreModeloTermico:

O construtor deve receber os parâmetros do condutor. Leia o arquivo 601.pdf para entender profundamente o metodo de cálculo

Implementar métodos privados para cada componente do balanço térmico:

_calculate_joule_heating(current, conductor_temp): P_J=I 
2
 
cdotR_ac(T_c)

_calculate_solar_heating(solar_rad, line_azimuth, ...): P_S

_calculate_convective_cooling(wind_speed, wind_angle, air_temp, conductor_temp): P_c

_calculate_radiative_cooling(air_temp, conductor_temp): P_r

Implementar um método público solve_conductor_temperature(...) que utilize um solver numérico (ex: scipy.optimize.fsolve) para encontrar a T_c que satisfaz a equação P_J+P_S−P_c−P_r=0.

Módulo 4: simulation.py
Responsabilidade: Orquestrar e executar a Simulação de Monte Carlo.

Funcionalidades:

Classe MonteCarloSimulator:

O construtor deve receber uma instância do CigreModeloTermico.

Implementar um método run_simulation(environmental_means, environmental_stds, line_azimuth, current, num_iterations) que:

Receba as médias e desvios-padrão de Ta, Qs, U, V para um ponto da linha em uma hora específica.

Execute um loop de num_iterations (ex: 10.000).

Dentro do loop:

Amostre valores aleatórios para Ta, Qs, U, V a partir de distribuições normais (np.random.normal).

Reconstrua a velocidade do vento (W_s) e a direção do vento a partir dos U e V amostrados.

Calcule o ângulo de ataque do vento em relação ao condutor usando o azimute da linha.

Chame thermal_model.solve_conductor_temperature() com os dados ambientais amostrados e a corrente daquela hora (vamos considerar uma corrente constante).

Calcular o percentil desejado (ex: 90%) para obter a temperatura que satisfaz o intervalo de confiança

Saída: temperatura do condutor para cada hora.



Módulo 5: risk_analysis.py
Responsabilidade: Analisar os resultados da simulação e calcular as métricas finais.

Funcionalidades:

Calcular Temperatura de Confiança:

Receber a distribuição de temperaturas do Monte Carlo.

.

Calcular Risco Térmico:

Receber a distribuição de temperaturas e uma temperatura máxima de projeto (T_max).

Calcular a probabilidade de exceder T_max (contar quantos valores na distribuição são T_max e dividir pelo número total de iterações).

Verificar risco térmico conforme NBR 5422 ( ver arquivo 5422.pdf)

Módulo 6: main.py
Responsabilidade: Orquestrar todo o fluxo de trabalho.

Funcionalidades:

Importar os módulos e classes.

Carregar configurações do config.py.

Instanciar o DataLoader e carregar todos os dados.

Processar os dados geoespaciais e executar a Krigagem para todas as horas, gerando os campos de média e desvio padrão.

Iterar sobre cada ponto da linha e cada hora do período de análise:

Obter a corrente elétrica para a hora atual.

Obter as médias e desvios-padrão das variáveis ambientais para o ponto/hora atual.

Executar a simulação de Monte Carlo.

Executar a análise de risco para obter a temperatura do percentil 90 e o risco térmico.

Armazenar os resultados.

Salvar os resultados finais em um arquivo CSV na pasta saida/.

4. Dependências
O arquivo requirements.txt deve conter:

pandas
openpyxl
numpy
scipy
pyproj
pykrige
matplotlib  # Para visualização opcional dos resultados
