# Módulo para visualização dos resultados da análise
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os

logger = logging.getLogger(__name__)

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class VisualizadorResultados:
    """Classe para criação de visualizações dos resultados da análise térmica."""
    
    def __init__(self, diretorio_saida: str = None):
        """
        Inicializa o visualizador.
        
        Args:
            diretorio_saida: Diretório para salvar os gráficos
        """
        self.diretorio_saida = diretorio_saida or "saida/graficos"
        self._criar_diretorio_saida()
        
        # Configurações de cores para classificação de risco
        self.cores_risco = {
            'baixo': '#2ecc71',      # Verde
            'moderado': '#f39c12',   # Laranja
            'alto': '#e74c3c',       # Vermelho
            'critico': '#8e44ad'     # Roxo
        }
        
    def _criar_diretorio_saida(self):
        """Cria o diretório de saída se não existir."""
        os.makedirs(self.diretorio_saida, exist_ok=True)
        
    def plotar_distribuicao_temperaturas(self, temperaturas: np.ndarray, 
                                       temperatura_limite: float = 75,
                                       titulo: str = "Distribuição de Temperaturas do Condutor",
                                       salvar: bool = True) -> plt.Figure:
        """
        Plota histograma da distribuição de temperaturas.
        
        Args:
            temperaturas: Array de temperaturas
            temperatura_limite: Temperatura limite para referência
            titulo: Título do gráfico
            salvar: Se deve salvar o arquivo
            
        Returns:
            Figure do matplotlib
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histograma
        ax1.hist(temperaturas, bins=50, density=True, alpha=0.7, 
                color='skyblue', edgecolor='black')
        ax1.axvline(temperatura_limite, color='red', linestyle='--', 
                   linewidth=2, label=f'Limite ({temperatura_limite}°C)')
        ax1.axvline(np.mean(temperaturas), color='green', linestyle='-', 
                   linewidth=2, label=f'Média ({np.mean(temperaturas):.1f}°C)')
        ax1.axvline(np.percentile(temperaturas, 90), color='orange', linestyle='-', 
                   linewidth=2, label=f'P90 ({np.percentile(temperaturas, 90):.1f}°C)')
        
        ax1.set_xlabel('Temperatura (°C)')
        ax1.set_ylabel('Densidade')
        ax1.set_title('Histograma de Temperaturas')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(temperaturas, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.axhline(temperatura_limite, color='red', linestyle='--', 
                   linewidth=2, label=f'Limite ({temperatura_limite}°C)')
        ax2.set_ylabel('Temperatura (°C)')
        ax2.set_title('Box Plot de Temperaturas')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(titulo, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if salvar:
            caminho = os.path.join(self.diretorio_saida, 'distribuicao_temperaturas.png')
            plt.savefig(caminho, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo: {caminho}")
        
        return fig
    
    def plotar_serie_temporal_risco(self, resultados_df: pd.DataFrame,
                                   salvar: bool = True) -> plt.Figure:
        """
        Plota série temporal do risco térmico.
        
        Args:
            resultados_df: DataFrame com resultados da análise
            salvar: Se deve salvar o arquivo
            
        Returns:
            Figure do matplotlib
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Converter hora para datetime se necessário
        if 'hora' in resultados_df.columns:
            try:
                horas = pd.to_datetime(resultados_df['hora'])
            except:
                horas = resultados_df['hora']
        else:
            horas = resultados_df.index
        
        # 1. Temperatura do condutor
        ax1 = axes[0]
        ax1.plot(horas, resultados_df['temperatura_condutor_p90'], 
                'b-', linewidth=1.5, label='Temperatura P90')
        ax1.plot(horas, resultados_df['temperatura_condutor_media'], 
                'g-', linewidth=1, alpha=0.7, label='Temperatura Média')
        
        if 'temperatura_ar_media' in resultados_df.columns:
            ax1.plot(horas, resultados_df['temperatura_ar_media'], 
                    'orange', linewidth=1, alpha=0.7, label='Temperatura do Ar')
        
        ax1.axhline(75, color='red', linestyle='--', alpha=0.8, label='Limite (75°C)')
        ax1.set_ylabel('Temperatura (°C)')
        ax1.set_title('Evolução Temporal da Temperatura do Condutor')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Risco térmico
        ax2 = axes[1]
        cores_risco_valores = resultados_df['risco_termico'].apply(self._obter_cor_risco)
        ax2.scatter(horas, resultados_df['risco_termico'], 
                   c=cores_risco_valores, alpha=0.6, s=10)
        ax2.axhline(0.01, color='green', linestyle=':', alpha=0.8, label='Risco Baixo (1%)')
        ax2.axhline(0.05, color='orange', linestyle=':', alpha=0.8, label='Risco Moderado (5%)')
        ax2.axhline(0.10, color='red', linestyle=':', alpha=0.8, label='Risco Alto (10%)')
        
        ax2.set_ylabel('Probabilidade de Excedência')
        ax2.set_title('Evolução Temporal do Risco Térmico')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 3. Ampacidade
        ax3 = axes[2]
        if 'ampacidade_calculada' in resultados_df.columns:
            ax3.plot(horas, resultados_df['ampacidade_calculada'], 
                    'purple', linewidth=1.5, label='Ampacidade Calculada')
            if 'corrente_operacao' in resultados_df.columns:
                ax3.axhline(resultados_df['corrente_operacao'].iloc[0], 
                           color='red', linestyle='--', alpha=0.8, label='Corrente de Operação')
        
        ax3.set_ylabel('Corrente (A)')
        ax3.set_xlabel('Tempo')
        ax3.set_title('Evolução Temporal da Ampacidade')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Formatação do eixo temporal
        for ax in axes:
            if isinstance(horas.iloc[0], (pd.Timestamp, datetime)):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if salvar:
            caminho = os.path.join(self.diretorio_saida, 'serie_temporal_risco.png')
            plt.savefig(caminho, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo: {caminho}")
        
        return fig
    
    def plotar_mapa_calor_espacial(self, resultados_df: pd.DataFrame,
                                  variavel: str = 'temperatura_condutor_p90',
                                  salvar: bool = True) -> plt.Figure:
        """
        Plota mapa de calor espacial ao longo da linha.
        
        Args:
            resultados_df: DataFrame com resultados
            variavel: Variável para plotar
            salvar: Se deve salvar o arquivo
            
        Returns:
            Figure do matplotlib
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Preparar dados para heatmap
        if 'progressiva' in resultados_df.columns and 'hora' in resultados_df.columns:
            # Criar pivot table
            pivot_data = resultados_df.pivot_table(
                values=variavel,
                index='hora',
                columns='progressiva',
                aggfunc='mean'
            )
            
            # Plotar heatmap
            im = ax.imshow(pivot_data.values, aspect='auto', cmap='hot', interpolation='bilinear')
            
            # Configurar eixos
            ax.set_xticks(range(0, len(pivot_data.columns), max(1, len(pivot_data.columns)//10)))
            ax.set_xticklabels([f'{x/1000:.1f}' for x in pivot_data.columns[::max(1, len(pivot_data.columns)//10)]])
            ax.set_xlabel('Posição na Linha (km)')
            
            ax.set_yticks(range(0, len(pivot_data.index), max(1, len(pivot_data.index)//10)))
            ax.set_yticklabels(pivot_data.index[::max(1, len(pivot_data.index)//10)])
            ax.set_ylabel('Hora')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f'{variavel}')
            
            plt.title(f'Mapa de Calor: {variavel} ao longo da linha')
        else:
            # Gráfico de linha simples se não houver dados espaciais/temporais
            if 'progressiva' in resultados_df.columns:
                ax.plot(resultados_df['progressiva']/1000, resultados_df[variavel], 'b-', linewidth=2)
                ax.set_xlabel('Posição na Linha (km)')
            else:
                ax.plot(resultados_df[variavel], 'b-', linewidth=2)
                ax.set_xlabel('Ponto da Linha')
            
            ax.set_ylabel(variavel)
            ax.set_title(f'Variação Espacial: {variavel}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if salvar:
            caminho = os.path.join(self.diretorio_saida, f'mapa_calor_{variavel}.png')
            plt.savefig(caminho, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo: {caminho}")
        
        return fig
    
    def plotar_analise_sensibilidade(self, sensibilidades: Dict[str, float],
                                   salvar: bool = True) -> plt.Figure:
        """
        Plota análise de sensibilidade das variáveis.
        
        Args:
            sensibilidades: Dicionário com sensibilidades por variável
            salvar: Se deve salvar o arquivo
            
        Returns:
            Figure do matplotlib
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        variaveis = list(sensibilidades.keys())
        valores = list(sensibilidades.values())
        
        # Remover valores NaN
        dados_validos = [(v, val) for v, val in zip(variaveis, valores) if not np.isnan(val)]
        
        if not dados_validos:
            ax.text(0.5, 0.5, 'Nenhum dado válido de sensibilidade', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        variaveis_validas, valores_validos = zip(*dados_validos)
        
        # Criar gráfico de barras
        cores = plt.cm.viridis(np.linspace(0, 1, len(variaveis_validas)))
        bars = ax.bar(variaveis_validas, valores_validos, color=cores)
        
        # Adicionar valores nas barras
        for bar, valor in zip(bars, valores_validos):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{valor:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel('Sensibilidade (adimensional)')
        ax.set_title('Análise de Sensibilidade das Variáveis Ambientais')
        ax.set_xticklabels(variaveis_validas, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if salvar:
            caminho = os.path.join(self.diretorio_saida, 'analise_sensibilidade.png')
            plt.savefig(caminho, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo: {caminho}")
        
        return fig
    
    def plotar_dashboard_resumo(self, resultados_df: pd.DataFrame,
                               relatorio_qualidade: Dict = None,
                               salvar: bool = True) -> plt.Figure:
        """
        Cria dashboard com resumo da análise.
        
        Args:
            resultados_df: DataFrame com resultados
            relatorio_qualidade: Relatório de qualidade dos dados
            salvar: Se deve salvar o arquivo
            
        Returns:
            Figure do matplotlib
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Layout do dashboard
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Estatísticas principais (texto)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plotar_estatisticas_principais(ax1, resultados_df)
        
        # 2. Distribuição de risco
        ax2 = fig.add_subplot(gs[0, 1])
        self._plotar_distribuicao_risco(ax2, resultados_df)
        
        # 3. Qualidade dos dados
        ax3 = fig.add_subplot(gs[0, 2])
        self._plotar_qualidade_dados(ax3, relatorio_qualidade)
        
        # 4. Série temporal temperatura
        ax4 = fig.add_subplot(gs[1, :])
        self._plotar_temperatura_resumo(ax4, resultados_df)
        
        # 5. Variação espacial
        ax5 = fig.add_subplot(gs[2, :2])
        self._plotar_variacao_espacial(ax5, resultados_df)
        
        # 6. Histograma temperaturas
        ax6 = fig.add_subplot(gs[2, 2])
        self._plotar_histograma_resumo(ax6, resultados_df)
        
        plt.suptitle('Dashboard - Análise de Risco Térmico', fontsize=16, fontweight='bold')
        
        if salvar:
            caminho = os.path.join(self.diretorio_saida, 'dashboard_resumo.png')
            plt.savefig(caminho, dpi=300, bbox_inches='tight')
            logger.info(f"Dashboard salvo: {caminho}")
        
        return fig
    
    def _obter_cor_risco(self, valor_risco: float) -> str:
        """Retorna cor baseada no valor do risco."""
        if valor_risco < 0.01:
            return self.cores_risco['baixo']
        elif valor_risco < 0.05:
            return self.cores_risco['moderado']
        elif valor_risco < 0.10:
            return self.cores_risco['alto']
        else:
            return self.cores_risco['critico']
    
    def _plotar_estatisticas_principais(self, ax, resultados_df):
        """Plota estatísticas principais como texto."""
        ax.axis('off')
        
        # Calcular estatísticas
        temp_media = resultados_df['temperatura_condutor_p90'].mean()
        temp_max = resultados_df['temperatura_condutor_p90'].max()
        risco_medio = resultados_df['risco_termico'].mean()
        risco_max = resultados_df['risco_termico'].max()
        
        texto = f"""ESTATÍSTICAS PRINCIPAIS
        
Temperatura P90:
  Média: {temp_media:.1f}°C
  Máxima: {temp_max:.1f}°C

Risco Térmico:
  Médio: {risco_medio:.4f} ({risco_medio*100:.2f}%)
  Máximo: {risco_max:.4f} ({risco_max*100:.2f}%)

Total de Pontos: {len(resultados_df):,}
"""
        
        ax.text(0.05, 0.95, texto, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace')
    
    def _plotar_distribuicao_risco(self, ax, resultados_df):
        """Plota distribuição de risco em pizza."""
        # Classificar riscos
        riscos = resultados_df['risco_termico']
        
        baixo = (riscos < 0.01).sum()
        moderado = ((riscos >= 0.01) & (riscos < 0.05)).sum()
        alto = ((riscos >= 0.05) & (riscos < 0.10)).sum()
        critico = (riscos >= 0.10).sum()
        
        valores = [baixo, moderado, alto, critico]
        labels = ['Baixo', 'Moderado', 'Alto', 'Crítico']
        cores = [self.cores_risco['baixo'], self.cores_risco['moderado'], 
                self.cores_risco['alto'], self.cores_risco['critico']]
        
        # Remover categorias sem dados
        dados_validos = [(v, l, c) for v, l, c in zip(valores, labels, cores) if v > 0]
        
        if dados_validos:
            valores_val, labels_val, cores_val = zip(*dados_validos)
            ax.pie(valores_val, labels=labels_val, colors=cores_val, autopct='%1.1f%%')
        
        ax.set_title('Distribuição de Risco')
    
    def _plotar_qualidade_dados(self, ax, relatorio_qualidade):
        """Plota qualidade dos dados."""
        ax.axis('off')
        
        if relatorio_qualidade and 'resumo_geral' in relatorio_qualidade:
            resumo = relatorio_qualidade['resumo_geral']
            texto = f"""QUALIDADE DOS DADOS
            
Estações: {resumo.get('num_estacoes', 'N/A')}
Aproveitamento: {resumo.get('taxa_aproveitamento', 0):.1%}
Qualidade: {resumo.get('qualidade_geral', 'N/A')}

Registros:
  Originais: {resumo.get('registros_originais', 0):,}
  Válidos: {resumo.get('registros_validos', 0):,}
"""
        else:
            texto = "QUALIDADE DOS DADOS\n\nRelatório não disponível"
        
        ax.text(0.05, 0.95, texto, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace')
    
    def _plotar_temperatura_resumo(self, ax, resultados_df):
        """Plota série temporal de temperatura resumida."""
        if 'hora' in resultados_df.columns:
            try:
                horas = pd.to_datetime(resultados_df['hora'])
                ax.plot(horas, resultados_df['temperatura_condutor_p90'], 'b-', linewidth=1)
                ax.axhline(75, color='red', linestyle='--', alpha=0.8)
                ax.set_xlabel('Tempo')
            except:
                ax.plot(resultados_df['temperatura_condutor_p90'], 'b-', linewidth=1)
                ax.axhline(75, color='red', linestyle='--', alpha=0.8)
                ax.set_xlabel('Registro')
        else:
            ax.plot(resultados_df['temperatura_condutor_p90'], 'b-', linewidth=1)
            ax.axhline(75, color='red', linestyle='--', alpha=0.8)
            ax.set_xlabel('Registro')
        
        ax.set_ylabel('Temperatura P90 (°C)')
        ax.set_title('Evolução Temporal - Temperatura P90')
        ax.grid(True, alpha=0.3)
    
    def _plotar_variacao_espacial(self, ax, resultados_df):
        """Plota variação espacial."""
        if 'progressiva' in resultados_df.columns:
            # Agrupar por posição
            agrupado = resultados_df.groupby('progressiva').agg({
                'temperatura_condutor_p90': 'mean',
                'risco_termico': 'mean'
            }).reset_index()
            
            ax.plot(agrupado['progressiva']/1000, agrupado['temperatura_condutor_p90'], 
                   'b-', linewidth=2, label='Temperatura P90')
            ax.set_xlabel('Posição na Linha (km)')
            ax.set_ylabel('Temperatura P90 (°C)', color='b')
            ax.tick_params(axis='y', labelcolor='b')
            
            # Segundo eixo para risco
            ax2 = ax.twinx()
            ax2.plot(agrupado['progressiva']/1000, agrupado['risco_termico'], 
                    'r-', linewidth=2, label='Risco Térmico')
            ax2.set_ylabel('Risco Térmico', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
        else:
            ax.plot(resultados_df['temperatura_condutor_p90'], 'b-', linewidth=2)
            ax.set_xlabel('Ponto da Linha')
            ax.set_ylabel('Temperatura P90 (°C)')
        
        ax.set_title('Variação Espacial')
        ax.grid(True, alpha=0.3)
    
    def _plotar_histograma_resumo(self, ax, resultados_df):
        """Plota histograma resumido."""
        ax.hist(resultados_df['temperatura_condutor_p90'], bins=20, 
               density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(75, color='red', linestyle='--', linewidth=2)
        ax.axvline(resultados_df['temperatura_condutor_p90'].mean(), 
                  color='green', linestyle='-', linewidth=2)
        ax.set_xlabel('Temperatura P90 (°C)')
        ax.set_ylabel('Densidade')
        ax.set_title('Distribuição de Temperaturas')
        ax.grid(True, alpha=0.3)