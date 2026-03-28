# -*- coding: utf-8 -*-
"""
Painel de Análise de Ações e Indicadores Fundamentalistas
==========================================================

Sistema completo para análise fundamentalista de ações, incluindo:
- Cálculo de Preço Teto (Método Bazin)
- Preço Justo (Método Graham)
- Indicadores fundamentalistas (P/L, P/VP, ROE, DY, etc.)
- Análise de sazonalidade e dividendos
- Cálculo de CAGR de receita e lucro

Autor: Sistema de Análise Financeira
Versão: 2.0 (Melhorada)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import io
from typing import Tuple, Dict, List, Optional
import logging

# ==============================================================================
# CONFIGURAÇÕES GLOBAIS
# ==============================================================================

# Configuração de logging para melhor rastreamento de erros
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constantes do sistema
FUSO_HORARIO_PADRAO = 'America/Sao_Paulo'
MAX_WORKERS_THREAD_POOL = 5
ANOS_PADRAO = 5
TAXA_RETORNO_PADRAO = 6.0
MULTIPLICADOR_GRAHAM = 22.5  # Fórmula de Benjamin Graham
MAX_TICKERS_PERMITIDOS = 20  # Melhoria 7: limite para evitar sobrecarga da API

# Helper global de fuso horário — evita repetir pytz.timezone() e datetime.now()
# em cada função que precisa da hora atual no fuso de São Paulo
def _agora_sp() -> datetime:
    """Retorna datetime.now() no fuso horário de São Paulo."""
    return datetime.now(pytz.timezone(FUSO_HORARIO_PADRAO))


# Tickers padrão para exemplificação
TICKERS_EXEMPLO = (
    "BBAS3.SA, BBSE3.SA, BRAP4.SA, CMIG4.SA, "
    "CXSE3.SA, EGIE3.SA, ISAE4.SA, ITSA4.SA ,PETR4.SA, UNIP6.SA"
)

# ==============================================================================
# CONFIGURAÇÃO DA INTERFACE STREAMLIT
# ==============================================================================

st.set_page_config(
    page_title="Análise de Ações e Indicadores Fundamentalistas",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para interface profissional + Meta tag UTF-8
st.markdown("""
    <meta charset="UTF-8">
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
    <style>
    .main { background-color: #fcfcfc; }
    .stDataFrame { border: 1px solid #e6e9ef; border-radius: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 2px solid #eee; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; background-color: transparent; 
        border: none; font-weight: 600; font-size: 16px;
    }
    .stAlert { border-radius: 12px; border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .explicacao-container {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #eaeaea;
        margin-top: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# FUNÇÕES DE VALIDAÇÃO E SANITIZAÇÃO
# ==============================================================================

def validar_ticker(ticker: str) -> bool:
    """
    Valida o formato de um ticker de ação.
    
    Args:
        ticker: String com o símbolo do ticker
        
    Returns:
        bool: True se o ticker é válido, False caso contrário
        
    Exemplos:
        >>> validar_ticker("PETR4.SA")
        True
        >>> validar_ticker("INVALID@#$")
        False
    """
    if not ticker or not isinstance(ticker, str):
        return False
    
    # Padrão: Letras, números, ponto e hífen (sem caracteres especiais perigosos)
    padrao = re.compile(r'^[A-Z0-9\.\-]+$')
    return bool(padrao.match(ticker.strip().upper()))


def sanitizar_tickers(ticker_input: str) -> List[str]:
    """
    Sanitiza e valida uma lista de tickers inseridos pelo usuário.
    
    Args:
        ticker_input: String com tickers separados por vírgula
        
    Returns:
        List[str]: Lista de tickers válidos e sanitizados
        
    Exemplos:
        >>> sanitizar_tickers("PETR4.SA, VALE3.SA")
        ['PETR4.SA', 'VALE3.SA']
    """
    if not ticker_input:
        return []
    
    tickers_brutos = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    tickers_validos = [t for t in tickers_brutos if validar_ticker(t)]
    
    if len(tickers_validos) < len(tickers_brutos):
        tickers_invalidos = set(tickers_brutos) - set(tickers_validos)
        logger.warning(f"Tickers inválidos removidos: {tickers_invalidos}")
        st.warning(
            f"⚠️ Alguns tickers foram removidos por formato inválido: "
            f"{', '.join(tickers_invalidos)}"
        )
    
    # Melhoria 7: limita quantidade máxima de tickers para evitar sobrecarga da API
    if len(tickers_validos) > MAX_TICKERS_PERMITIDOS:
        st.warning(
            f"⚠️ Limite de {MAX_TICKERS_PERMITIDOS} tickers por análise. "
            f"Os {len(tickers_validos) - MAX_TICKERS_PERMITIDOS} ticker(s) excedentes foram removidos."
        )
        tickers_validos = tickers_validos[:MAX_TICKERS_PERMITIDOS]
    
    return tickers_validos


def validar_anos(anos) -> bool:
    """
    Valida o número de anos para análise.
    
    Args:
        anos: Número de anos (int ou float — st.number_input retorna float por padrão)
        
    Returns:
        bool: True se válido (entre 1 e 20 anos)
        
    Notas:
        - Melhoria 6: aceita float além de int, pois st.number_input retorna float
          (ex: 5.0 em vez de 5), evitando falha silenciosa na validação
    """
    return isinstance(anos, (int, float)) and 1 <= int(anos) <= 20


def validar_taxa_retorno(taxa: float) -> bool:
    """
    Valida a taxa de retorno.
    
    Args:
        taxa: Taxa de retorno percentual
        
    Returns:
        bool: True se válida (entre 0.1% e 100%)
    """
    return isinstance(taxa, (int, float)) and 0.1 <= taxa <= 100.0


# ==============================================================================
# FUNÇÕES DE COLETA DE DADOS - COM CACHE E TRATAMENTO ROBUSTO DE ERROS
# ==============================================================================

@st.cache_data(ttl=3600)  # Cache de 1 hora
def obter_sazonalidade_e_dividendos(
    ticker_symbol: str, 
    anos: int
) -> Tuple[str, str]:
    """
    Analisa sazonalidade de preços e meses de pagamento de dividendos.
    
    Esta função identifica:
    1. Os meses em que historicamente os dividendos são pagos
    2. Os 3 meses do ano onde o preço da ação tende a ser mais baixo
    
    Args:
        ticker_symbol: Símbolo do ticker (ex: "PETR4.SA")
        anos: Número de anos para análise histórica
        
    Returns:
        Tuple contendo:
            - str: Meses de pagamento de dividendos (ex: "Jan, Jul")
            - str: Meses com preços historicamente mais baixos (ex: "Mar, Jun, Set")
            
    Raises:
        Retorna ("Erro", "Erro") em caso de falha na coleta de dados
        
    Notas:
        - Usa threshold de 30% dos anos para considerar um mês recorrente
        - Calcula média mensal de preços para identificar sazonalidade
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period=f"{anos}y")
        dividends = ticker.dividends
        
        # Validação de dados históricos
        if hist.empty:
            logger.warning(f"Sem dados históricos para {ticker_symbol}")
            return "N/A", "N/A"
        
        # Análise de meses de pagamento de dividendos/JCP — método de consistência anual
        #
        # Correção estatística v4.8:
        # A abordagem anterior usava todo o histórico do yfinance (sem filtro de data)
        # e contava eventos brutos de pagamento, não anos distintos com pagamento.
        # Isso gerava dois problemas:
        #   1. Empresas que pagam mensalmente (FIIs) teriam todos os 12 meses listados,
        #      pois a contagem de eventos sempre superava o threshold trivialmente.
        #   2. O período analisado não respeitava a janela de anos escolhida pelo usuário.
        #
        # Método correto:
        #   - Filtra dividendos estritamente dentro do período escolhido pelo usuário
        #   - Para cada mês (1-12), conta em quantos ANOS DISTINTOS houve pagamento
        #   - Calcula a frequência: anos_com_pagamento / total_anos_no_período
        #   - Exibe apenas meses com frequência >= 60% (consistentes ao longo dos anos)
        #   - Tolera naturalmente 1 ou 2 anos sem pagamento num histórico longo
        #
        # Threshold de 60%:
        #   5 anos  → mínimo 3 anos com pagamento no mês
        #   10 anos → mínimo 6 anos com pagamento no mês
        #   Imune ao número de eventos por mês (paga 1x ou 4x no mês: conta como 1 ano)

        THRESHOLD_CONSISTENCIA = 0.60  # 60% dos anos analisados

        if not dividends.empty:
            # 1. Filtra dividendos pelo período escolhido pelo usuário
            fuso_horario = pytz.timezone(FUSO_HORARIO_PADRAO)
            end_date   = datetime.now(fuso_horario)
            start_date = end_date - timedelta(days=anos * 365)

            # Garante compatibilidade de timezone no índice dos dividendos
            div_idx = dividends.index
            if div_idx.tzinfo is None:
                div_idx = div_idx.tz_localize('UTC').tz_convert(fuso_horario)
            else:
                div_idx = div_idx.tz_convert(fuso_horario)

            div_filtrado = dividends[
                (div_idx >= start_date) & (div_idx <= end_date)
            ].copy()
            div_filtrado.index = div_idx[
                (div_idx >= start_date) & (div_idx <= end_date)
            ]

            if not div_filtrado.empty:
                # 2. Para cada mês, identifica em quantos anos distintos houve pagamento
                div_df = pd.DataFrame({
                    'mes': div_filtrado.index.month,
                    'ano': div_filtrado.index.year
                })

                # Anos distintos com pelo menos 1 pagamento em cada mês
                anos_por_mes = (
                    div_df.groupby('mes')['ano']
                    .nunique()
                )

                # Total de anos distintos no período filtrado
                total_anos = div_df['ano'].nunique()
                if total_anos == 0:
                    total_anos = anos  # fallback seguro

                # 3. Filtra meses com frequência >= threshold
                frequencia_por_mes = anos_por_mes / total_anos
                meses_consistentes = (
                    frequencia_por_mes[frequencia_por_mes >= THRESHOLD_CONSISTENCIA]
                    .index.sort_values()
                )

                nomes_meses_pagto = [
                    datetime(2000, m, 1).strftime('%b')
                    for m in meses_consistentes
                ]
                col_pagto = ", ".join(nomes_meses_pagto) if nomes_meses_pagto else "Sem Div."
            else:
                col_pagto = "Sem Div."
        else:
            col_pagto = "Sem Div."

        # Análise de sazonalidade de preços — método de desvio relativo anual
        #
        # Correção estatística v4.7:
        # A abordagem anterior calculava a média simples do preço absoluto por mês
        # ao longo dos anos. Isso é enviesado: meses antigos têm preços nominais
        # menores simplesmente porque a ação valia menos naquele tempo, não porque
        # o mês seja sazonalmente mais barato.
        #
        # Método correto: para cada ano, calcula o desvio percentual de cada mês
        # em relação à média anual daquele ano. Depois agrega esses desvios ao
        # longo de todos os anos. Os meses com maior desvio negativo médio são
        # aqueles que, recorrentemente, ficam abaixo da média do próprio ano —
        # um padrão sazonal real, independente do nível de preço absoluto.
        hist['Month'] = hist.index.month
        hist['Year']  = hist.index.year

        # Média anual de preço para cada ano
        media_anual = hist.groupby('Year')['Close'].mean()

        # Desvio percentual de cada dia em relação à média anual do seu ano
        hist['DesvioRelativo'] = hist.apply(
            lambda row: ((row['Close'] / media_anual[row['Year']]) - 1) * 100
            if media_anual[row['Year']] > 0 else np.nan,
            axis=1
        )

        # Média do desvio relativo por mês (agrega todos os anos)
        desvio_mensal = hist.groupby('Month')['DesvioRelativo'].mean()

        # Os 3 meses com maior desvio negativo médio = historicamente mais baratos
        meses_baratos = desvio_mensal.nsmallest(3).index.sort_values()
        desvios_baratos = desvio_mensal[meses_baratos]
        nomes_meses_baratos = [
            f"{datetime(2000, m, 1).strftime('%b')} ({desvios_baratos[m]:.1f}%)"
            for m in meses_baratos
        ]
        col_sazonalidade = ", ".join(nomes_meses_baratos)
        
        return col_pagto, col_sazonalidade
        
    except Exception as e:
        logger.error(f"Erro ao obter sazonalidade para {ticker_symbol}: {str(e)}")
        return "Erro", "Erro"


@st.cache_data(ttl=3600)  # Cache de 1 hora
def obter_dados_dpa(
    ticker_symbol: str, 
    anos: int
) -> Tuple[pd.DataFrame, float, float, str]:
    """
    Obtém e calcula estatísticas de Dividendos Por Ação (DPA).
    
    Coleta histórico de dividendos e calcula:
    - DPA anual para cada ano
    - Média de DPA do período (exatamente N anos solicitados)
    - Mediana de DPA do período (exatamente N anos solicitados)
    
    Args:
        ticker_symbol: Símbolo do ticker (ex: "PETR4.SA")
        anos: Número EXATO de anos para análise
        
    Returns:
        Tuple contendo:
            - DataFrame: Dados anuais de DPA (exatamente N anos mais recentes completos)
            - float: Média de DPA (calculada sobre exatamente N anos)
            - float: Mediana de DPA (calculada sobre exatamente N anos)
            - str: Mensagem de erro (vazio se sucesso)
            
    Notas:
        - Exclui o ano corrente (incompleto) da análise
        - Usa fuso horário de São Paulo
        - Pega os N anos mais recentes completos para cálculo
        - Retorna valores zerados em caso de erro
        
    Exemplo:
        Se usuário escolhe 5 anos em 2026:
        - Usa anos: 2021, 2022, 2023, 2024, 2025 (exatamente 5 anos)
        - Calcula média e mediana apenas desses 5 anos
    """
    fuso_horario = pytz.timezone(FUSO_HORARIO_PADRAO)
    end_date = datetime.now(fuso_horario)
    start_date = end_date - timedelta(days=anos * 365)
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        dividends = ticker.dividends
        
        # Validação de dividendos
        if dividends.empty:
            logger.info(f"Sem histórico de dividendos para {ticker_symbol}")
            return pd.DataFrame(), 0.0, 0.0, ""
        
        # Filtra dividendos pelo período solicitado
        div_filtrado = dividends.loc[start_date:end_date]
        
        # Agrupa por ano e soma os dividendos
        div_anual = div_filtrado.resample('YE').sum().reset_index()
        div_anual.columns = ['Ano', 'DPA']
        div_anual['Ano'] = div_anual['Ano'].dt.year
        div_anual['Ticker'] = ticker_symbol
        
        # Exclui ano corrente (incompleto)
        div_completo = div_anual[
            div_anual['Ano'] < datetime.now(fuso_horario).year
        ].copy()
        
        # IMPORTANTE: Limita aos N anos mais recentes solicitados pelo usuário
        # Ordena por ano decrescente e pega apenas os N anos solicitados
        div_completo = div_completo.sort_values('Ano', ascending=False).head(anos).copy()
        
        # Calcula estatísticas SOMENTE dos N anos solicitados
        if not div_completo.empty and len(div_completo) > 0:
            media_dpa = float(div_completo['DPA'].mean())
            mediana_dpa = float(div_completo['DPA'].median())
        else:
            media_dpa = 0.0
            mediana_dpa = 0.0
        
        # Reordena para exibição (mais antigo para mais recente)
        div_completo = div_completo.sort_values('Ano', ascending=True)
        
        return div_completo, media_dpa, mediana_dpa, ""
        
    except Exception as e:
        logger.error(f"Erro ao obter DPA para {ticker_symbol}: {str(e)}")
        return pd.DataFrame(), 0.0, 0.0, f"Erro: {str(e)}"


@st.cache_data(ttl=3600)  # Cache de 1 hora
def obter_dados_fundamentalistas(ticker_symbol: str) -> Dict:
    """
    Obtém indicadores fundamentalistas de uma ação via yfinance.
    
    Coleta os principais indicadores para análise fundamentalista:
    - P/L (Preço/Lucro)
    - P/VP (Preço/Valor Patrimonial)
    - ROE (Return on Equity)
    - DY (Dividend Yield)
    - Dívida/PL
    - Dívida/EBITDA
    - EV/EBITDA
    - LPA (Lucro Por Ação)
    - VPA (Valor Patrimonial por Ação)
    
    Args:
        ticker_symbol: Símbolo do ticker (ex: "PETR4.SA")
        
    Returns:
        Dict: Dicionário com indicadores fundamentalistas
        
    Notas:
        - Retorna apenas o ticker em caso de erro
        - Converte Dívida/PL de base 100 para decimal
        - Todos os valores podem ser None se não disponíveis
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Tratamento especial para Dívida/PL (vem em base 100 no campo debtToEquity)
        # Correção v2.1 - Problema A: condição 'is not None' evita falso negativo
        # quando o valor é 0.0 (que antes era tratado como ausente pela condição booleana)
        divida_pl = None
        if info.get('debtToEquity') is not None:
            try:
                divida_pl = float(info.get('debtToEquity')) / 100
            except (TypeError, ValueError):
                divida_pl = None
        
        # Correção v2.1 - Problema B: fallback via balance_sheet para tickers .SA
        # O yfinance tem cobertura incompleta de debtToEquity para ações brasileiras.
        # Quando o campo não vem preenchido no info, tenta calcular diretamente
        # pelo balanço patrimonial: Dívida Total / Patrimônio Líquido
        if divida_pl is None:
            try:
                bs = ticker.balance_sheet
                if bs is not None and not bs.empty:
                    # Nomes possíveis para dívida total no balanço
                    chaves_divida = [
                        'Total Debt', 'Long Term Debt', 'Short Long Term Debt'
                    ]
                    # Nomes possíveis para patrimônio líquido no balanço
                    chaves_pl = [
                        'Stockholders Equity', 'Total Stockholder Equity',
                        'Common Stock Equity'
                    ]
                    
                    chave_divida = next((k for k in chaves_divida if k in bs.index), None)
                    chave_pl = next((k for k in chaves_pl if k in bs.index), None)
                    
                    if chave_divida and chave_pl:
                        divida_valor = bs.loc[chave_divida].iloc[0]
                        pl_valor = bs.loc[chave_pl].iloc[0]
                        
                        if pl_valor and pl_valor != 0:
                            divida_pl = float(divida_valor / pl_valor)
                            logger.info(
                                f"{ticker_symbol}: Dívida/PL calculado via balance_sheet "
                                f"(fallback) = {divida_pl:.4f}"
                            )
            except Exception as e:
                logger.warning(
                    f"{ticker_symbol}: Fallback Dívida/PL também falhou — {str(e)}"
                )
                divida_pl = None
        
        # v4.5: fallback para Dívida/EBITDA — mesmo padrão aplicado ao Dívida/PL
        # O campo 'debtToEbitda' raramente é preenchido pelo yfinance para tickers .SA.
        # Quando ausente, calcula via: Dívida Total / EBITDA
        # onde EBITDA = EBIT (Operating Income) + Depreciação & Amortização
        divida_ebitda = info.get('debtToEbitda')
        
        if divida_ebitda is None:
            try:
                bs  = ticker.balance_sheet
                fin = ticker.financials
                cf  = ticker.cashflow
                
                if (bs is not None and not bs.empty and
                    fin is not None and not fin.empty and
                    cf is not None and not cf.empty):
                    
                    # Dívida Total — mesmo mapeamento usado no fallback de Dívida/PL
                    chaves_divida = ['Total Debt', 'Long Term Debt', 'Short Long Term Debt']
                    chave_divida = next((k for k in chaves_divida if k in bs.index), None)
                    
                    # EBIT — nomes possíveis no financials
                    chaves_ebit = ['EBIT', 'Operating Income']
                    chave_ebit = next((k for k in chaves_ebit if k in fin.index), None)
                    
                    # Depreciação & Amortização — no cashflow
                    chaves_da = [
                        'Depreciation And Amortization',
                        'Depreciation',
                        'Reconciled Depreciation'
                    ]
                    chave_da = next((k for k in chaves_da if k in cf.index), None)
                    
                    if chave_divida and chave_ebit and chave_da:
                        divida_valor = bs.loc[chave_divida].iloc[0]
                        ebit_valor   = fin.loc[chave_ebit].iloc[0]
                        da_valor     = cf.loc[chave_da].iloc[0]
                        
                        ebitda_valor = ebit_valor + abs(da_valor)
                        
                        if ebitda_valor and ebitda_valor != 0:
                            divida_ebitda = float(divida_valor / ebitda_valor)
                            logger.info(
                                f"{ticker_symbol}: Dívida/EBITDA calculado via fallback "
                                f"(balance_sheet + financials + cashflow) = {divida_ebitda:.4f}"
                            )
            except Exception as e:
                logger.warning(
                    f"{ticker_symbol}: Fallback Dívida/EBITDA também falhou — {str(e)}"
                )
                divida_ebitda = None
        
        return {
            'Ticker': ticker_symbol,
            'P/L': info.get('trailingPE'),
            'P/VP': info.get('priceToBook'),
            'ROE': info.get('returnOnEquity'),
            'DY': info.get('dividendYield'),
            'Setor': info.get('sector'),
            'Dívida/PL': divida_pl,
            'Dívida/EBITDA': divida_ebitda,
            'EV/EBITDA': info.get('enterpriseToEbitda'),
            'LPA': info.get('trailingEps'),
            'VPA': info.get('bookValue'),
            # Melhoria 2: incluído para evitar segunda chamada ao yfinance
            # em calcular_payout_ratio — reutiliza o info já coletado aqui
            '_dividendRate': info.get('dividendRate'),
            # Consolidação: preço atual coletado aqui para evitar segunda
            # chamada ao yfinance em obter_preco_atual (Melhoria — consolidação)
            '_precoAtual': float(
                info.get('regularMarketPrice') or info.get('currentPrice') or 0.0
            )
        }
        
    except Exception as e:
        logger.error(
            f"Erro ao obter dados fundamentalistas para {ticker_symbol}: {str(e)}"
        )
        return {'Ticker': ticker_symbol}


@st.cache_data(ttl=3600)  # Cache de 1 hora
def obter_dados_cagr(ticker_symbol: str, years: int = 5) -> Dict:
    """
    Calcula CAGR (Compound Annual Growth Rate) de Receita e Lucro.
    
    O CAGR representa a taxa de crescimento anual composta, indicando
    o crescimento médio anual de um indicador ao longo do período.
    
    Fórmula: CAGR = (Valor_Final / Valor_Inicial) ^ (1 / n_anos) - 1
    
    Args:
        ticker_symbol: Símbolo do ticker (ex: "PETR4.SA")
        years: Número de anos para cálculo (padrão: 5)
        
    Returns:
        Dict: Dicionário com 'CAGR Receita 5a' e 'CAGR Lucro 5a'
        
    Notas:
        - Retorna None para indicadores sem dados suficientes
        - Requer pelo menos 2 anos de dados para cálculo
        - Usa dados financeiros mais recentes disponíveis
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        financials = ticker.financials
        
        # Melhoria 8: log explícito quando não há dados financeiros disponíveis
        if financials is None or financials.empty:
            logger.info(
                f"Sem dados financeiros disponíveis para {ticker_symbol} — "
                f"CAGR Receita e Lucro retornarão None"
            )
            return {'CAGR Receita 5a': None, 'CAGR Lucro 5a': None}
        
        def calcular_cagr(df: pd.DataFrame, chaves_possiveis: List[str]) -> Optional[float]:
            """
            Calcula CAGR para uma métrica específica.
            
            Args:
                df: DataFrame com dados financeiros
                chaves_possiveis: Lista de nomes possíveis da métrica
                
            Returns:
                float ou None: CAGR calculado ou None se impossível calcular
            """
            # Encontra a chave correta no DataFrame
            chave = next((k for k in chaves_possiveis if k in df.index), None)
            if chave is None:
                return None
            
            # Extrai série temporal e limpa dados
            serie = df.loc[chave].sort_index().iloc[-years-1:].dropna()
            
            # Requer pelo menos 2 pontos para cálculo
            if len(serie) < 2:
                return None
            
            # Calcula CAGR
            valor_inicial = serie.iloc[0]
            valor_final = serie.iloc[-1]
            n_periodos = len(serie) - 1
            
            # Evita divisão por zero
            if valor_inicial == 0:
                return None
            
            # Guard: base negativa na potenciação fracionária gera NaN/erro
            # (ex: lucro negativo num dos extremos). Retorna None neste caso.
            if (valor_final / valor_inicial) < 0:
                return None
            
            cagr = float(((valor_final / valor_inicial) ** (1 / n_periodos)) - 1)
            return cagr
        
        # Calcula CAGR para Receita e Lucro
        cagr_receita = calcular_cagr(financials, ['Total Revenue'])
        cagr_lucro = calcular_cagr(financials, ['Net Income'])
        
        return {
            'CAGR Receita 5a': cagr_receita,
            'CAGR Lucro 5a': cagr_lucro
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter CAGR para {ticker_symbol}: {str(e)}")
        return {}


@st.cache_data(ttl=3600)  # Cache de 1 hora — Melhoria 5: consistente com demais funções de coleta
def obter_preco_atual(ticker_symbol: str) -> float:
    """
    Obtém o preço atual de mercado de uma ação.
    
    Tenta obter o preço de diferentes campos disponíveis no yfinance,
    priorizando 'regularMarketPrice' e depois 'currentPrice'.
    
    Args:
        ticker_symbol: Símbolo do ticker (ex: "PETR4.SA")
        
    Returns:
        float: Preço atual ou 0.0 se não disponível
        
    Notas:
        - Retorna 0.0 em caso de erro
        - Tenta múltiplos campos para maior confiabilidade
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Tenta obter preço de diferentes campos
        preco = (
            info.get('regularMarketPrice') or 
            info.get('currentPrice') or 
            0.0
        )
        
        return float(preco)
        
    except Exception as e:
        logger.error(f"Erro ao obter preço para {ticker_symbol}: {str(e)}")
        return 0.0


# ==============================================================================
# FUNÇÕES DE CÁLCULO DE VALUATION
# ==============================================================================

def calcular_preco_teto_bazin(media_dpa: float, taxa_retorno: float) -> float:
    """
    Calcula o Preço Teto usando o Método Bazin.
    
    O Preço Teto representa o valor máximo que um investidor deveria
    pagar por uma ação para obter a taxa de retorno desejada em dividendos.
    
    Fórmula: Preço Teto = Média DPA / Taxa de Retorno
    
    Args:
        media_dpa: Média de Dividendos Por Ação
        taxa_retorno: Taxa de retorno desejada (em decimal, ex: 0.06 para 6%)
        
    Returns:
        float: Preço Teto calculado ou 0.0 se inválido
        
    Exemplo:
        >>> calcular_preco_teto_bazin(1.20, 0.06)
        20.0
    """
    if media_dpa > 0 and taxa_retorno > 0:
        return media_dpa / taxa_retorno
    return 0.0


def calcular_preco_graham(lpa: Optional[float], vpa: Optional[float]) -> float:
    """
    Calcula o Preço Justo usando a Fórmula de Benjamin Graham.
    
    A fórmula de Graham equilibra lucro e patrimônio para encontrar
    o valor intrínseco de uma ação.
    
    Fórmula: Preço = √(22.5 × LPA × VPA)
    
    Onde 22.5 = 15 (P/L máximo) × 1.5 (P/VP máximo)
    
    Args:
        lpa: Lucro Por Ação
        vpa: Valor Patrimonial por Ação
        
    Returns:
        float: Preço Justo de Graham ou 0.0 se inválido
        
    Exemplo:
        >>> calcular_preco_graham(2.0, 10.0)
        21.21
        
    Notas:
        - Requer LPA e VPA positivos
        - Baseado nos princípios de Value Investing
    """
    if lpa and vpa and lpa > 0 and vpa > 0:
        return float(np.sqrt(MULTIPLICADOR_GRAHAM * lpa * vpa))
    return 0.0


def calcular_margem_seguranca(
    preco_teto: float, 
    preco_atual: float
) -> float:
    """
    Calcula a Margem de Segurança em relação ao Preço Teto.
    
    Margem positiva indica que o preço está abaixo do teto (oportunidade).
    Margem negativa indica que o preço está acima do teto (caro).
    
    Fórmula: MS = ((Preço Teto / Preço Atual) - 1) × 100
    
    Args:
        preco_teto: Preço Teto calculado
        preco_atual: Preço atual de mercado
        
    Returns:
        float: Margem de Segurança em percentual ou -100 se inválido
        
    Exemplo:
        >>> calcular_margem_seguranca(20.0, 15.0)
        33.33
    """
    if preco_atual > 0 and preco_teto > 0:
        return ((preco_teto / preco_atual) - 1) * 100
    return -100.0


# ==============================================================================
# FUNÇÃO DE PROCESSAMENTO PARALELO
# ==============================================================================

def calcular_cagr_dpa(df_dpa: pd.DataFrame, anos: int) -> float:
    """
    Calcula a Taxa de Crescimento Anual Composta (CAGR) do DPA.
    
    Args:
        df_dpa: DataFrame com histórico de DPA
        anos: Número de anos para análise
        
    Returns:
        float: CAGR em percentual (ex: 8.5 para 8.5% ao ano)
        
    Notas:
        - Retorna 0.0 se não houver dados suficientes
        - Usa apenas anos completos (exclui ano atual)
    """
    try:
        if df_dpa.empty or len(df_dpa) < 2:
            return 0.0
        
        # Usa helper global — evita repetir pytz.timezone() localmente
        agora = _agora_sp()
        
        # Filtrar apenas anos completos
        df_completo = df_dpa[df_dpa['Ano'] < agora.year].copy()
        
        if len(df_completo) < 2:
            return 0.0
        
        # Pegar últimos N anos
        df_completo = df_completo.sort_values('Ano', ascending=False).head(anos)
        
        if len(df_completo) < 2:
            return 0.0
        
        df_completo = df_completo.sort_values('Ano', ascending=True)
        
        valor_inicial = df_completo.iloc[0]['DPA']
        valor_final = df_completo.iloc[-1]['DPA']
        
        if valor_inicial <= 0 or valor_final <= 0:
            return 0.0
        
        num_anos = len(df_completo) - 1
        cagr = ((valor_final / valor_inicial) ** (1 / num_anos) - 1) * 100
        
        return round(cagr, 2)
        
    except Exception as e:
        logger.error(f"Erro ao calcular CAGR DPA: {str(e)}")
        return 0.0


def calcular_payout_ratio(dados_info: Dict) -> float:
    """
    Calcula o Payout Ratio (percentual do lucro distribuído como dividendos).
    
    Args:
        dados_info: Dicionário já coletado por obter_dados_fundamentalistas()
                    Reutiliza '_dividendRate' e 'LPA' para evitar nova chamada
                    ao yfinance (Melhoria 2 — elimina requisição duplicada)
        
    Returns:
        float: Payout Ratio em percentual (ex: 58.5 para 58.5%)
        
    Notas:
        - Retorna 0.0 se não houver dados
        - Payout > 100% indica dividendos insustentáveis
    """
    try:
        # Melhoria 2: usa dados já coletados — sem nova requisição ao yfinance
        dividendos_anuais = dados_info.get('_dividendRate') or 0
        lucro_por_acao = dados_info.get('LPA') or 0
        
        if lucro_por_acao <= 0 or dividendos_anuais <= 0:
            return 0.0
        
        payout = (dividendos_anuais / lucro_por_acao) * 100
        
        return round(payout, 2)
        
    except Exception as e:
        logger.error(f"Erro ao calcular Payout Ratio: {str(e)}")
        return 0.0


def calcular_anos_consecutivos(df_dpa: pd.DataFrame) -> int:
    """
    Calcula quantos anos consecutivos a empresa pagou dividendos.
    
    Args:
        df_dpa: DataFrame com histórico de DPA
        
    Returns:
        int: Número de anos consecutivos com DPA > 0
        
    Notas:
        - Conta a partir do ano mais recente retroativamente
        - Para se houver um ano sem dividendos
    """
    try:
        if df_dpa.empty:
            return 0
        
        # Usa helper global — evita repetir pytz.timezone() localmente
        agora = _agora_sp()
        
        # Filtrar apenas anos completos
        df_completo = df_dpa[df_dpa['Ano'] < agora.year].copy()
        
        if df_completo.empty:
            return 0
        
        # Ordenar do mais recente para o mais antigo
        df_completo = df_completo.sort_values('Ano', ascending=False)
        
        anos_consecutivos = 0
        
        for _, row in df_completo.iterrows():
            if row['DPA'] > 0:
                anos_consecutivos += 1
            else:
                break  # Para na primeira quebra de sequência
        
        return anos_consecutivos
        
    except Exception as e:
        logger.error(f"Erro ao calcular anos consecutivos: {str(e)}")
        return 0


def processar_ticker(
    ticker: str, 
    anos: int, 
    taxa_retorno: float
) -> Dict:
    """
    Processa todos os dados e cálculos para um único ticker.
    
    Esta função centraliza todo o processamento de um ticker:
    1. Coleta de dados DPA
    2. Coleta de dados fundamentalistas
    3. Cálculo de CAGR
    4. Análise de sazonalidade
    5. Cálculo de preço atual
    6. Cálculo de Preço Teto (Bazin)
    7. Cálculo de Preço Justo (Graham)
    8. Cálculo de Margem de Segurança
    9. Geração de alertas de oportunidade
    
    Args:
        ticker: Símbolo do ticker
        anos: Número de anos para análise
        taxa_retorno: Taxa de retorno desejada (em decimal)
        
    Returns:
        Dict: Dicionário com todas as análises organizadas em seções:
            - 'v': Dados de valuation
            - 'i': Indicadores fundamentalistas
            - 's': Estatísticas de DPA
            - 'g': DataFrame para gráfico
            - 'a': Alerta de oportunidade (se houver)
            
    Notas:
        - Função otimizada para execução paralela
        - Tratamento robusto de erros em cada etapa
        - Retorna estrutura consistente mesmo em caso de erros parciais
    """
    try:
        # Coleta de dados
        df_dpa, media_dpa, mediana_dpa, _ = obter_dados_dpa(ticker, anos)
        dados_fundamentalistas = obter_dados_fundamentalistas(ticker)
        dados_fundamentalistas.update(obter_dados_cagr(ticker))
        
        meses_pagamento, meses_baratos = obter_sazonalidade_e_dividendos(ticker, anos)
        # Reutiliza preço já coletado em obter_dados_fundamentalistas
        # para evitar segunda chamada ao yfinance (consolidação)
        preco_atual = dados_fundamentalistas.get('_precoAtual') or obter_preco_atual(ticker)
        
        # Cálculos de valuation
        preco_teto = calcular_preco_teto_bazin(media_dpa, taxa_retorno)
        
        lpa = dados_fundamentalistas.get('LPA')
        vpa = dados_fundamentalistas.get('VPA')
        preco_graham = calcular_preco_graham(lpa, vpa)
        
        # Preço Teto baseado na mediana — usado para a Margem de Segurança
        # conforme documentação: mais conservador, resistente a dividendos extraordinários
        preco_teto_mediana = calcular_preco_teto_bazin(mediana_dpa, taxa_retorno)
        margem_seguranca = calcular_margem_seguranca(preco_teto_mediana, preco_atual)
        
        # Novas métricas adicionais (não afetam cálculos existentes)
        cagr_dpa = calcular_cagr_dpa(df_dpa, anos)
        # Melhoria 2: passa o dict já coletado — sem segunda chamada ao yfinance
        payout_ratio = calcular_payout_ratio(dados_fundamentalistas)
        anos_consecutivos = calcular_anos_consecutivos(df_dpa)
        
        # Geração de alerta de oportunidade
        alerta = None
        if margem_seguranca > 0:
            alerta = (
                f"Oportunidade em {ticker}: "
                f"Preço Atual R$ {preco_atual:.2f} < "
                f"Teto R$ {preco_teto:.2f}"
            )
        
        # Retorna dados estruturados
        return {
            'v': {
                'Ticker': ticker,
                'Preço Atual': preco_atual,
                'Preço Teto (Bazin)': preco_teto,
                'Preço Graham': preco_graham,
                'Margem Segurança (%)': margem_seguranca,
                'Pagamento Dividendos': meses_pagamento,
                'Melhor mês para compra': meses_baratos
            },
            'i': dados_fundamentalistas,
            'i_extras': {
                'Ticker': ticker,
                'Payout Ratio (%)': payout_ratio
            },
            's': {
                'Ticker': ticker,
                'Preço Atual': preco_atual,
                'Média DPA': media_dpa,
                'Mediana DPA': mediana_dpa,
                'CAGR DPA 5a (%)': cagr_dpa,
                'Anos Consecutivos': anos_consecutivos
            },
            'g': df_dpa,
            'a': alerta
        }
        
    except Exception as e:
        logger.error(f"Erro ao processar ticker {ticker}: {str(e)}")
        # Retorna estrutura mínima em caso de erro
        return {
            'v': {'Ticker': ticker, 'Preço Atual': 0.0, 'Preço Teto (Bazin)': 0.0, 
                  'Preço Graham': 0.0, 'Margem Segurança (%)': -100.0,
                  'Pagamento Dividendos': 'Erro', 'Melhor mês para compra': 'Erro'},
            'i': {'Ticker': ticker},
            'i_extras': {'Ticker': ticker, 'Payout Ratio (%)': 0.0},
            's': {'Ticker': ticker, 'Preço Atual': 0.0, 'Média DPA': 0.0, 'Mediana DPA': 0.0,
                  'CAGR DPA 5a (%)': 0.0, 'Anos Consecutivos': 0},
            'g': pd.DataFrame(),
            'a': None
        }


# ==============================================================================
# INTERFACE DO USUÁRIO - SIDEBAR
# ==============================================================================

with st.sidebar:
    st.header("Opções de Análise")
    
    # Entrada manual de tickers
    st.subheader("Entrada Manual")
    ticker_input = st.text_input(
        "insira Tickers (separados por vírgula)",
        value=TICKERS_EXEMPLO,
        help="Digite os tickers separados por vírgula (ex: PETR4.SA, VALE3.SA)"
    )
    
    st.divider()
    
    # Upload de arquivo CSV
    st.subheader("Upload de Arquivo CSV")
    uploaded_file = st.file_uploader(
        "Escolha o arquivo",
        type=['csv'],
        label_visibility="collapsed",
        help="O arquivo CSV deve conter uma coluna chamada 'Ticker'"
    )
    
    if uploaded_file:
        try:
            df_csv = pd.read_csv(uploaded_file)
            
            if 'Ticker' in df_csv.columns:
                ticker_input = ",".join(df_csv['Ticker'].astype(str).tolist())
                st.success(f"✅ {len(df_csv)} tickers carregados com sucesso!")
            else:
                st.error("❌ Erro: O arquivo CSV deve conter uma coluna 'Ticker'")
                
        except pd.errors.EmptyDataError:
            st.error("❌ Erro: O arquivo CSV está vazio")
        except pd.errors.ParserError:
            st.error("❌ Erro: Não foi possível ler o arquivo CSV. Verifique o formato.")
        except Exception as e:
            st.error(f"❌ Erro ao processar arquivo: {str(e)}")
    
    st.divider()
    
    # Parâmetros de análise
    anos_input = st.number_input(
        "Número de Anos para Análise",
        value=ANOS_PADRAO,
        min_value=1,
        max_value=20,
        help="Período histórico para cálculo de médias e tendências"
    )
    
    taxa_retorno_input = st.number_input(
        "Taxa de Retorno Anual Desejada (%)",
        value=TAXA_RETORNO_PADRAO,
        min_value=0.1,
        max_value=100.0,
        step=0.5,
        help="Taxa de retorno desejada para cálculo do Preço Teto"
    )
    
    st.divider()
    
    # Filtros (para versões futuras - mantidos para compatibilidade)
    max_pl_filter = st.number_input(
        "P/L Máximo (0 para Desativar)",
        value=0.0,
        help="Filtro de P/L máximo aceitável"
    )
    
    min_dy_filter = st.number_input(
        "DY Mínimo (%) (0 para Desativar)",
        value=0.0,
        help="Filtro de Dividend Yield mínimo"
    )
    
    min_ms_filter = st.number_input(
        "Margem de Segurança Mínima (%)",
        value=-100.0,
        help="Filtro de Margem de Segurança mínima"
    )

# ==============================================================================
# INTERFACE PRINCIPAL
# ==============================================================================

st.title("Análise de Ações e Indicadores Fundamentalistas")

if st.button("Gerar Relatório", type="primary"):
    # Validação de inputs
    if not validar_anos(anos_input):
        st.error("❌ Número de anos inválido. Use valores entre 1 e 20.")
        st.stop()
    
    if not validar_taxa_retorno(taxa_retorno_input):
        st.error("❌ Taxa de retorno inválida. Use valores entre 0.1% e 100%.")
        st.stop()
    
    # Sanitização e validação de tickers
    tickers = sanitizar_tickers(ticker_input)
    
    if not tickers:
        st.error("❌ Nenhum ticker válido foi inserido. Verifique o formato.")
        st.stop()
    
    # Inicialização de variáveis
    data_valuation = []
    data_indicadores = []
    data_indicadores_extras = []
    data_estatisticas = []
    data_grafico = []
    alertas = []
    taxa_retorno_decimal = taxa_retorno_input / 100
    
    # Processamento paralelo com feedback visual
    with st.spinner(f"🔄 Processando {len(tickers)} ticker(s)..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Melhoria 13: workers dinâmicos — evita criar threads ociosas com poucos tickers
        workers_dinamicos = min(len(tickers), MAX_WORKERS_THREAD_POOL)
        with ThreadPoolExecutor(max_workers=workers_dinamicos) as executor:
            # Submete todas as tarefas
            futures = {
                executor.submit(processar_ticker, t, anos_input, taxa_retorno_decimal): t 
                for t in tickers
            }
            
            # Processa resultados conforme completam
            # timeout=60s por ticker — evita threads penduradas indefinidamente
            total = len(futures)
            for idx, future in enumerate(as_completed(futures, timeout=60), 1):
                ticker = futures[future]
                
                try:
                    resultado = future.result()
                    
                    # Organiza resultados
                    data_valuation.append(resultado['v'])
                    data_indicadores.append(resultado['i'])
                    data_indicadores_extras.append(resultado['i_extras'])
                    data_estatisticas.append(resultado['s'])
                    
                    if not resultado['g'].empty:
                        data_grafico.append(resultado['g'])
                    
                    if resultado['a']:
                        alertas.append(resultado['a'])
                    
                    # Atualiza progresso
                    progress_bar.progress(idx / total)
                    status_text.text(f"✅ Processado: {ticker} ({idx}/{total})")
                    
                except Exception as e:
                    logger.error(f"Erro no future para {ticker}: {str(e)}")
                    status_text.text(f"⚠️ Erro em: {ticker} ({idx}/{total})")
        
        progress_bar.empty()
        status_text.empty()

    # Salva resultados no session_state para preservar entre reruns
    # (evita que interações na tab4 disparem novo processamento)
    if data_valuation:
        st.session_state['relatorio'] = {
            'data_valuation':        data_valuation,
            'data_indicadores':      data_indicadores,
            'data_indicadores_extras': data_indicadores_extras,
            'data_estatisticas':     data_estatisticas,
            'data_grafico':          data_grafico,
            'alertas':               alertas,
            'anos_input':            anos_input,
            'taxa_retorno_input':    taxa_retorno_input,
            'taxa_retorno_decimal':  taxa_retorno_decimal,
        }

# Exibição de resultados — lê do session_state, não de variáveis locais
# Dessa forma, interações na tab4 (number_input de quantidade) não perdem os dados
if st.session_state.get('relatorio'):
    rel                      = st.session_state['relatorio']
    data_valuation           = rel['data_valuation']
    data_indicadores         = rel['data_indicadores']
    data_indicadores_extras  = rel['data_indicadores_extras']
    data_estatisticas        = rel['data_estatisticas']
    data_grafico             = rel['data_grafico']
    alertas                  = rel['alertas']
    anos_input               = rel['anos_input']
    taxa_retorno_input       = rel['taxa_retorno_input']
    taxa_retorno_decimal     = rel['taxa_retorno_decimal']

    # Exibição de resultados
    if data_valuation:
        st.success(f"✅ Análise concluída! {len(data_valuation)} ticker(s) processado(s).")
        
        # Melhoria 9: identifica tickers que retornaram sem dados (preço = 0 e teto = 0)
        tickers_sem_dados = [
            r['Ticker'] for r in data_valuation
            if r.get('Preço Atual', 0) == 0.0 and r.get('Preço Teto (Bazin)', 0) == 0.0
        ]
        if tickers_sem_dados:
            st.warning(
                f"⚠️ Os seguintes tickers não retornaram dados — verifique se os símbolos "
                f"estão corretos (ex: PETR4.SA): **{', '.join(tickers_sem_dados)}**"
            )
        
        # Criação de DataFrames
        df_valuation = pd.DataFrame(data_valuation)
        
        # Melhoria 10: ordena por Margem de Segurança decrescente — melhores
        # oportunidades aparecem primeiro. A ordenação é feita aqui, uma única vez,
        # antes de qualquer uso do DataFrame, garantindo consistência em todas as tabs.
        df_valuation = df_valuation.sort_values(
            'Margem Segurança (%)', ascending=False
        ).reset_index(drop=True)
        
        # Ordenação alfabética por Ticker nos 3 módulos (Valuation, Indicadores, Gráfico DPA)
        df_valuation = df_valuation.sort_values('Ticker', ascending=True).reset_index(drop=True)
        
        df_indicadores = pd.DataFrame(data_indicadores)
        
        # Melhoria 11: remove LPA e VPA do df_indicadores pois são colunas internas
        # usadas apenas para calcular o Preço Graham em processar_ticker().
        # Melhoria 2: remove também '_dividendRate', usado apenas internamente
        # para calcular Payout Ratio sem nova chamada ao yfinance.
        # Nenhuma dessas colunas deve aparecer na tabela exibida ao usuário.
        colunas_remover = [col for col in ['LPA', 'VPA', '_dividendRate', '_precoAtual'] if col in df_indicadores.columns]
        if colunas_remover:
            df_indicadores = df_indicadores.drop(columns=colunas_remover)
        
        # Adicionar novas métricas aos indicadores (Payout Ratio)
        if data_indicadores_extras:
            df_indicadores_extras = pd.DataFrame(data_indicadores_extras)
            df_indicadores = pd.merge(
                df_indicadores, 
                df_indicadores_extras, 
                on='Ticker', 
                how='left'
            )
        
        # Alerta sobre instituições financeiras
        # Guard: coluna 'Setor' pode estar ausente se nenhum ticker retornou
        # o campo 'sector' do yfinance (ex: falha de API ou ticker inválido)
        tem_setor_financeiro = (
            'Setor' in df_indicadores.columns and
            any(df_indicadores['Setor'].str.contains('Financial', case=False, na=False))
        )
        
        if tem_setor_financeiro:
            st.info(
                "ℹ️ **Nota sobre Instituições Financeiras:** "
                "Identificámos bancos ou seguradoras na sua análise. "
                "Para estas empresas, o indicador **Dívida/EBITDA** não é aplicável, "
                "pois o modelo de negócio baseia-se na intermediação financeira. "
                "Foque em métricas como ROE e P/VP."
            )

        # Tabs de visualização
        tab1, tab2, tab3, tab4 = st.tabs([
            "💎 Valuation",
            "📊 Indicadores",
            "📈 Gráfico DPA",
            "💰 Projeção de Ganhos"
        ])
        
        # Tab 1: Valuation
        with tab1:
            st.dataframe(
                df_valuation.style.map(
                    lambda x: 'background-color: #e6ffed;' 
                    if isinstance(x, (int, float)) and x > 0 
                    else '',
                    subset=['Margem Segurança (%)']
                ).format({
                    'Preço Atual': 'R$ {:.2f}',
                    'Preço Teto (Bazin)': 'R$ {:.2f}',
                    'Preço Graham': 'R$ {:.2f}',
                    'Margem Segurança (%)': '{:.2f}%'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # v4.2: Cards de oportunidades movidos para dentro da aba Valuation
            # Exibidos após a tabela, antes das explicações
            if alertas:
                st.markdown("---")
                st.markdown("### 🔔 Oportunidades Identificadas")
                
                oportunidades = [
                    r for r in data_valuation
                    if r.get("Margem Segurança (%)", -100) > 0
                ]
                
                if oportunidades:
                    cols_por_linha = 3
                    for i in range(0, len(oportunidades), cols_por_linha):
                        grupo = oportunidades[i:i + cols_por_linha]
                        colunas = st.columns(len(grupo))
                        
                        for col, op in zip(colunas, grupo):
                            ticker_op   = op.get("Ticker", "")
                            preco_atual = op.get("Preço Atual", 0.0)
                            preco_teto  = op.get("Preço Teto (Bazin)", 0.0)
                            margem      = op.get("Margem Segurança (%)", 0.0)
                            
                            if margem >= 20:
                                cor_borda = "#28a745"
                                cor_fundo = "#f0fff4"
                                cor_badge = "#28a745"
                                icone     = "🟢"
                            elif margem >= 10:
                                cor_borda = "#fd7e14"
                                cor_fundo = "#fff8f0"
                                cor_badge = "#fd7e14"
                                icone     = "🟡"
                            else:
                                cor_borda = "#17a2b8"
                                cor_fundo = "#f0faff"
                                cor_badge = "#17a2b8"
                                icone     = "🔵"
                            
                            with col:
                                html_card = (
                                    "<div style='"
                                    "background-color:" + cor_fundo + ";"
                                    "border:2px solid " + cor_borda + ";"
                                    "border-radius:14px;"
                                    "padding:20px 18px;"
                                    "text-align:center;"
                                    "box-shadow:0 4px 12px rgba(0,0,0,0.08);"
                                    "margin-bottom:12px;'>"
                                    "<div style='font-size:28px;margin-bottom:4px;'>" + icone + "</div>"
                                    "<div style='font-size:22px;font-weight:800;color:" + cor_borda + ";letter-spacing:1px;margin-bottom:10px;'>" + ticker_op + "</div>"
                                    "<div style='display:inline-block;background-color:" + cor_badge + ";color:white;border-radius:20px;padding:3px 14px;font-size:13px;font-weight:700;margin-bottom:14px;'>+"
                                    + f"{margem:.1f}" + "% de margem</div>"
                                    "<div style='margin:6px 0;font-size:14px;color:#555;'>"
                                    "<span style='font-weight:600;'>Preço Atual:</span>"
                                    "<span style='font-size:16px;font-weight:700;color:#222;'> R$ " + f"{preco_atual:.2f}" + "</span></div>"
                                    "<div style='margin:6px 0;font-size:14px;color:#555;'>"
                                    "<span style='font-weight:600;'>Preço Teto:</span>"
                                    "<span style='font-size:16px;font-weight:700;color:" + cor_borda + ";'> R$ " + f"{preco_teto:.2f}" + "</span></div>"
                                    "</div>"
                                )
                                st.markdown(html_card, unsafe_allow_html=True)
            
            # Explicações dos indicadores de Valuation
            st.markdown('<div class="explicacao-container">', unsafe_allow_html=True)
            st.subheader("O que significam os indicadores e como são calculados?")
            
            st.write(
                f"**Preço Teto (Média):** "
                f"É o preço máximo que um investidor aceitaria pagar por uma ação para "
                f"garantir uma rentabilidade mínima desejada em dividendos (baseado na "
                f"média histórica do DPA). Por exemplo, se uma ação paga R$ 1,20 de média "
                f"e você deseja 6% de retorno, seu teto é R$ 20,00. Se pagar mais que isso, "
                f"o seu rendimento real será menor que o desejado."
            )
            st.latex(
                r"Preço\ Teto = \frac{Média\ DPA\ (" + str(anos_input) + 
                r"\ anos)}{Taxa\ de\ Retorno\ Desejada}"
            )
            
            st.write(
                f"**Preço Teto (Mediana):** "
                f"Este cálculo utiliza o valor central dos dividendos pagos no período "
                f"escolhido. É uma métrica de segurança adicional para evitar distorções "
                f"causadas por anos onde a empresa pagou dividendos extraordinários "
                f"(não-recorrentes) que elevam a média artificialmente. A mediana ignora "
                f"estes extremos."
            )
            st.latex(
                r"Preço\ Teto\ (Mediana) = \frac{Mediana\ DPA\ (" + 
                str(anos_input) + r"\ anos)}{Taxa\ de\ Retorno\ Desejada}"
            )
            
            st.write(
                "**Preço Justo de Graham:** "
                "É uma fórmula desenvolvida por Benjamin Graham (mentor de Warren Buffett) "
                "para encontrar o valor intrínseco de uma ação equilibrando lucro e "
                "patrimônio. A fórmula assume que um investidor não deve pagar mais do que "
                "15 vezes o lucro (P/L) e 1.5 vezes o valor patrimonial (P/VP), resultando "
                "no multiplicador de 22.5."
            )
            st.latex(
                r"Preço\ Justo\ (Graham) = \sqrt{22.5 \times LPA \times VPA}"
            )
            
            st.write(
                f"**Margem de Segurança:** "
                f"Indica o quanto o Preço Teto está acima do preço atual de mercado. "
                f"Uma margem positiva sugere que a ação está sendo negociada com 'desconto' "
                f"em relação à sua capacidade de pagar dividendos. Ex: Se o teto é R$ 20 e "
                f"o preço é R$ 15, a sua margem é de 33,3%."
            )
            st.latex(
                r"MS = \left( \frac{Preço\ Teto}{Preço\ Atual} - 1 \right) \times 100"
            )
            
            st.write(
                f"**Pagamento Dividendos/JCP (meses consistentes):** "
                f"Identifica os meses em que a empresa pagou dividendos ou JCP de forma "
                f"consistente ao longo dos últimos {anos_input} anos analisados. "
                f"Apenas os meses que apresentaram pagamento em pelo menos 60% dos anos "
                f"do período são exibidos — ou seja, um mês que falhou uma ou duas vezes "
                f"num histórico longo ainda pode aparecer, desde que seja recorrente na "
                f"maioria dos anos."
            )
            st.write(
                f"**Como é calculado:** Para cada mês (Janeiro a Dezembro), o sistema "
                f"conta em quantos anos distintos do período houve pelo menos um pagamento "
                f"naquele mês. Depois calcula a frequência (anos com pagamento ÷ total de "
                f"anos analisados). Meses com frequência ≥ 60% são considerados consistentes "
                f"e exibidos. Esse método é imune à frequência de pagamento da empresa: "
                f"uma empresa que paga mensalmente e outra que paga trimestralmente são "
                f"avaliadas pelo mesmo critério — presença ou ausência de pagamento no mês, "
                f"dentro de cada ano."
            )
            st.latex(
                r"Frequência\ do\ Mês\ (\%) = \frac{Anos\ com\ pagamento\ no\ mês}{Total\ de\ anos\ analisados} \times 100"
            )
            st.write(
                f"**Como interpretar:** Se aparecem os meses Mar, Jun, Set e Dez, significa "
                f"que a empresa pagou dividendos/JCP nesses meses em pelo menos "
                f"{int(anos_input * 0.6)} dos {anos_input} anos analisados — um padrão "
                f"confiável de distribuição. Se o campo exibe 'Sem Div.', a empresa não "
                f"possui histórico consistente de pagamentos no período selecionado. "
                f"Importante: aumentar o número de anos na barra lateral pode revelar ou "
                f"excluir meses conforme o histórico mais longo seja considerado."
            )

            st.write(
                f"**Sazonalidade (Melhor mês para compra):** "
                f"O app analisa o histórico de preços dos últimos {anos_input} anos "
                f"e identifica os 3 meses que, recorrentemente, apresentam preços "
                f"abaixo da média anual da própria ação. O percentual exibido ao lado "
                f"de cada mês (ex: Mar -8,2%) indica o quanto, em média, o preço "
                f"ficou abaixo da média daquele ano — quanto mais negativo, maior o "
                f"desconto histórico típico nesse mês."
            )
            st.write(
                f"**Por que esse método é mais confiável?** "
                f"A abordagem anterior comparava preços absolutos entre meses, o que "
                f"é estatisticamente incorreto: meses antigos têm preços nominais "
                f"menores simplesmente porque a ação valia menos naquele tempo, não "
                f"porque o mês seja sazonalmente mais barato. O método atual calcula, "
                f"para cada ano, o desvio percentual de cada mês em relação à média "
                f"anual daquele ano, e depois agrega esses desvios ao longo de todos "
                f"os {anos_input} anos analisados. Isso revela padrões sazonais reais, "
                f"independentes do nível de preço absoluto da ação."
            )
            st.latex(
                r"Desvio\ Mensal\ (\%) = \left( \frac{Preço\ Médio\ do\ Mês}{Média\ Anual\ do\ Ano} - 1 \right) \times 100"
            )
            st.write(
                f"**Como interpretar:** Um desvio de -8,2% em Março significa que, "
                f"nos últimos {anos_input} anos, o preço da ação em Março ficou, em "
                f"média, 8,2% abaixo da média anual. Isso é um padrão sazonal "
                f"recorrente — não uma garantia, mas uma tendência histórica que pode "
                f"auxiliar na identificação de janelas de oportunidade para compra. "
                f"Quanto maior o número de anos analisados, mais robusto e confiável "
                f"é esse padrão."
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 2: Indicadores Fundamentalistas
        with tab2:
            colunas_indicadores = [
                'Ticker', 'P/L', 'P/VP', 'ROE', 'DY',
                'Payout Ratio (%)',
                'Dívida/PL', 'Dívida/EBITDA', 'EV/EBITDA',
                'CAGR Receita 5a', 'CAGR Lucro 5a'
                # Melhoria 12: 'Setor' removido da exibição — está em inglês (vem do yfinance)
                # e é usado apenas internamente para detectar instituições financeiras.
                # O df_indicadores ainda contém a coluna para essa lógica funcionar.
            ]
            
            # CORREÇÃO 3: Filtra a lista de colunas para incluir apenas as que
            # existem no DataFrame, evitando KeyError quando obter_dados_cagr()
            # falha e não retorna 'CAGR Receita 5a' / 'CAGR Lucro 5a'
            colunas_existentes = [
                col for col in colunas_indicadores if col in df_indicadores.columns
            ]
            
            df_indicadores_filtrado = df_indicadores[
                df_indicadores['Ticker'].isin(df_valuation['Ticker'])
            ][colunas_existentes].sort_values('Ticker', ascending=True).reset_index(drop=True)
            
            st.dataframe(
                df_indicadores_filtrado,
                use_container_width=True,
                hide_index=True
            )
            
            # Explicações dos indicadores fundamentalistas
            st.markdown('<div class="explicacao-container">', unsafe_allow_html=True)
            st.subheader("O que significam os indicadores e como são calculados?")
            
            st.write(
                f"**CAGR (Compound Annual Growth Rate) da Receita/Lucro:** "
                f"É a taxa de crescimento anual composta de um indicador (Receita ou Lucro) "
                f"durante um período específico (neste caso, 5 anos). Indica se a empresa "
                f"tem mantido um crescimento consistente. Valores altos e positivos mostram "
                f"que a empresa está a expandir as suas operações e resultados de forma "
                f"saudável."
            )
            st.latex(
                r"CAGR = \left( \frac{Valor\ Final}{Valor\ Inicial} \right)^{\frac{1}{n}} - 1"
            )
            
            st.write(
                "**P/L (Preço/Lucro):** "
                "Indica quantos anos de lucro a empresa levaria para gerar o seu valor de "
                "mercado atual. Um P/L mais baixo pode sugerir que a ação está subvalorizada. "
                "Exemplo: Um P/L de 5 indica que em 5 anos o lucro acumulado equivaleria ao "
                "preço pago pela ação (payback de 5 anos)."
            )
            
            st.write(
                "**P/VP (Preço/Valor Patrimonial):** "
                "Compara o preço de mercado da ação com o valor contábil dos ativos da "
                "empresa por ação. Um P/VP menor que 1 pode indicar que a ação está sendo "
                "vendida por menos do que valem os seus ativos líquidos. Se for 0,5, está a "
                "comprar R$ 1,00 de patrimônio por R$ 0,50."
            )
            
            st.write(
                "**ROE (Return on Equity):** "
                "Mostra o quanto a empresa consegue gerar de lucro para cada R$ 1 de "
                "patrimônio líquido. Um valor mais alto indica uma gestão mais eficiente. "
                "Ex: Um ROE de 20% significa que a empresa gerou R$ 20 de lucro para cada "
                "R$ 100 de capital próprio investido."
            )
            
            st.write(
                "**DY (Dividend Yield):** "
                "É o rendimento de dividendos de uma ação, ou seja, o percentual de retorno "
                "que recebe em dividendos em relação ao preço atual da ação. É o indicador "
                "preferido de quem procura renda passiva recorrente."
            )
            
            st.write(
                "**Payout Ratio (Taxa de Pagamento):** "
                "Percentual do lucro que a empresa distribui como dividendos aos acionistas. "
                "Um Payout de 50-70% é considerado equilibrado: a empresa paga bons dividendos "
                "e ainda retém capital para crescer. Valores acima de 90% podem indicar "
                "dividendos insustentáveis (empresa paga quase todo o lucro). Valores acima "
                "de 100% são críticos - a empresa está pagando mais dividendos do que lucra."
            )
            st.latex(
                r"Payout\ Ratio = \frac{Dividendos\ Pagos}{Lucro\ Líquido} \times 100"
            )
            
            st.write(
                "**Dívida/PL (Dívida/Patrimônio Líquido):** "
                "Avalia o nível de endividamento da empresa. Valores menores que 1 indicam "
                "uma situação confortável, onde a dívida é menor que o patrimônio da "
                "companhia. Uma dívida de 0,5 significa que a empresa deve R$ 0,50 para "
                "cada R$ 1,00 que possui."
            )
            
            st.info(
                "ℹ️ **Sobre as colunas Dívida/PL e Dívida/EBITDA:** "
                "Para ações brasileiras (.SA), estas informações nem sempre estão disponíveis "
                "diretamente na fonte de dados (yfinance). Quando isso ocorre, o sistema tenta "
                "calcular automaticamente os valores a partir dos demonstrativos financeiros da empresa: "
                "Dívida/PL via balanço patrimonial (Dívida Total ÷ Patrimônio Líquido) e "
                "Dívida/EBITDA via balanço + demonstrativo de resultados + fluxo de caixa "
                "(Dívida Total ÷ EBITDA, onde EBITDA = EBIT + Depreciação & Amortização). "
                "Caso alguma coluna apareça vazia para um ticker, significa que os dados necessários "
                "não estavam disponíveis em nenhuma das fontes no momento da consulta. "
                "Isso é uma limitação da fonte de dados, não um erro do sistema."
            )

            st.write(
                "**Dívida/EBITDA:** "
                "Indica quantos anos de geração de caixa operacional (EBITDA) seriam "
                "necessários para a empresa quitar a sua dívida total. Valores abaixo de "
                "2.0x são considerados muito seguros. Se o valor for negativo, significa "
                "que a empresa possui mais dinheiro em caixa do que dívidas (Caixa Líquido), "
                "oferecendo máxima segurança."
            )
            
            st.write(
                "**EV/EBITDA (Enterprise Value / EBITDA):** "
                "Relação entre o Valor da Empresa (incluindo dívida) e a sua geração de "
                "caixa operacional. É como o 'P/L do negócio inteiro'. Valores abaixo de 10 "
                "são frequentemente vistos como positivos, indicando uma empresa "
                "potencialmente subvalorizada."
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 3: Gráfico DPA
        with tab3:
            if data_grafico:
                # Criação do gráfico
                fig = go.Figure()
                
                # Ordena as barras do gráfico alfabeticamente por ticker
                data_grafico_sorted = sorted(
                    data_grafico,
                    key=lambda d: d['Ticker'].iloc[0]
                )
                
                for dados in data_grafico_sorted:
                    fig.add_trace(go.Bar(
                        x=dados['Ano'],
                        y=dados['DPA'],
                        name=dados['Ticker'].iloc[0]
                    ))
                
                fig.update_layout(
                    title="Evolução do DPA ao Longo dos Anos",
                    xaxis_title="Ano",
                    yaxis_title="DPA (R$)",
                    barmode='group',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabela de estatísticas — ordenada alfabeticamente por Ticker
                df_estatisticas = pd.DataFrame(data_estatisticas).sort_values(
                    'Ticker', ascending=True
                ).reset_index(drop=True)
                # Melhoria 3: cálculo feito via divisão vetorial direta (linha a linha).
                # O padrão anterior usava if/else avaliado uma única vez para toda a
                # coluna — funcionava por acidente pois taxa_retorno_decimal sempre > 0
                # após validação. Agora o padrão é explícito e correto.
                # Os valores resultantes são idênticos ao comportamento anterior.
                if taxa_retorno_decimal > 0:
                    df_estatisticas['Preço Teto (Mediana)'] = (
                        df_estatisticas['Mediana DPA'] / taxa_retorno_decimal
                    )
                else:
                    df_estatisticas['Preço Teto (Mediana)'] = 0
                
                # Calcula Margem de Segurança baseada no Preço Teto (Mediana)
                df_estatisticas['Margem de Segurança Mediana (%)'] = df_estatisticas.apply(
                    lambda row: ((row['Preço Teto (Mediana)'] / row['Preço Atual']) - 1) * 100 
                    if row['Preço Atual'] > 0 and row['Preço Teto (Mediana)'] > 0 
                    else -100.0,
                    axis=1
                )
                
                # v4.2: highlight verde pastel nas linhas com Margem Mediana positiva
                def highlight_margem_mediana(row):
                    margem_val = row.get('Margem de Segurança Mediana (%)', -100)
                    if isinstance(margem_val, (int, float)) and margem_val > 0:
                        return ['background-color: #e6ffed'] * len(row)  # consistente com aba Valuation
                    return [''] * len(row)

                st.dataframe(
                    df_estatisticas.style
                        .apply(highlight_margem_mediana, axis=1)
                        .format({
                            'Preço Atual': 'R$ {:.2f}',
                            'Média DPA': '{:.2f}',
                            'Mediana DPA': '{:.2f}',
                            'Preço Teto (Mediana)': 'R$ {:.2f}',
                            'Margem de Segurança Mediana (%)': '{:.2f}%',
                            'CAGR DPA 5a (%)': '{:.2f}%',
                            'Anos Consecutivos': '{:.0f}'
                        }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Explicações educativas sobre Média e Mediana DPA
                st.markdown('<div class="explicacao-container">', unsafe_allow_html=True)
                st.subheader("O que significam Média DPA e Mediana DPA?")
                
                st.write(
                    f"**Média DPA (Dividendos Por Ação - Média):** "
                    f"É a soma de todos os dividendos pagos por ação nos últimos {anos_input} anos "
                    f"dividida pelo número de anos. A média considera TODOS os valores igualmente, "
                    f"o que significa que um ano excepcional (com dividendos extraordinários muito altos) "
                    f"ou um ano ruim (com dividendos muito baixos) pode distorcer significativamente o resultado. "
                    f"Por exemplo, se uma empresa pagou R$ 1,00 por 4 anos e R$ 5,00 em 1 ano (dividendo extraordinário), "
                    f"a média será R$ 1,80, que pode não refletir a capacidade recorrente de pagamento."
                )
                st.latex(
                    r"Média\ DPA = \frac{DPA_1 + DPA_2 + ... + DPA_n}{n}"
                )
                
                st.write(
                    f"**Mediana DPA (Dividendos Por Ação - Mediana):** "
                    f"É o valor CENTRAL quando os dividendos dos {anos_input} anos são ordenados do menor "
                    f"para o maior. A mediana é mais resistente a valores extremos (outliers) e representa "
                    f"melhor a capacidade 'típica' de pagamento de dividendos da empresa. Se houver um número "
                    f"par de anos, a mediana é a média dos dois valores centrais. Por exemplo, com os mesmos "
                    f"dados anteriores (R$ 1,00, R$ 1,00, R$ 1,00, R$ 1,00, R$ 5,00), a mediana seria R$ 1,00, "
                    f"refletindo melhor o pagamento recorrente."
                )
                st.latex(
                    r"Mediana\ DPA = Valor\ Central\ dos\ DPAs\ Ordenados"
                )
                
                st.write(
                    f"**Diferenças Práticas entre Média e Mediana:**"
                )
                st.write(
                    f"• **Sensibilidade a Valores Extremos:** A média é afetada por dividendos extraordinários "
                    f"(não-recorrentes), enquanto a mediana ignora esses extremos e foca no comportamento típico."
                )
                st.write(
                    f"• **Quando usar Média:** Use quando a empresa tem histórico estável de dividendos, sem "
                    f"grandes variações ou pagamentos extraordinários. A média captura melhor o crescimento gradual."
                )
                st.write(
                    f"• **Quando usar Mediana:** Use quando há suspeita de dividendos extraordinários (eventos únicos) "
                    f"ou grande volatilidade nos pagamentos. A mediana oferece uma visão mais conservadora e realista."
                )
                
                st.write(
                    f"**Impacto na Tomada de Decisão de Investimento:**"
                )
                st.write(
                    f"• **Preço Teto Baseado na Média:** Pode superestimar o valor justo se houver dividendos "
                    f"extraordinários no histórico, levando você a pagar mais caro por uma ação do que deveria. "
                    f"Resultado: Rentabilidade futura pode ser menor que a esperada."
                )
                st.write(
                    f"• **Preço Teto Baseado na Mediana:** Oferece uma margem de segurança adicional, pois "
                    f"considera apenas a capacidade típica/recorrente de pagamento. É mais conservador e protege "
                    f"contra surpresas negativas. Resultado: Maior probabilidade de atingir ou superar a rentabilidade desejada."
                )
                st.write(
                    f"• **Estratégia Recomendada:** Compare AMBOS os valores. Se Preço Teto (Média) e Preço Teto (Mediana) "
                    f"são muito diferentes, investigue se houve dividendos extraordinários. Se forem similares, indica "
                    f"consistência no pagamento. Use o Preço Teto (Mediana) como referência para decisões mais conservadoras "
                    f"e seguras, especialmente se você busca renda passiva previsível."
                )
                
                st.write(
                    f"**Margem de Segurança (%) - Baseada no Preço Teto (Mediana):**"
                )
                st.write(
                    f"A Margem de Segurança indica o quanto o Preço Teto (Mediana) está acima ou abaixo do Preço Atual de mercado. "
                    f"Este indicador é calculado usando a Mediana DPA (e não a Média), oferecendo uma análise mais conservadora "
                    f"e realista da oportunidade de investimento. A fórmula compara o valor justo baseado na capacidade típica "
                    f"de pagamento de dividendos com o preço que o mercado está cobrando hoje."
                )
                st.latex(
                    r"Margem\ de\ Segurança\ (\%) = \left( \frac{Preço\ Teto\ (Mediana)}{Preço\ Atual} - 1 \right) \times 100"
                )
                
                st.write(
                    f"**Interpretação dos Valores:**"
                )
                st.write(
                    f"• **Margem POSITIVA (ex: +25%):** O Preço Teto (Mediana) é MAIOR que o Preço Atual. "
                    f"Isso significa que a ação está sendo negociada com 'desconto' em relação à sua capacidade típica "
                    f"de pagar dividendos. Quanto maior a margem positiva, maior o potencial de valorização ou maior a "
                    f"rentabilidade futura esperada. Exemplo: Margem de +25% indica que você está pagando R$ 100 por uma "
                    f"ação que 'vale' R$ 125 baseado nos dividendos recorrentes."
                )
                st.write(
                    f"• **Margem NEGATIVA (ex: -15%):** O Preço Teto (Mediana) é MENOR que o Preço Atual. "
                    f"Isso significa que a ação está sendo negociada ACIMA do valor justo baseado nos dividendos. "
                    f"A ação pode estar 'cara' e você não conseguirá atingir sua taxa de retorno desejada se comprar agora. "
                    f"Exemplo: Margem de -15% indica que você está pagando R$ 100 por uma ação que 'vale' apenas R$ 85 "
                    f"baseado nos dividendos recorrentes."
                )
                st.write(
                    f"• **Margem PRÓXIMA DE ZERO (ex: -5% a +5%):** O Preço Atual está muito próximo do Preço Teto (Mediana). "
                    f"A ação está sendo negociada próxima ao 'valor justo'. Não há grande desconto, mas também não está "
                    f"muito cara. Pode ser uma oportunidade neutra, adequada para quem busca apenas manter a carteira."
                )
                
                st.write(
                    f"**Significado para o Investidor:**"
                )
                st.write(
                    f"• **Tomada de Decisão:** Margens positivas indicam potenciais oportunidades de compra. Margens negativas "
                    f"sugerem aguardar uma correção de preço ou buscar outras opções. A Margem de Segurança ajuda a evitar "
                    f"pagar caro demais por uma ação e protege contra perdas."
                )
                st.write(
                    f"• **Gestão de Risco:** Uma margem positiva oferece 'colchão de proteção'. Se a empresa reduzir dividendos "
                    f"no futuro, você ainda tem espaço para absorver essa queda sem prejuízo significativo. Com margem negativa, "
                    f"qualquer redução nos dividendos resultará em perda imediata."
                )
                st.write(
                    f"• **Rentabilidade Esperada:** Se você comprar com margem positiva de +20% e a taxa de retorno desejada é 6%, "
                    f"sua rentabilidade real poderá ser superior (por exemplo, 7,2% ao ano), pois pagou menos que o 'valor justo'. "
                    f"Com margem negativa de -20%, sua rentabilidade será inferior ao desejado (por exemplo, 4,8% ao ano)."
                )
                st.write(
                    f"• **Estratégia Conservadora:** Investidores conservadores buscam margens de segurança de pelo menos +15% a +20% "
                    f"antes de comprar. Investidores mais agressivos podem aceitar margens menores (+5% a +10%). Margens negativas "
                    f"geralmente devem ser evitadas, a menos que você tenha forte convicção de que os dividendos crescerão "
                    f"significativamente no futuro."
                )
                
                st.write(
                    f"**CAGR DPA (Crescimento de Dividendos):**"
                )
                st.write(
                    f"O CAGR DPA mede a taxa de crescimento anual composta dos dividendos pagos pela empresa nos últimos "
                    f"{anos_input} anos. É um indicador fundamental para estratégias buy-and-hold, pois mostra se você "
                    f"receberá MAIS dividendos no futuro. Um CAGR DPA positivo significa que os dividendos estão crescendo "
                    f"consistentemente, protegendo sua renda passiva contra a inflação e aumentando seu yield on cost ao "
                    f"longo do tempo."
                )
                st.latex(
                    r"CAGR\ DPA = \left( \frac{DPA\ Final}{DPA\ Inicial} \right)^{\frac{1}{anos}} - 1"
                )
                st.write(
                    f"• **CAGR Positivo (+5% a +15%):** Dividendos crescentes. Sua renda passiva aumentará com o tempo. "
                    f"Exemplo: Se você recebe R$ 1.000/ano hoje e o CAGR DPA é +8%, em 10 anos receberá R$ 2.159/ano. ✅"
                )
                st.write(
                    f"• **CAGR Neutro (0% a +3%):** Dividendos estagnados. Sua renda passiva se mantém, mas perde para "
                    f"a inflação ao longo do tempo. ⚠️"
                )
                st.write(
                    f"• **CAGR Negativo (< 0%):** Dividendos em queda. Risco alto - a empresa pode estar com problemas "
                    f"ou reduzindo distribuições. Evite para estratégias de longo prazo. ❌"
                )
                
                st.write(
                    f"**Anos Consecutivos Pagando Dividendos:**"
                )
                st.write(
                    f"Este indicador mostra quantos anos seguidos a empresa pagou dividendos sem interrupção. É uma medida "
                    f"de CONFIABILIDADE fundamental para investidores buy-and-hold. Empresas que pagam dividendos "
                    f"consistentemente por 10, 15, 20 anos demonstram compromisso com os acionistas e solidez financeira."
                )
                st.write(
                    f"• **15+ anos consecutivos:** 'Dividend Aristocrat' - Confiabilidade máxima. Empresas raras que "
                    f"mantêm pagamentos em crises, recessões e ciclos econômicos completos. ⭐⭐⭐"
                )
                st.write(
                    f"• **10-15 anos consecutivos:** Alta confiabilidade. Empresas maduras com histórico comprovado. "
                    f"Adequadas para renda passiva de longo prazo. ⭐⭐"
                )
                st.write(
                    f"• **5-10 anos consecutivos:** Confiabilidade moderada. Empresas ainda construindo histórico. "
                    f"Monitore de perto. ⭐"
                )
                st.write(
                    f"• **< 5 anos consecutivos:** Baixa confiabilidade. Pode ser empresa nova em dividendos ou com "
                    f"pagamentos inconsistentes. Risco maior para estratégias de renda passiva. ⚠️"
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("ℹ️ Não há dados de dividendos disponíveis para gráfico.")

        # Tab 4: Projeção de Ganhos
        with tab4:
            st.subheader("Projeção de Ganhos com Dividendos/JCP")
            st.info(
                "ℹ️ Informe abaixo a quantidade de ações que você possui de cada ticker. "
                "A projeção é calculada com base na **Mediana DPA** do período analisado — "
                "critério conservador, resistente a dividendos extraordinários."
            )

            ano_corrente = _agora_sp().year

            # --- Paleta de cores para os gráficos — uma cor por ticker ---
            PALETA_CORES = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
            ]

            # --- Entrada de quantidade de ações por ticker ---
            st.markdown("#### Quantidade de ações por ticker")
            qtd_acoes = {}

            # Ordena tickers alfabeticamente
            tickers_com_dpa = sorted(
                [r for r in data_estatisticas if r.get('Mediana DPA', 0) > 0],
                key=lambda r: r['Ticker']
            )

            if not tickers_com_dpa:
                st.warning("⚠️ Nenhum ticker com histórico de dividendos disponível para projeção.")
            else:
                # Inicializa session_state para persistência das quantidades
                # Os valores digitados pelo usuário são mantidos entre reruns
                # e entre relatórios, até que o usuário os altere manualmente
                if 'qtd_acoes' not in st.session_state:
                    st.session_state['qtd_acoes'] = {}

                # Organiza inputs em 4 colunas para compactar a interface
                cols_input = st.columns(4)
                for i, row in enumerate(tickers_com_dpa):
                    tk = row['Ticker']
                    # Recupera valor salvo ou usa 0 como padrão inicial
                    valor_salvo = st.session_state['qtd_acoes'].get(tk, 0)
                    with cols_input[i % 4]:
                        qtd_digitada = st.number_input(
                            tk,
                            min_value=0,
                            value=valor_salvo,
                            step=1,
                            key=f"qtd_{tk}"
                        )
                        # Persiste o valor digitado no session_state
                        st.session_state['qtd_acoes'][tk] = qtd_digitada
                        qtd_acoes[tk] = qtd_digitada

                # --- Cálculo da projeção ---
                linhas_projecao = []
                total_investido  = 0.0
                total_acoes      = 0

                for row in tickers_com_dpa:
                    tk       = row['Ticker']
                    mediana  = row.get('Mediana DPA', 0.0)
                    preco    = row.get('Preço Atual', 0.0)
                    qtd      = qtd_acoes.get(tk, 0)

                    ganho_anual    = mediana * qtd
                    ganho_mensal   = ganho_anual / 12
                    yield_on_cost  = (mediana / preco * 100) if preco > 0 else 0.0
                    investido_tk   = preco * qtd

                    total_investido += investido_tk
                    total_acoes     += qtd

                    linhas_projecao.append({
                        'Ticker':                  tk,
                        'Qtd. Ações':              qtd,
                        'Preço Atual (R$)':        preco,
                        'Total Investido (R$)':    investido_tk,
                        'Mediana DPA (R$)':        mediana,
                        'Ganho Anual (R$)':        ganho_anual,
                        'Ganho Mensal Médio (R$)': ganho_mensal,
                        'Yield on Cost (%)':       yield_on_cost
                    })

                df_projecao = pd.DataFrame(linhas_projecao)

                # Linha de totais — inclui total investido e total de ações
                total_anual  = df_projecao['Ganho Anual (R$)'].sum()
                total_mensal = df_projecao['Ganho Mensal Médio (R$)'].sum()
                linha_total  = {
                    'Ticker':                  'TOTAL',
                    'Qtd. Ações':              total_acoes,
                    'Preço Atual (R$)':        '—',
                    'Total Investido (R$)':    total_investido,
                    'Mediana DPA (R$)':        '—',
                    'Ganho Anual (R$)':        total_anual,
                    'Ganho Mensal Médio (R$)': total_mensal,
                    'Yield on Cost (%)':       '—'
                }
                df_projecao_exib = pd.concat(
                    [df_projecao, pd.DataFrame([linha_total])],
                    ignore_index=True
                )

                st.markdown("#### Tabela de Projeção Anual")
                st.dataframe(
                    df_projecao_exib.style.format({
                        'Preço Atual (R$)':        lambda x: f'R$ {x:.2f}' if isinstance(x, float) else x,
                        'Total Investido (R$)':    lambda x: f'R$ {x:.2f}' if isinstance(x, float) else x,
                        'Mediana DPA (R$)':        lambda x: f'R$ {x:.2f}' if isinstance(x, float) else x,
                        'Ganho Anual (R$)':        lambda x: f'R$ {x:.2f}' if isinstance(x, float) else x,
                        'Ganho Mensal Médio (R$)': lambda x: f'R$ {x:.2f}' if isinstance(x, float) else x,
                        'Yield on Cost (%)':       lambda x: f'{x:.2f}%'   if isinstance(x, float) else x,
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                # --- Gráfico 1: Ganho anual por ticker (barras horizontais, cor por ticker) ---
                df_graf_barras = (
                    df_projecao[df_projecao['Qtd. Ações'] > 0]
                    .sort_values('Ganho Anual (R$)', ascending=True)
                )

                if not df_graf_barras.empty:
                    st.markdown("#### Ganho Anual Projetado por Ticker")
                    # Atribui uma cor distinta a cada ticker
                    cores_barras = [
                        PALETA_CORES[i % len(PALETA_CORES)]
                        for i in range(len(df_graf_barras))
                    ]
                    fig_barras = go.Figure(go.Bar(
                        x=df_graf_barras['Ganho Anual (R$)'],
                        y=df_graf_barras['Ticker'],
                        orientation='h',
                        marker_color=cores_barras,
                        text=[f'R$ {v:.2f}' for v in df_graf_barras['Ganho Anual (R$)']],
                        textposition='outside'
                    ))
                    fig_barras.update_layout(
                        xaxis_title="Ganho Anual Projetado (R$)",
                        yaxis_title="Ticker",
                        hovermode='y unified',
                        margin=dict(l=10, r=60, t=30, b=40)
                    )
                    st.plotly_chart(fig_barras, use_container_width=True)

                # --- Gráfico 2: Projeção mensal ao longo do ano corrente ---
                # Barras empilhadas por ticker — cor distinta por ação
                st.markdown(f"#### Projeção de Recebimentos Mês a Mês — {ano_corrente}")

                NOMES_MESES = [
                    'Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                    'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'
                ]
                # Abreviações em português → número do mês
                # (o yfinance retorna abreviações em inglês via strftime('%b') no locale padrão)
                MAP_MES = {
                    'Jan': 1, 'Feb': 1, 'Fev': 1,
                    'Mar': 3,
                    'Apr': 4, 'Abr': 4,
                    'May': 5, 'Mai': 5,
                    'Jun': 6,
                    'Jul': 7,
                    'Aug': 8, 'Ago': 8,
                    'Sep': 9, 'Set': 9,
                    'Oct': 10, 'Out': 10,
                    'Nov': 11,
                    'Dec': 12, 'Dez': 12
                }

                # Mapa ticker → meses de pagamento consistentes
                mapa_pagto = {
                    r['Ticker']: r.get('Pagamento Dividendos', '')
                    for r in data_valuation
                }

                # Calcula recebimento mensal por ticker individualmente
                # para gerar barras empilhadas com cor distinta por ação
                recebimentos_por_ticker = {}
                for idx_tk, row in enumerate(linhas_projecao):
                    tk          = row['Ticker']
                    ganho_anual = row['Ganho Anual (R$)']
                    qtd         = row['Qtd. Ações']

                    if qtd == 0 or ganho_anual == 0:
                        continue

                    pagto_str = mapa_pagto.get(tk, '')
                    meses_pagto_num = []
                    if pagto_str and pagto_str not in ('Sem Div.', 'N/A', 'Erro'):
                        for parte in pagto_str.split(','):
                            abrev = parte.strip()[:3].capitalize()
                            num = MAP_MES.get(abrev)
                            if num:
                                meses_pagto_num.append(num)

                    valores_tk = [0.0] * 12
                    if not meses_pagto_num:
                        for m in range(12):
                            valores_tk[m] = ganho_anual / 12
                    else:
                        valor_por_mes = ganho_anual / len(meses_pagto_num)
                        for m in meses_pagto_num:
                            valores_tk[m - 1] = valor_por_mes

                    recebimentos_por_ticker[tk] = {
                        'valores': valores_tk,
                        'cor':     PALETA_CORES[idx_tk % len(PALETA_CORES)]
                    }

                # Total mensal agregado para a linha acumulada
                totais_mensais = [0.0] * 12
                for tk_data in recebimentos_por_ticker.values():
                    for m in range(12):
                        totais_mensais[m] += tk_data['valores'][m]

                acumulado_mensal = []
                acum = 0.0
                for v in totais_mensais:
                    acum += v
                    acumulado_mensal.append(acum)

                fig_mensal = go.Figure()

                # Barra empilhada por ticker — cor distinta para cada ação
                for tk, tk_data in recebimentos_por_ticker.items():
                    fig_mensal.add_trace(go.Bar(
                        x=NOMES_MESES,
                        y=tk_data['valores'],
                        name=tk,
                        marker_color=tk_data['cor'],
                        yaxis='y1'
                    ))

                # Linha — acumulado no ano
                fig_mensal.add_trace(go.Scatter(
                    x=NOMES_MESES,
                    y=acumulado_mensal,
                    name='Acumulado no ano (R$)',
                    mode='lines+markers',
                    line=dict(color='#000000', width=2),
                    marker=dict(size=7),
                    yaxis='y2'
                ))

                fig_mensal.update_layout(
                    xaxis_title="Mês",
                    yaxis=dict(title="Recebimento no Mês (R$)", showgrid=False),
                    yaxis2=dict(
                        title="Acumulado no Ano (R$)",
                        overlaying='y',
                        side='right',
                        showgrid=False
                    ),
                    barmode='stack',
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(l=10, r=10, t=40, b=40)
                )
                st.plotly_chart(fig_mensal, use_container_width=True)

                st.caption(
                    f"📌 Meses sem barra indicam ausência de pagamento histórico consistente "
                    f"naquele mês. Tickers sem meses definidos têm ganho distribuído uniformemente."
                )

                # ---------------------------------------------------------------
                # TABELA DE RANKING DE MESES — formatação rica com HTML
                # ---------------------------------------------------------------
                st.markdown("---")
                st.markdown("#### 🏆 Ranking de Meses por Recebimento")

                MEDALHAS = {0: '🥇', 1: '🥈', 2: '🥉'}
                COR_TOP3 = {
                    0: ('background:#FFF9C4;font-weight:700;', '#856404'),  # ouro
                    1: ('background:#F0F0F0;font-weight:700;', '#555'),     # prata
                    2: ('background:#FFE0CC;font-weight:700;', '#7a3b00'),  # bronze
                }

                # Monta lista de meses ordenada por total decrescente
                ranking_meses = sorted(
                    [
                        {
                            'mes_num': m,
                            'mes_nome': NOMES_MESES[m - 1],
                            'total': totais_mensais[m - 1],
                            'tickers_pagam': ', '.join([
                                tk for tk, d in recebimentos_por_ticker.items()
                                if d['valores'][m - 1] > 0
                            ]) or '—'
                        }
                        for m in range(1, 13)
                    ],
                    key=lambda x: x['total'],
                    reverse=True
                )

                total_ano_ranking = sum(r['total'] for r in ranking_meses)

                # Constrói HTML da tabela
                linhas_html = ''
                for pos, r in enumerate(ranking_meses):
                    medalha  = MEDALHAS.get(pos, f'{pos + 1}º')
                    pct      = (r['total'] / total_ano_ranking * 100) if total_ano_ranking > 0 else 0
                    valor_fmt = f"R$ {r['total']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')

                    if pos in COR_TOP3:
                        estilo_linha, cor_texto = COR_TOP3[pos]
                    else:
                        estilo_linha = 'background:#ffffff;'
                        cor_texto    = '#333'

                    linhas_html += f"""
                    <tr style="{estilo_linha}color:{cor_texto};border-bottom:1px solid #eee;">
                        <td style="padding:10px 14px;text-align:center;font-size:18px;">{medalha}</td>
                        <td style="padding:10px 14px;font-weight:600;font-size:15px;">{r['mes_nome']}</td>
                        <td style="padding:10px 14px;text-align:right;font-size:15px;">{valor_fmt}</td>
                        <td style="padding:10px 14px;text-align:right;">{pct:.1f}%</td>
                        <td style="padding:10px 14px;font-size:13px;color:#555;">{r['tickers_pagam']}</td>
                    </tr>"""

                tabela_html = f"""
                <div style="overflow-x:auto;border-radius:12px;border:1px solid #e0e0e0;
                            box-shadow:0 2px 8px rgba(0,0,0,0.07);margin-bottom:24px;">
                  <table style="width:100%;border-collapse:collapse;font-family:sans-serif;">
                    <thead>
                      <tr style="background:#1f77b4;color:white;">
                        <th style="padding:12px 14px;text-align:center;">Posição</th>
                        <th style="padding:12px 14px;text-align:left;">Mês</th>
                        <th style="padding:12px 14px;text-align:right;">Total a Receber</th>
                        <th style="padding:12px 14px;text-align:right;">% do Ano</th>
                        <th style="padding:12px 14px;text-align:left;">Tickers que pagam</th>
                      </tr>
                    </thead>
                    <tbody>{linhas_html}</tbody>
                    <tfoot>
                      <tr style="background:#f5f5f5;font-weight:700;border-top:2px solid #1f77b4;">
                        <td colspan="2" style="padding:10px 14px;">TOTAL ANUAL</td>
                        <td style="padding:10px 14px;text-align:right;">
                          R$ {total_ano_ranking:,.2f}
                        </td>
                        <td style="padding:10px 14px;text-align:right;">100%</td>
                        <td></td>
                      </tr>
                    </tfoot>
                  </table>
                </div>""".replace(',', 'X').replace('.', ',').replace('X', '.')

                # Corrige apenas os valores monetários (evita corromper o HTML)
                # A substituição acima é para formato BR — refaz só os valores numéricos
                tabela_html = f"""
                <div style="overflow-x:auto;border-radius:12px;border:1px solid #e0e0e0;
                            box-shadow:0 2px 8px rgba(0,0,0,0.07);margin-bottom:24px;">
                  <table style="width:100%;border-collapse:collapse;font-family:sans-serif;">
                    <thead>
                      <tr style="background:#1f77b4;color:white;">
                        <th style="padding:12px 14px;text-align:center;">Posição</th>
                        <th style="padding:12px 14px;text-align:left;">Mês</th>
                        <th style="padding:12px 14px;text-align:right;">Total a Receber (R$)</th>
                        <th style="padding:12px 14px;text-align:right;">% do Ano</th>
                        <th style="padding:12px 14px;text-align:left;">Tickers que pagam</th>
                      </tr>
                    </thead>
                    <tbody>{"".join([
                        f"<tr style='{(COR_TOP3[p][0] if p in COR_TOP3 else 'background:#ffffff;')}color:{(COR_TOP3[p][1] if p in COR_TOP3 else '#333')};border-bottom:1px solid #eee;'>"
                        f"<td style='padding:10px 14px;text-align:center;font-size:18px;'>{MEDALHAS.get(p, str(p+1)+'º')}</td>"
                        f"<td style='padding:10px 14px;font-weight:600;font-size:15px;'>{r['mes_nome']}</td>"
                        f"<td style='padding:10px 14px;text-align:right;font-size:15px;'>R$ {r['total']:,.2f}</td>"
                        f"<td style='padding:10px 14px;text-align:right;'>{((r['total']/total_ano_ranking*100) if total_ano_ranking>0 else 0):.1f}%</td>"
                        f"<td style='padding:10px 14px;font-size:13px;color:#555;'>{r['tickers_pagam']}</td>"
                        f"</tr>"
                        for p, r in enumerate(ranking_meses)
                    ])}</tbody>
                    <tfoot>
                      <tr style="background:#f5f5f5;font-weight:700;border-top:2px solid #1f77b4;">
                        <td colspan="2" style="padding:10px 14px;">TOTAL ANUAL</td>
                        <td style="padding:10px 14px;text-align:right;">R$ {total_ano_ranking:,.2f}</td>
                        <td style="padding:10px 14px;text-align:right;">100%</td>
                        <td></td>
                      </tr>
                    </tfoot>
                  </table>
                </div>"""
                st.markdown(tabela_html, unsafe_allow_html=True)

                # Identifica meses sem registro estatístico (total = 0)
                # e exibe mensagem informativa ao usuário
                meses_sem_registro = [
                    r['mes_nome'] for r in ranking_meses if r['total'] == 0
                ]
                if meses_sem_registro:
                    st.info(
                        f"ℹ️ **Meses sem registros estatísticos:** "
                        f"Para os **{int(anos_input)} anos analisados**, os meses "
                        f"**{', '.join(meses_sem_registro)}** não apresentaram padrão "
                        f"consistente de pagamento de dividendos/JCP em nenhum dos tickers "
                        f"selecionados. O valor projetado para esses meses é R$ 0,00. "
                        f"Aumentar o número de anos na barra lateral pode revelar padrões "
                        f"adicionais caso existam."
                    )
                # EXPORTAR SIMULAÇÃO PARA EXCEL
                # ---------------------------------------------------------------
                st.markdown("---")
                st.markdown("#### 💾 Exportar Simulação")

                col_nome, col_btn = st.columns([3, 1])
                with col_nome:
                    nome_simulacao = st.text_input(
                        "Nome da simulação",
                        value=f"Simulacao_{_agora_sp().strftime('%Y%m%d')}",
                        key="nome_simulacao",
                        help="Digite um nome para identificar esta simulação nos arquivos exportados"
                    )
                with col_btn:
                    st.markdown("<br>", unsafe_allow_html=True)

                data_hora_sim  = _agora_sp().strftime('%d/%m/%Y %H:%M:%S')
                nome_base      = nome_simulacao.strip().replace(' ', '_')
                sufixo_dt      = _agora_sp().strftime('%Y%m%d_%H%M')
                nome_xlsx      = f"{nome_base}_{sufixo_dt}.xlsx"
                nome_csv       = f"{nome_base}_{sufixo_dt}.csv"

                # Monta dados da simulação por ticker
                df_export_sim = df_projecao[[
                    'Ticker', 'Qtd. Ações', 'Preço Atual (R$)', 'Total Investido (R$)',
                    'Mediana DPA (R$)', 'Ganho Anual (R$)', 'Ganho Mensal Médio (R$)', 'Yield on Cost (%)'
                ]].copy()
                df_export_sim.loc[len(df_export_sim)] = {
                    'Ticker':                  'TOTAL',
                    'Qtd. Ações':              total_acoes,
                    'Preço Atual (R$)':        None,
                    'Total Investido (R$)':    total_investido,
                    'Mediana DPA (R$)':        None,
                    'Ganho Anual (R$)':        total_anual,
                    'Ganho Mensal Médio (R$)': total_mensal,
                    'Yield on Cost (%)':       None
                }

                # Monta projeção mensal
                df_export_mensal = pd.DataFrame({
                    'Mês':           NOMES_MESES,
                    'Total (R$)':    totais_mensais,
                    'Acumulado (R$)':acumulado_mensal,
                    '% do Ano':      [
                        round(v / total_ano_ranking * 100, 1) if total_ano_ranking > 0 else 0
                        for v in totais_mensais
                    ]
                })

                # Monta ranking de meses
                df_export_ranking = pd.DataFrame([
                    {
                        'Posição':           p + 1,
                        'Mês':               r['mes_nome'],
                        'Total (R$)':        round(r['total'], 2),
                        '% do Ano':          round((r['total'] / total_ano_ranking * 100) if total_ano_ranking > 0 else 0, 1),
                        'Tickers que pagam': r['tickers_pagam']
                    }
                    for p, r in enumerate(ranking_meses)
                ])

                # Metadados
                df_meta = pd.DataFrame({
                    'Campo': ['Nome da Simulação', 'Data e Hora', 'Anos Analisados', 'Taxa de Retorno (%)'],
                    'Valor': [nome_simulacao, data_hora_sim, int(anos_input), taxa_retorno_input]
                })

                # --- Botões de download lado a lado ---
                col_xlsx, col_csv = st.columns(2)

                # Botão Excel — requer openpyxl no requirements.txt
                with col_xlsx:
                    try:
                        buffer_excel = io.BytesIO()
                        with pd.ExcelWriter(buffer_excel, engine='openpyxl') as writer:
                            df_meta.to_excel(writer, sheet_name='Informações', index=False)
                            df_export_sim.to_excel(writer, sheet_name='Projeção por Ticker', index=False)
                            df_export_mensal.to_excel(writer, sheet_name='Projeção Mensal', index=False)
                            df_export_ranking.to_excel(writer, sheet_name='Ranking de Meses', index=False)
                        buffer_excel.seek(0)
                        st.download_button(
                            label="📥 Baixar em Excel (.xlsx)",
                            data=buffer_excel,
                            file_name=nome_xlsx,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help=f"Simulação '{nome_simulacao}' — {data_hora_sim} | Requer openpyxl no requirements.txt",
                            use_container_width=True
                        )
                    except ImportError:
                        st.warning(
                            "⚠️ **Excel indisponível:** a biblioteca `openpyxl` não está instalada. "
                            "Adicione `openpyxl` ao `requirements.txt` e faça novo deploy para habilitar."
                        )

                # Botão CSV — sem dependência extra, sempre disponível
                with col_csv:
                    # CSV único: metadados + separador + projeção por ticker
                    # + separador + mensal + separador + ranking
                    linhas_csv = []
                    linhas_csv.append("=== INFORMAÇÕES DA SIMULAÇÃO ===")
                    linhas_csv.append(df_meta.to_csv(index=False))
                    linhas_csv.append("=== PROJEÇÃO POR TICKER ===")
                    linhas_csv.append(df_export_sim.to_csv(index=False))
                    linhas_csv.append("=== PROJEÇÃO MENSAL ===")
                    linhas_csv.append(df_export_mensal.to_csv(index=False))
                    linhas_csv.append("=== RANKING DE MESES ===")
                    linhas_csv.append(df_export_ranking.to_csv(index=False))
                    csv_completo = "\n".join(linhas_csv)

                    st.download_button(
                        label="📄 Baixar em CSV (.csv)",
                        data=csv_completo.encode('utf-8-sig'),  # utf-8-sig para Excel BR reconhecer acentos
                        file_name=nome_csv,
                        mime="text/csv",
                        help=f"Simulação '{nome_simulacao}' — {data_hora_sim} | Sem dependências extras",
                        use_container_width=True
                    )

                # ---------------------------------------------------------------
                # BOTÃO DE IMPRESSÃO — apenas conteúdo da tab Projeção de Ganhos
                # CSS @media print oculta sidebar, header e demais tabs
                # ---------------------------------------------------------------
                st.markdown("---")
                st.markdown("#### 🖨️ Imprimir Projeção")
                st.caption(
                    "Clique no botão abaixo para abrir o diálogo de impressão do Windows. "
                    "Selecione 'Microsoft Print to PDF' ou 'Salvar como PDF' para guardar o documento."
                )

                css_print = """
                <style>
                @media print {
                    /* Oculta elementos fora da tab ativa */
                    [data-testid="stSidebar"],
                    [data-testid="stHeader"],
                    [data-testid="stToolbar"],
                    [data-testid="stDecoration"],
                    [data-testid="stStatusWidget"],
                    button[kind="primary"],
                    .stTabs [data-baseweb="tab-list"],
                    footer { display: none !important; }

                    /* Garante que o conteúdo imprime em largura total */
                    [data-testid="stAppViewContainer"] { margin: 0 !important; }
                    [data-testid="block-container"]    { padding: 0 !important; }

                    /* Evita quebra de página dentro de tabelas */
                    table, .stDataFrame { page-break-inside: avoid; }
                }
                </style>
                <script>
                function imprimirPagina() { window.print(); }
                </script>
                <button onclick="imprimirPagina()"
                    style="background:#1f77b4;color:white;border:none;padding:10px 28px;
                           border-radius:8px;font-size:15px;font-weight:600;cursor:pointer;
                           box-shadow:0 2px 6px rgba(0,0,0,0.15);">
                    🖨️ Imprimir / Salvar PDF
                </button>"""
                st.markdown(css_print, unsafe_allow_html=True)

                # ---------------------------------------------------------------
                # SEÇÃO EXPLICATIVA
                # ---------------------------------------------------------------
                st.markdown("---")
                st.markdown('<div class="explicacao-container">', unsafe_allow_html=True)
                st.subheader("O que significam os indicadores e como são calculados?")

                st.write(
                    "**Ganho Anual Projetado (R$):** "
                    "É o total estimado de dividendos e/ou JCP que você receberá no ano, "
                    "com base na quantidade de ações informada e na Mediana DPA do período "
                    "analisado. Utiliza a mediana — e não a média — por ser mais resistente "
                    "a distorções causadas por pagamentos extraordinários não-recorrentes."
                )
                st.latex(
                    r"Ganho\ Anual = Qtd.\ Ações \times Mediana\ DPA"
                )

                st.write(
                    "**Ganho Mensal Médio (R$):** "
                    "Divide o Ganho Anual Projetado por 12, oferecendo uma referência de "
                    "renda passiva mensal equivalente. É uma média — os recebimentos reais "
                    "ocorrem nos meses de pagamento histórico de cada empresa, conforme "
                    "mostrado no gráfico de projeção mensal."
                )
                st.latex(
                    r"Ganho\ Mensal\ Médio = \frac{Ganho\ Anual}{12}"
                )

                st.write(
                    "**Yield on Cost (%):** "
                    "Indica o rendimento real em dividendos sobre o preço atual da ação. "
                    "Diferente do Dividend Yield divulgado pelo mercado (que pode usar médias "
                    "ou estimativas), o Yield on Cost usa a Mediana DPA histórica do período "
                    "analisado — refletindo a capacidade típica e recorrente de pagamento "
                    "da empresa."
                )
                st.latex(
                    r"Yield\ on\ Cost\ (\%) = \frac{Mediana\ DPA}{Preço\ Atual} \times 100"
                )

                st.write(
                    f"**Gráfico de Recebimentos Mês a Mês ({ano_corrente}):** "
                    "Distribui os ganhos projetados pelos meses em que cada empresa "
                    "historicamente realiza pagamentos (campo 'Pagamento Dividendos', "
                    "com threshold de 60% de consistência). O valor de cada mês é a soma "
                    "dos recebimentos esperados de todos os tickers naquele mês. "
                    "A linha verde mostra o total acumulado ao longo do ano. "
                    "Tickers sem meses de pagamento definidos têm o ganho distribuído "
                    "uniformemente pelos 12 meses."
                )

                st.write(
                    "**Importante — Limitações da Projeção:** "
                    "Os valores são estimativas baseadas em comportamento histórico de "
                    "dividendos. Pagamentos futuros dependem dos resultados e decisões "
                    "da empresa e não são garantidos. A projeção não considera variações "
                    "no número de ações, bonificações, desdobramentos ou alterações na "
                    "política de dividendos. Use como referência de planejamento, não como "
                    "garantia de rendimento."
                )

                st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.warning(
            "⚠️ Não foi possível processar os dados. "
            "Verifique os tickers e tente novamente."
        )
