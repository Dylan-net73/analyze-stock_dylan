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
import streamlit.components.v1 as st_components
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
# Injetado via st_components.html para evitar problemas de encoding
# com caracteres especiais (& na URL do Google Fonts, aspas no CSS)
_css_global = (
    '<meta charset="UTF-8">'
    '<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">'
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">'
    '<style>'
    'html, body, [class*="css"] { font-family: Inter, sans-serif; }'
    '.main { background-color: #f8f9fc; }'
    '[data-testid="stSidebar"] { background-color: #f0f2f8; border-right: 1px solid #e2e8f0; }'
    'h1 { color: #1a1f36 !important; font-weight: 700 !important; letter-spacing: -0.5px; }'
    'h2, h3, h4 { color: #2d3748 !important; font-weight: 600 !important; }'
    'p, li { color: #4a5568; }'
    '.stDataFrame { border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }'
    '.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 2px solid #e2e8f0; background-color: transparent; }'
    '.stTabs [data-baseweb="tab"] { height: 48px; background-color: transparent; border: none; font-weight: 600; font-size: 15px; color: #718096; padding: 0 16px; border-radius: 8px 8px 0 0; }'
    '.stTabs [aria-selected="true"] { color: #4361ee !important; border-bottom: 3px solid #4361ee !important; background-color: #eef1ff !important; }'
    '.stAlert { border-radius: 10px; border: none; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }'
    '.explicacao-container { background-color: #f7f9ff; padding: 28px 30px; border-radius: 12px; border-left: 5px solid #4361ee; border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; margin-top: 30px; box-shadow: 0 2px 10px rgba(67,97,238,0.06); }'
    '.stDownloadButton > button { background-color: #ffffff; color: #4361ee; border: 1.5px solid #4361ee; border-radius: 8px; padding: 6px 16px; font-weight: 500; font-size: 13px; }'
    '.stDownloadButton > button:hover { background-color: #eef1ff; }'
    '</style>'
)
st_components.html(_css_global, height=0)

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

def _classificar_status_preco(
    timestamp_preco: Optional[datetime],
    origem: str
) -> Tuple[str, str]:
    """
    Classifica o status de atualização do preço e gera mensagem ao usuário.

    Verifica se o preço obtido é atual, recente ou defasado, considerando:
    - Horário do pregão da B3 (10h–18h, horário de Brasília)
    - Diferença entre o timestamp do preço e o momento atual
    - Origem do dado (fast_info, info, history_1d, history_5d)

    Args:
        timestamp_preco: datetime com fuso horário do momento do preço (ou None)
        origem: string identificando a fonte do preço

    Returns:
        Tuple:
            - str: ícone de status ('✅', '⚠️' ou '🔴')
            - str: mensagem descritiva para exibição ao usuário
    """
    fuso_sp = pytz.timezone(FUSO_HORARIO_PADRAO)
    agora   = datetime.now(fuso_sp)

    # Horário do pregão B3
    ABERTURA_B3  = agora.replace(hour=10, minute=0, second=0, microsecond=0)
    FECHAMENTO_B3 = agora.replace(hour=18, minute=0, second=0, microsecond=0)
    mercado_aberto = ABERTURA_B3 <= agora <= FECHAMENTO_B3

    # Sem timestamp disponível — informa apenas a origem
    if timestamp_preco is None:
        if origem in ('fast_info', 'info'):
            return '✅', 'Preço atual (tempo real)'
        if origem == 'history_1d':
            if not mercado_aberto:
                return '⚠️', 'Mercado fechado — preço do fechamento de hoje'
            return '⚠️', 'Preço de fechamento do dia (intraday indisponível)'
        if origem == 'history_5d':
            if not mercado_aberto:
                return '⚠️', 'Mercado fechado — último fechamento disponível'
            return '🔴', 'Preço desatualizado — último fechamento disponível'
        return '⚠️', 'Origem do preço desconhecida'

    # Garante que o timestamp está no fuso de SP para comparação correta
    if timestamp_preco.tzinfo is None:
        timestamp_preco = pytz.utc.localize(timestamp_preco).astimezone(fuso_sp)
    else:
        timestamp_preco = timestamp_preco.astimezone(fuso_sp)

    diferenca = agora - timestamp_preco
    minutos   = int(diferenca.total_seconds() / 60)
    horas     = minutos // 60
    dias      = diferenca.days

    # Formata defasagem de forma legível
    if dias >= 1:
        label_defasagem = f"{dias} dia(s)"
    elif horas >= 1:
        minutos_resto = minutos % 60
        label_defasagem = f"{horas}h{minutos_resto:02d}min" if minutos_resto else f"{horas}h"
    else:
        label_defasagem = f"{minutos} min"

    data_hora_fmt = timestamp_preco.strftime('%d/%m %H:%M')

    # Preço considerado atual: ≤ 15 minutos E de fonte confiável
    if minutos <= 15 and origem in ('fast_info', 'info'):
        return '✅', f'Preço atual — atualizado às {timestamp_preco.strftime("%H:%M")}'

    # Preço recente mas não em tempo real
    if minutos <= 60:
        return '⚠️', f'Preço pode estar desatualizado em ~{label_defasagem} ({data_hora_fmt})'

    # Preço defasado (mais de 1 hora)
    if not mercado_aberto:
        return '⚠️', f'Mercado fechado — último fechamento: {data_hora_fmt}'

    return '🔴', f'Preço desatualizado em ~{label_defasagem} — referência: {data_hora_fmt}'


def _resolver_preco(preco_info: float, ticker_symbol: str) -> Tuple[float, str, str]:
    """
    Resolve o preço atual de uma ação com cascata de fallbacks.

    Em ambientes cloud (ex: Render), o Yahoo Finance frequentemente bloqueia
    ou retorna vazio nos campos regularMarketPrice/currentPrice para IPs de
    datacenter. Este helper percorre uma sequência de tentativas da mais
    recente para a mais antiga, retornando o primeiro valor válido encontrado:

    1. fast_info['last_price']        — preço intraday via endpoint alternativo
    2. regularMarketPrice/currentPrice — preço em tempo real via info
    3. history(period='1d')            — fechamento do dia corrente
    4. history(period='5d')            — último fechamento dos últimos 5 dias
                                         (cobre feriados e fins de semana)

    Args:
        preco_info: Valor obtido de info.get('regularMarketPrice') ou 'currentPrice'
        ticker_symbol: Símbolo do ticker (ex: "PETR4.SA")

    Returns:
        Tuple contendo:
            - float: Preço mais recente disponível ou 0.0 se todas as tentativas falharem
            - str:   Ícone de status ('✅', '⚠️' ou '🔴')
            - str:   Mensagem descritiva de status do preço para exibição ao usuário
    """
    fuso_sp = pytz.timezone(FUSO_HORARIO_PADRAO)

    try:
        tk = yf.Ticker(ticker_symbol)

        # Tentativa 1 — fast_info: endpoint alternativo, menos bloqueado no Render
        try:
            fi = tk.fast_info
            preco_fi = getattr(fi, 'last_price', None)
            if preco_fi and float(preco_fi) > 0:
                preco = float(preco_fi)
                # Tenta obter timestamp via fast_info
                ts = None
                try:
                    ts_raw = getattr(fi, 'regular_market_time', None)
                    if ts_raw:
                        if isinstance(ts_raw, (int, float)):
                            ts = datetime.fromtimestamp(ts_raw, tz=fuso_sp)
                        elif isinstance(ts_raw, datetime):
                            ts = ts_raw
                except Exception:
                    ts = None
                icone, msg = _classificar_status_preco(ts, 'fast_info')
                logger.info(f"{ticker_symbol}: preço obtido via fast_info ({preco:.2f}) | {msg}")
                return preco, icone, msg
        except Exception as e:
            logger.debug(f"{ticker_symbol}: fast_info falhou — {str(e)}")

        # Tentativa 2 — regularMarketPrice / currentPrice via info
        if preco_info and float(preco_info) > 0:
            preco = float(preco_info)
            # Tenta extrair timestamp do campo regularMarketTime via info já coletado
            ts = None
            try:
                ts_raw = tk.info.get('regularMarketTime')
                if ts_raw:
                    if isinstance(ts_raw, (int, float)):
                        ts = datetime.fromtimestamp(float(ts_raw), tz=fuso_sp)
                    elif isinstance(ts_raw, datetime):
                        ts = ts_raw
            except Exception:
                ts = None
            icone, msg = _classificar_status_preco(ts, 'info')
            logger.info(f"{ticker_symbol}: preço obtido via info ({preco:.2f}) | {msg}")
            return preco, icone, msg

        # Tentativa 3 — fechamento do dia corrente via history(1d)
        hist_1d = tk.history(period='1d')
        if not hist_1d.empty and hist_1d['Close'].iloc[-1] > 0:
            preco = float(hist_1d['Close'].iloc[-1])
            ts = None
            try:
                ts_raw = hist_1d.index[-1]
                if hasattr(ts_raw, 'to_pydatetime'):
                    ts = ts_raw.to_pydatetime()
                    if ts.tzinfo is None:
                        ts = pytz.utc.localize(ts)
                    ts = ts.astimezone(fuso_sp)
            except Exception:
                ts = None
            icone, msg = _classificar_status_preco(ts, 'history_1d')
            logger.info(f"{ticker_symbol}: preço obtido via history(1d) ({preco:.2f}) | {msg}")
            return preco, icone, msg

        # Tentativa 4 — último fechamento dos últimos 5 dias
        # (cobre feriados, fins de semana e atrasos de dados)
        hist_5d = tk.history(period='5d')
        if not hist_5d.empty and hist_5d['Close'].iloc[-1] > 0:
            preco = float(hist_5d['Close'].iloc[-1])
            ts = None
            try:
                ts_raw = hist_5d.index[-1]
                if hasattr(ts_raw, 'to_pydatetime'):
                    ts = ts_raw.to_pydatetime()
                    if ts.tzinfo is None:
                        ts = pytz.utc.localize(ts)
                    ts = ts.astimezone(fuso_sp)
            except Exception:
                ts = None
            icone, msg = _classificar_status_preco(ts, 'history_5d')
            logger.info(f"{ticker_symbol}: preço obtido via history(5d) ({preco:.2f}) | {msg}")
            return preco, icone, msg

    except Exception as e:
        logger.warning(f"_resolver_preco falhou para {ticker_symbol}: {str(e)}")

    logger.warning(f"{ticker_symbol}: todas as tentativas de preço falharam — retornando 0.0")
    return 0.0, '🔴', 'Preço indisponível — todas as tentativas falharam'

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
            # Cascata de fallbacks para ambientes cloud (ex: Render) onde
            # regularMarketPrice/currentPrice podem vir zerados por bloqueio
            # do Yahoo Finance a IPs de datacenter.
            # _resolver_preco retorna tupla: (preco, icone_status, msg_status)
            '_precoTuple': _resolver_preco(
                info.get('regularMarketPrice') or info.get('currentPrice'),
                ticker_symbol
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
    Em ambientes cloud onde esses campos vêm zerados, utiliza fallback
    via history(period='2d') para obter o último fechamento disponível.
    
    Args:
        ticker_symbol: Símbolo do ticker (ex: "PETR4.SA")
        
    Returns:
        float: Preço atual ou 0.0 se não disponível
        
    Notas:
        - Retorna 0.0 em caso de erro
        - Tenta múltiplos campos para maior confiabilidade
        - Fallback via histórico para ambientes cloud (ex: Render)
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        preco_info = info.get('regularMarketPrice') or info.get('currentPrice')
        preco, _, _ = _resolver_preco(preco_info, ticker_symbol)
        return preco
        
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
        # _precoTuple = (preco_float, icone_status, msg_status)
        preco_tuple = dados_fundamentalistas.get('_precoTuple')
        if preco_tuple and isinstance(preco_tuple, tuple) and len(preco_tuple) == 3:
            preco_atual, status_icone, status_msg = preco_tuple
        else:
            # Fallback de segurança: chama obter_preco_atual se _precoTuple ausente
            preco_atual  = obter_preco_atual(ticker)
            status_icone = '⚠️'
            status_msg   = 'Status do preço indisponível'
        # Garante que preco_atual é float válido
        if not preco_atual or preco_atual <= 0:
            preco_atual  = obter_preco_atual(ticker)
            status_icone = '⚠️'
            status_msg   = 'Status do preço indisponível'
        
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
                'Status Preço': f"{status_icone} {status_msg}",
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
            'v': {'Ticker': ticker, 'Preço Atual': 0.0, 'Status Preço': '🔴 Erro ao obter preço',
                  'Preço Teto (Bazin)': 0.0, 
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
        colunas_remover = [col for col in ['LPA', 'VPA', '_dividendRate', '_precoTuple'] if col in df_indicadores.columns]
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
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💎 Valuation",
            "📊 Indicadores",
            "📈 Gráfico DPA",
            "💰 Projeção de Ganhos",
            "🏦 Avaliação Gordon | Lynch"
        ])
        
        # Tab 1: Valuation
        with tab1:
            # Remove a coluna 'Status Preço' do DataFrame antes de qualquer
            # operação de estilo ou exibição. Usa .drop() diretamente no
            # DataFrame para garantia absoluta de que a coluna não aparece
            # na tabela. O painel de status é exibido abaixo da tabela.
            df_valuation_exib = df_valuation.drop(
                columns=[c for c in ['Status Preço'] if c in df_valuation.columns]
            ).copy()

            # Highlight dark-mode safe: aplica cor ao texto da margem (verde/vermelho)
            # e fundo neutro sutil na linha inteira via apply por linha.
            # Usa rgba para compatibilidade com dark e light mode.
            def _highlight_valuation(row):
                margem = row.get('Margem Segurança (%)', None)
                if isinstance(margem, (int, float)) and margem > 0:
                    return ['border-left: 3px solid #3B6D11'] + [''] * (len(row) - 1)
                return [''] * len(row)

            st.dataframe(
                df_valuation_exib.style
                    .apply(_highlight_valuation, axis=1)
                    .map(
                        lambda x: 'color: #3B6D11; font-weight: 600;'
                        if isinstance(x, (int, float)) and x > 0
                        else ('color: #A32D2D; font-weight: 600;'
                              if isinstance(x, (int, float)) and x < 0
                              else ''),
                        subset=['Margem Segurança (%)']
                    )
                    .format({
                        'Preço Atual': 'R$ {:.2f}',
                        'Preço Teto (Bazin)': 'R$ {:.2f}',
                        'Preço Graham': 'R$ {:.2f}',
                        'Margem Segurança (%)': '{:.2f}%'
                    }),
                use_container_width=True,
                hide_index=True
            )

            # ---------------------------------------------------------------
            # PAINEL DE STATUS DOS PREÇOS — exibido abaixo da tabela
            # Mostra o status de atualização de cada ticker individualmente,
            # agrupado por categoria (✅ atual, ⚠️ atenção, 🔴 defasado)
            # ---------------------------------------------------------------
            if 'Status Preço' in df_valuation.columns:
                st.markdown("---")
                st.markdown("#### 🕐 Status de Atualização dos Preços")

                status_por_ticker = {
                    r['Ticker']: r.get('Status Preço', '⚠️ Status indisponível')
                    for r in data_valuation
                }

                atuais    = {t: s for t, s in status_por_ticker.items() if s.startswith('✅')}
                atencao   = {t: s for t, s in status_por_ticker.items() if s.startswith('⚠️')}
                defasados = {t: s for t, s in status_por_ticker.items() if s.startswith('🔴')}

                # Exibe resumo geral em uma linha
                total = len(status_por_ticker)
                partes_resumo = []
                if atuais:
                    partes_resumo.append(f"✅ {len(atuais)} atual(is)")
                if atencao:
                    partes_resumo.append(f"⚠️ {len(atencao)} com atenção")
                if defasados:
                    partes_resumo.append(f"🔴 {len(defasados)} defasado(s)")
                st.caption(f"**{total} ticker(s) analisados:** " + " · ".join(partes_resumo))

                # Detalhe por ticker em colunas compactas
                todos_status = list(status_por_ticker.items())
                cols_status = st.columns(min(len(todos_status), 4))
                for idx, (ticker_s, msg_s) in enumerate(todos_status):
                    with cols_status[idx % 4]:
                        # Cores dark-mode safe para cards de status
                        if msg_s.startswith('✅'):
                            cor_bg, cor_borda = "rgba(59,109,17,0.07)", "#3B6D11"
                        elif msg_s.startswith('🔴'):
                            cor_bg, cor_borda = "rgba(163,45,45,0.07)", "#A32D2D"
                        else:
                            cor_bg, cor_borda = "rgba(186,117,23,0.07)", "#BA7517"

                        st.markdown(
                            f"<div style='background:{cor_bg};border-left:3px solid {cor_borda};"
                            f"border-radius:8px;padding:9px 12px;margin-bottom:8px;"
                            f"box-shadow:0 1px 4px rgba(0,0,0,0.05);'>"
                            f"<span style='font-weight:600;font-size:12px;color:#1a1f36;'>{ticker_s}</span><br>"
                            f"<span style='font-size:11px;color:#718096;'>{msg_s}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                # Aviso consolidado apenas se houver defasados ou atenção
                if defasados or atencao:
                    st.info(
                        "ℹ️ Preços marcados com ⚠️ ou 🔴 podem não refletir o valor em tempo real. "
                        "Isso ocorre quando o servidor (Render) tem acesso limitado à API do Yahoo Finance. "
                        "O sistema tentou automaticamente todas as fontes disponíveis e exibiu o dado mais recente encontrado."
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
                            
                            # Cores dark-mode safe: rgba baixa opacidade
                            if margem >= 20:
                                cor_borda = "#3B6D11"
                                cor_fundo = "rgba(59,109,17,0.07)"
                                cor_badge = "#3B6D11"
                                icone     = "🟢"
                            elif margem >= 10:
                                cor_borda = "#BA7517"
                                cor_fundo = "rgba(186,117,23,0.07)"
                                cor_badge = "#BA7517"
                                icone     = "🟡"
                            else:
                                cor_borda = "#185FA5"
                                cor_fundo = "rgba(24,95,165,0.07)"
                                cor_badge = "#185FA5"
                                icone     = "🔵"
                            
                            with col:
                                # Cards compactos dark-mode safe:
                                # fundo via rgba de baixa opacidade, borda esquerda colorida
                                html_card = (
                                    "<div style='"
                                    "background-color:" + cor_fundo + ";"
                                    "border:0.5px solid " + cor_borda + ";"
                                    "border-left:3px solid " + cor_borda + ";"
                                    "border-radius:8px;"
                                    "padding:12px 14px;"
                                    "margin-bottom:10px;'>"
                                    "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>"
                                    "<span style='font-size:14px;font-weight:500;'>" + ticker_op + "</span>"
                                    "<span style='font-size:11px;font-weight:500;color:" + cor_borda + ";'>+"
                                    + f"{margem:.1f}" + "%</span>"
                                    "</div>"
                                    "<div style='display:flex;justify-content:space-between;font-size:12px;'>"
                                    "<span>Atual <strong>R$ " + f"{preco_atual:.2f}" + "</strong></span>"
                                    "<span>Teto <strong style='color:" + cor_borda + ";'>R$ " + f"{preco_teto:.2f}" + "</strong></span>"
                                    "</div>"
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
                    title=dict(text="Evolução do DPA ao Longo dos Anos", font=dict(family="Inter", size=16, color="#1a1f36"), x=0.01),
                    xaxis_title="Ano",
                    yaxis_title="DPA (R$)",
                    barmode='group',
                    hovermode='x unified',
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font=dict(family="Inter", size=12, color="#4a5568"),
                    xaxis=dict(showgrid=False, linecolor='#e2e8f0'),
                    yaxis=dict(gridcolor='#f0f0f0', linecolor='#e2e8f0'),
                    legend=dict(font=dict(family="Inter", size=11), bgcolor='rgba(255,255,255,0.8)', bordercolor='#e2e8f0', borderwidth=1, orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(l=10, r=10, t=60, b=40)
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
                
                # v4.2: highlight dark-mode safe nas linhas com Margem Mediana positiva
                # Usa border-left e cor de texto em vez de background hardcoded
                # para compatibilidade com dark e light mode
                def highlight_margem_mediana(row):
                    margem_val = row.get('Margem de Segurança Mediana (%)', -100)
                    if isinstance(margem_val, (int, float)) and margem_val > 0:
                        return ['border-left: 3px solid #3B6D11'] + [''] * (len(row) - 1)
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
            # Paleta pastel moderna para os graficos
            PALETA_CORES = [
                '#7EB8F7', '#82D9C5', '#B5A8F5', '#F7A08C', '#F7D07E',
                '#85CBA7', '#F5A6C8', '#A8C8F0', '#F0C98A', '#98D4D4',
                '#C5B4E8', '#F7B89A', '#A8DDB5', '#F5C8A0', '#90C8E8',
                '#D4A8D4', '#B8E0B8', '#F0D4A0', '#A8C4E0', '#D4C8A8'
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
                        plot_bgcolor='#ffffff',
                        paper_bgcolor='#ffffff',
                        font=dict(family="Inter", size=12, color="#4a5568"),
                        xaxis=dict(gridcolor='#f0f0f0', linecolor='#e2e8f0'),
                        yaxis=dict(showgrid=False, linecolor='#e2e8f0'),
                        margin=dict(l=10, r=80, t=30, b=40)
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
                    line=dict(color='#4361ee', width=2.5),
                    marker=dict(size=7, color='#4361ee', line=dict(color='white', width=1.5)),
                    yaxis='y2'
                ))

                fig_mensal.update_layout(
                    xaxis_title="Mês",
                    yaxis=dict(title="Recebimento no Mês (R$)", showgrid=True, gridcolor='#f0f0f0', linecolor='#e2e8f0'),
                    yaxis2=dict(title="Acumulado no Ano (R$)", overlaying='y', side='right', showgrid=False, linecolor='#e2e8f0'),
                    barmode='stack',
                    hovermode='x unified',
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font=dict(family="Inter", size=12, color="#4a5568"),
                    xaxis=dict(showgrid=False, linecolor='#e2e8f0'),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(family="Inter", size=11), bgcolor='rgba(255,255,255,0.8)', bordercolor='#e2e8f0', borderwidth=1),
                    margin=dict(l=10, r=10, t=50, b=40)
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

                # Tabela moderna dark-mode safe via st_components.html
                # Usa variáveis CSS nativas para compatibilidade com dark/light mode
                # Barra de progresso inline por linha, medalhas no top 3
                _linhas_ranking = ""
                for pos, r in enumerate(ranking_meses):
                    medalha   = MEDALHAS.get(pos, f"{pos + 1}º")
                    pct       = (r['total'] / total_ano_ranking * 100) if total_ano_ranking > 0 else 0
                    valor_fmt = f"R$ {r['total']:_.2f}".replace("_", ".").replace(".", ",", 1) if r['total'] > 0 else "R$ 0,00"
                    # Fundo sutil para top 3
                    if pos == 0:
                        bg = "background:rgba(186,117,23,0.08);"
                    elif pos == 1:
                        bg = "background:rgba(120,120,120,0.07);"
                    elif pos == 2:
                        bg = "background:rgba(186,117,23,0.05);"
                    else:
                        bg = ""
                    barra_pct = int(pct)
                    _linhas_ranking += (
                        f"<tr style='{bg}border-top:0.5px solid var(--sep);'>"
                        f"<td style='padding:9px 12px;text-align:center;font-size:15px;'>{medalha}</td>"
                        f"<td style='padding:9px 12px;font-weight:500;'>{r['mes_nome']}</td>"
                        f"<td style='padding:9px 12px;text-align:right;font-weight:500;'>{valor_fmt}</td>"
                        f"<td style='padding:9px 12px;'>"
                        f"<div style='display:flex;align-items:center;gap:8px;'>"
                        f"<div style='flex:1;height:5px;background:var(--bar-bg);border-radius:99px;overflow:hidden;'>"
                        f"<div style='width:{barra_pct}%;height:100%;background:var(--bar-fg);border-radius:99px;'></div>"
                        f"</div>"
                        f"<span style='font-size:11px;color:var(--txt-sec);min-width:32px;'>{pct:.1f}%</span>"
                        f"</div></td>"
                        f"<td style='padding:9px 12px;font-size:12px;color:var(--txt-sec);'>{r['tickers_pagam']}</td>"
                        f"</tr>"
                    )
                total_fmt = f"R$ {total_ano_ranking:_.2f}".replace("_", ".").replace(".", ",", 1)
                _html_ranking = f"""<style>
                .rank-wrap{{font-family:sans-serif;font-size:13px;}}
                @media(prefers-color-scheme:dark){{
                    .rank-wrap{{--sep:rgba(255,255,255,0.1);--hd-bg:#1e2130;--hd-fg:#a0aec0;
                        --ft-bg:rgba(255,255,255,0.05);--txt:#e2e8f0;--txt-sec:#718096;
                        --bar-bg:rgba(255,255,255,0.1);--bar-fg:#4361ee;}}
                }}
                @media(prefers-color-scheme:light){{
                    .rank-wrap{{--sep:rgba(0,0,0,0.08);--hd-bg:#f7f9ff;--hd-fg:#718096;
                        --ft-bg:#f7f9ff;--txt:#1a1f36;--txt-sec:#718096;
                        --bar-bg:rgba(0,0,0,0.08);--bar-fg:#4361ee;}}
                }}
                .rank-wrap table{{width:100%;border-collapse:collapse;color:var(--txt);}}
                .rank-wrap th{{padding:10px 12px;font-weight:500;color:var(--hd-fg);
                    background:var(--hd-bg);border-bottom:0.5px solid var(--sep);font-size:12px;}}
                </style>
                <div class="rank-wrap" style="border:0.5px solid rgba(128,128,128,0.2);
                    border-radius:10px;overflow:hidden;margin-bottom:16px;">
                  <table>
                    <thead><tr>
                      <th style="text-align:center;width:52px;">#</th>
                      <th style="text-align:left;">Mês</th>
                      <th style="text-align:right;">Total (R$)</th>
                      <th style="text-align:left;min-width:140px;">% do ano</th>
                      <th style="text-align:left;">Tickers que pagam</th>
                    </tr></thead>
                    <tbody>{_linhas_ranking}</tbody>
                    <tfoot><tr style="background:var(--ft-bg);border-top:0.5px solid var(--sep);font-weight:500;">
                      <td colspan="2" style="padding:9px 12px;">Total anual</td>
                      <td style="padding:9px 12px;text-align:right;">{total_fmt}</td>
                      <td style="padding:9px 12px;font-size:11px;color:var(--txt-sec);">100%</td>
                      <td></td>
                    </tr></tfoot>
                  </table>
                </div>"""
                # Altura dinâmica: 12 linhas + cabeçalho + rodapé
                _altura_ranking = 42 + (len(ranking_meses) * 38) + 42
                st_components.html(_html_ranking, height=_altura_ranking)

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
                # EXPORTAR SIMULACAO + IMPRIMIR
                # ---------------------------------------------------------------
                st.markdown("---")
                st.markdown("#### 💾 Exportar e Imprimir")

                nome_simulacao = st.text_input(
                    "Nome da simulação",
                    value=f"Simulacao_{_agora_sp().strftime('%Y%m%d')}",
                    key="nome_simulacao",
                    help="Digite um nome para identificar esta simulação nos arquivos exportados"
                )

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

                # --- Botoes Excel, CSV e Imprimir em linha unica ---
                col_xlsx, col_csv, col_pdf = st.columns([1, 1, 1])

                # Botao Excel
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
                            label="📥 Excel (.xlsx)",
                            data=buffer_excel,
                            file_name=nome_xlsx,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help=f"Simulação '{nome_simulacao}' — {data_hora_sim}",
                            use_container_width=True
                        )
                    except ImportError:
                        st.warning(
                            "⚠️ **Excel indisponível:** a biblioteca `openpyxl` não está instalada. "
                            "Adicione `openpyxl` ao `requirements.txt` e faça novo deploy para habilitar."
                        )

                # Botao CSV — sem dependencia extra, sempre disponivel
                with col_csv:
                    # CSV unico: metadados + separador + projecao por ticker
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
                        label="📄 CSV (.csv)",
                        data=csv_completo.encode('utf-8-sig'),  # utf-8-sig para Excel BR reconhecer acentos
                        file_name=nome_csv,
                        mime="text/csv",
                        help=f"Simulação '{nome_simulacao}' — {data_hora_sim}",
                        use_container_width=True
                    )

                # Botao Imprimir — mesma linha dos downloads.
                # Usa st_components.html() para garantir window.parent.print()
                # no contexto correto (local e Render).
                # CSS @media print oculta sidebar, header e demais tabs.
                with col_pdf:
                    _html_print = (
                        '<style>'
                        '@media print {'
                        '[data-testid="stSidebar"],'
                        '[data-testid="stHeader"],'
                        '[data-testid="stToolbar"],'
                        '[data-testid="stDecoration"],'
                        '[data-testid="stStatusWidget"],'
                        'button[kind="primary"],'
                        '.stTabs [data-baseweb="tab-list"],'
                        'footer { display: none !important; }'
                        '[data-testid="stAppViewContainer"] { margin: 0 !important; }'
                        '[data-testid="block-container"] { padding: 0 !important; }'
                        'table, .stDataFrame { page-break-inside: avoid; }'
                        '}'
                        '</style>'
                        '<button onclick="window.parent.print()" '
                        'style="background:#ffffff;color:#4361ee;border:1.5px solid #4361ee;'
                        'border-radius:8px;padding:6px 0;font-size:13px;font-weight:500;'
                        'cursor:pointer;width:100%;box-shadow:0 1px 4px rgba(67,97,238,0.1);">'
                        '🖨️ Imprimir / PDF'
                        '</button>'
                    )
                    st_components.html(_html_print, height=42)

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
        
        # ==============================================================================
        # Tab 5: Avaliação Gordon | Lynch
        # ==============================================================================
        with tab5:
            st.subheader("🏦 Avaliação Gordon | Lynch")

            # --- Parâmetros configuráveis pelo usuário ---
            st.markdown("#### ⚙️ Parâmetros do Modelo")
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                taxa_crescimento_gordon = st.number_input(
                    "Taxa de Crescimento dos Dividendos (g) % a.a.",
                    value=5.0,
                    min_value=0.1,
                    max_value=30.0,
                    step=0.5,
                    help=(
                        "Taxa anual de crescimento esperada dos dividendos. "
                        "5% a.a. é razoável para empresas maduras brasileiras. "
                        "Pode ser ajustado conforme expectativa macroeconômica."
                    ),
                    key="gordon_g"
                )
            with col_g2:
                taxa_selic_gordon = st.number_input(
                    "Taxa de Desconto (SELIC) % a.a.",
                    value=14.75,
                    min_value=1.0,
                    max_value=50.0,
                    step=0.25,
                    help=(
                        "Taxa de desconto para trazer os dividendos a valor presente. "
                        "Utiliza a SELIC como referência (14,75% a.a. em vigor). "
                        "Pode ser atualizada conforme decisões do COPOM."
                    ),
                    key="gordon_k"
                )

            g_dec  = taxa_crescimento_gordon / 100.0
            k_dec  = taxa_selic_gordon / 100.0

            # Valida k > g (condição obrigatória do Modelo de Gordon)
            if k_dec <= g_dec:
                st.error(
                    "❌ **Erro de parâmetros:** A taxa de desconto (SELIC) deve ser "
                    "MAIOR que a taxa de crescimento (g) para o Modelo de Gordon ser aplicável. "
                    f"Ajuste os valores: k={taxa_selic_gordon}% deve ser > g={taxa_crescimento_gordon}%."
                )
            else:
                # ---------------------------------------------------------------
                # Setores NÃO recomendados para o Modelo de Gordon
                # ---------------------------------------------------------------
                SETORES_NAO_INDICADOS = {
                    # Cíclicos — receita e lucro oscilam fortemente com o ciclo econômico
                    'Basic Materials':      'Empresa de materiais básicos (cíclica) — dividendos instáveis',
                    'Energy':               'Setor de energia/petróleo (cíclico) — dividendos voláteis',
                    'Consumer Cyclical':    'Setor cíclico ao consumo — dividendos irregulares',
                    'Industrials':          'Setor industrial (pode ser cíclico) — avalie histórico',
                    # Crescimento — retêm lucro para reinvestir, não distribuem dividendos consistentes
                    'Technology':           'Empresa de tecnologia — foco em crescimento, baixo DY',
                    'Communication Services': 'Setor de comunicações/tech — dividendos geralmente baixos',
                    'Healthcare':           'Setor de saúde — crescimento, dividendos inconsistentes',
                    # Real Estate — FIIs têm modelo próprio (FFO), Gordon não se aplica diretamente
                    'Real Estate':          'FII/Imóveis — use FFO Yield; Gordon não se aplica diretamente',
                }

                # ---------------------------------------------------------------
                # Calcula Valor Intrínseco Gordon para cada ticker com DPA válido
                # ---------------------------------------------------------------
                resultados_gordon = []

                for row in data_estatisticas:
                    tk         = row['Ticker']
                    mediana    = row.get('Mediana DPA', 0.0)
                    preco      = row.get('Preço Atual', 0.0)
                    anos_cons  = row.get('Anos Consecutivos', 0)
                    cagr_dpa_r = row.get('CAGR DPA 5a (%)', 0.0)

                    # Busca setor nos indicadores
                    setor_raw = None
                    for ind in data_indicadores:
                        if ind.get('Ticker') == tk:
                            setor_raw = ind.get('Setor')
                            break

                    # Verifica se setor é problemático para Gordon
                    aviso_setor = SETORES_NAO_INDICADOS.get(setor_raw or '', None)

                    # Só calcula Gordon se há DPA válido (mediana > 0)
                    if mediana <= 0 or preco <= 0:
                        resultados_gordon.append({
                            'Ticker':              tk,
                            'Mediana DPA (R$)':    mediana,
                            'Preço Atual (R$)':    preco,
                            'D1 Projetado (R$)':   0.0,
                            'Valor Gordon (R$)':   0.0,
                            'Margem Gordon (%)':   None,
                            'Sinal':               '⚪ Sem dados de dividendos',
                            'Anos Consecutivos':   anos_cons,
                            'CAGR DPA (%)':        cagr_dpa_r,
                            'Aviso Setor':         aviso_setor,
                            'Aplicavel':           False
                        })
                        continue

                    # D1 = DPA próximo período = Mediana DPA × (1 + g)
                    d1 = mediana * (1 + g_dec)

                    # P = D1 / (k - g)  — Fórmula de Gordon
                    valor_gordon = d1 / (k_dec - g_dec)

                    # Margem em relação ao preço atual
                    margem_gordon = ((valor_gordon / preco) - 1) * 100 if preco > 0 else None

                    # Sinal de compra/venda
                    if margem_gordon is None:
                        sinal = '⚪ Sem preço'
                    elif aviso_setor:
                        sinal = '⚠️ Setor inapropriado'
                    elif margem_gordon >= 20:
                        sinal = '🟢 Forte oportunidade'
                    elif margem_gordon >= 5:
                        sinal = '🔵 Possível oportunidade'
                    elif margem_gordon >= -5:
                        sinal = '🟡 Preço justo'
                    elif margem_gordon >= -20:
                        sinal = '🟠 Levemente caro'
                    else:
                        sinal = '🔴 Caro pelo Gordon'

                    resultados_gordon.append({
                        'Ticker':              tk,
                        'Mediana DPA (R$)':    round(mediana, 4),
                        'Preço Atual (R$)':    round(preco, 2),
                        'D1 Projetado (R$)':   round(d1, 4),
                        'Valor Gordon (R$)':   round(valor_gordon, 2),
                        'Margem Gordon (%)':   round(margem_gordon, 2) if margem_gordon is not None else None,
                        'Sinal':               sinal,
                        'Anos Consecutivos':   anos_cons,
                        'CAGR DPA (%)':        cagr_dpa_r,
                        'Aviso Setor':         aviso_setor,
                        'Aplicavel':           (aviso_setor is None and mediana > 0)
                    })

                df_gordon = pd.DataFrame(resultados_gordon)

                # ---------------------------------------------------------------
                # Aviso sobre setores inadequados
                # ---------------------------------------------------------------
                tickers_aviso = df_gordon[df_gordon['Aviso Setor'].notna()]
                if not tickers_aviso.empty:
                    avisos_unicos = tickers_aviso[['Ticker', 'Aviso Setor']].drop_duplicates()
                    linhas_aviso = [f"**{r['Ticker']}**: {r['Aviso Setor']}" for _, r in avisos_unicos.iterrows()]
                    st.warning(
                        "⚠️ **Atenção — setores possivelmente inapropriados para o Modelo de Gordon:**\n\n"
                        + "\n\n".join(linhas_aviso)
                        + "\n\nO Modelo de Gordon funciona melhor para empresas **maduras, estáveis e com histórico "
                        "consistente de dividendos crescentes** (ex: utilities, bancos, seguradoras, elétricas). "
                        "Para setores cíclicos ou de crescimento, use outros métodos de valuation."
                    )

                # ---------------------------------------------------------------
                # Tabela principal — Resultado Gordon
                # ---------------------------------------------------------------
                st.markdown("#### 📊 Modelo de Gordon - Avaliação")

                if df_gordon.empty:
                    st.info("ℹ️ Nenhum dado disponível para o Modelo de Gordon.")
                else:
                    df_exib = df_gordon[[
                        'Ticker', 'Preço Atual (R$)', 'Mediana DPA (R$)',
                        'D1 Projetado (R$)', 'Valor Gordon (R$)',
                        'Margem Gordon (%)', 'Anos Consecutivos', 'CAGR DPA (%)', 'Sinal'
                    ]].copy()

                    # Ordena alfabeticamente por Ticker
                    df_exib = df_exib.sort_values(
                        'Ticker',
                        ascending=True
                    ).reset_index(drop=True)

                    # Função de highlight por linha (dark-mode safe)
                    def _highlight_gordon(row):
                        sinal_v = row.get('Sinal', '')
                        margem_v = row.get('Margem Gordon (%)', None)
                        if isinstance(margem_v, (int, float)) and margem_v >= 20:
                            return ['border-left: 4px solid #2D7D32'] + [''] * (len(row) - 1)
                        if isinstance(margem_v, (int, float)) and margem_v >= 5:
                            return ['border-left: 4px solid #1565C0'] + [''] * (len(row) - 1)
                        if '⚠️' in str(sinal_v):
                            return ['border-left: 4px solid #BA7517'] + [''] * (len(row) - 1)
                        if isinstance(margem_v, (int, float)) and margem_v < -20:
                            return ['border-left: 4px solid #A32D2D'] + [''] * (len(row) - 1)
                        return [''] * len(row)

                    def _cor_margem_gordon(val):
                        if not isinstance(val, (int, float)):
                            return ''
                        if val >= 20:
                            return 'color: #2D7D32; font-weight: 700;'
                        if val >= 5:
                            return 'color: #1565C0; font-weight: 600;'
                        if val >= -5:
                            return 'color: #BA7517; font-weight: 600;'
                        return 'color: #A32D2D; font-weight: 600;'

                    fmt_gordon = {
                        'Preço Atual (R$)':    'R$ {:.2f}',
                        'Mediana DPA (R$)':    'R$ {:.4f}',
                        'D1 Projetado (R$)':   'R$ {:.4f}',
                        'Valor Gordon (R$)':   'R$ {:.2f}',
                        'Margem Gordon (%)':   lambda x: f'{x:.2f}%' if isinstance(x, (int, float)) else '—',
                        'CAGR DPA (%)':        lambda x: f'{x:.2f}%' if isinstance(x, (int, float)) else '—',
                    }

                    st.dataframe(
                        df_exib.style
                            .apply(_highlight_gordon, axis=1)
                            .map(_cor_margem_gordon, subset=['Margem Gordon (%)'])
                            .format(fmt_gordon),
                        use_container_width=True,
                        hide_index=True
                    )

                    # ---------------------------------------------------------------
                    # Cards de Oportunidades Gordon
                    # ---------------------------------------------------------------
                    oport_gordon = df_gordon[
                        (df_gordon['Margem Gordon (%)'].notna()) &
                        (df_gordon['Margem Gordon (%)'] >= 5) &
                        (df_gordon['Aplicavel'] == True)
                    ].sort_values('Margem Gordon (%)', ascending=False)

                    if not oport_gordon.empty:
                        st.markdown("---")
                        st.markdown("### 🔔 Oportunidades Identificadas pelo Modelo de Gordon")

                        cols_por_linha = 3
                        lista_oport = oport_gordon.to_dict('records')

                        for i in range(0, len(lista_oport), cols_por_linha):
                            grupo = lista_oport[i:i + cols_por_linha]
                            colunas_card = st.columns(len(grupo))

                            for col_c, op in zip(colunas_card, grupo):
                                margem_c   = op.get('Margem Gordon (%)', 0.0)
                                preco_c    = op.get('Preço Atual (R$)', 0.0)
                                gordon_c   = op.get('Valor Gordon (R$)', 0.0)
                                tk_c       = op.get('Ticker', '')
                                cagr_c     = op.get('CAGR DPA (%)', 0.0)
                                anos_c     = op.get('Anos Consecutivos', 0)

                                if margem_c >= 20:
                                    cor_borda = '#2D7D32'
                                    cor_fundo = 'rgba(45,125,50,0.07)'
                                    icone_c   = '🟢'
                                else:
                                    cor_borda = '#1565C0'
                                    cor_fundo = 'rgba(21,101,192,0.07)'
                                    icone_c   = '🔵'

                                with col_c:
                                    html_card_g = (
                                        "<div style='"
                                        "background-color:" + cor_fundo + ";"
                                        "border:0.5px solid " + cor_borda + ";"
                                        "border-left:4px solid " + cor_borda + ";"
                                        "border-radius:10px;"
                                        "padding:14px 16px;"
                                        "margin-bottom:12px;'>"
                                        "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>"
                                        "<span style='font-size:15px;font-weight:700;'>" + icone_c + " " + tk_c + "</span>"
                                        "<span style='font-size:12px;font-weight:700;color:" + cor_borda + ";'>+" + f"{margem_c:.1f}" + "%</span>"
                                        "</div>"
                                        "<div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;'>"
                                        "<span>Atual <strong>R$ " + f"{preco_c:.2f}" + "</strong></span>"
                                        "<span>Gordon <strong style='color:" + cor_borda + ";'>R$ " + f"{gordon_c:.2f}" + "</strong></span>"
                                        "</div>"
                                        "<div style='font-size:11px;color:#718096;margin-top:4px;'>"
                                        "CAGR DPA: <strong>" + f"{cagr_c:.1f}%" + "</strong> · "
                                        + str(anos_c) + " anos consecutivos"
                                        "</div>"
                                        "</div>"
                                    )
                                    st.markdown(html_card_g, unsafe_allow_html=True)
                    else:
                        st.info(
                            "ℹ️ Nenhuma oportunidade identificada com margem positiva ≥ 5% pelo Modelo de Gordon "
                            "nos tickers analisados. Os preços atuais podem estar acima do valor intrínseco calculado."
                        )

                    # ---------------------------------------------------------------
                    # Diagnóstico e Estratégia de Longo Prazo por Ticker
                    # ---------------------------------------------------------------
                    st.markdown("---")
                    st.markdown("### 🔬 Diagnóstico e Estratégia de Longo Prazo")
                    st.caption(
                        "Análise individual de cada ação com base no Modelo de Gordon, "
                        "contexto de mercado e orientações para estratégia de longo prazo. "
                        "Clique em cada ação para expandir o diagnóstico completo."
                    )

                    for _, row_g in df_gordon.sort_values('Ticker').iterrows():
                        tk_o      = row_g['Ticker']
                        margem_o  = row_g['Margem Gordon (%)']
                        sinal_o   = row_g['Sinal']
                        gordon_o  = row_g['Valor Gordon (R$)']
                        preco_o   = row_g['Preço Atual (R$)']
                        anos_o    = row_g['Anos Consecutivos']
                        cagr_o    = row_g['CAGR DPA (%)']
                        mediana_o = row_g['Mediana DPA (R$)']
                        aviso_o   = row_g['Aviso Setor']
                        aplicavel = row_g['Aplicavel']

                        margem_str = f"{margem_o:.2f}%" if isinstance(margem_o, (int, float)) else "N/D"
                        gordon_str = f"R$ {gordon_o:.2f}" if isinstance(gordon_o, (int, float)) and gordon_o > 0 else "N/D"

                        label_exp = f"{sinal_o}  |  {tk_o}  —  Valor Gordon: {gordon_str}  |  Margem: {margem_str}"

                        with st.expander(label_exp, expanded=False):

                            if aviso_o:
                                st.warning(f"⚠️ **Atenção ao setor:** {aviso_o}")

                            if not aplicavel or gordon_o == 0 or not isinstance(preco_o, (int, float)) or preco_o <= 0:
                                st.markdown(
                                    f"**{tk_o}** não possui histórico de dividendos suficiente para o Modelo de Gordon. "
                                    "O modelo exige que a empresa pague dividendos de forma consistente e crescente. "
                                    "Se a empresa não paga dividendos regularmente, utilize outros métodos de valuation, "
                                    "como P/L, P/VP ou Fluxo de Caixa Descontado (DCF)."
                                )
                            else:
                                # --------------------------------------------------
                                # Valores derivados para o diagnóstico
                                # --------------------------------------------------
                                diferenca_abs   = preco_o - gordon_o
                                acao_vs_gordon  = "acima" if preco_o > gordon_o else "abaixo"
                                pct_vs_gordon   = abs(margem_o) if isinstance(margem_o, (int, float)) else 0.0
                                dy_atual        = (mediana_o / preco_o * 100) if preco_o > 0 else 0.0

                                # Zona de suporte sugerida (±8% em torno do Valor Gordon)
                                zona_inferior   = gordon_o * 0.92
                                zona_superior   = gordon_o * 1.05

                                # Descrição do CAGR DPA vs premissa do modelo
                                if isinstance(cagr_o, (int, float)) and cagr_o > 0:
                                    cagr_vs_g = cagr_o - taxa_crescimento_gordon
                                    if cagr_vs_g >= 3:
                                        cagr_comentario = (
                                            f"O crescimento histórico dos dividendos ({cagr_o:.1f}% a.a.) é "
                                            f"**superior** à taxa assumida no modelo ({taxa_crescimento_gordon}% a.a.), "
                                            f"sugerindo que o Valor Gordon calculado pode ser **conservador**."
                                        )
                                    elif cagr_vs_g >= -2:
                                        cagr_comentario = (
                                            f"O crescimento histórico dos dividendos ({cagr_o:.1f}% a.a.) está "
                                            f"**alinhado** com a premissa do modelo ({taxa_crescimento_gordon}% a.a.). "
                                            f"O Valor Gordon reflete bem o histórico da empresa."
                                        )
                                    else:
                                        cagr_comentario = (
                                            f"O crescimento histórico dos dividendos ({cagr_o:.1f}% a.a.) está "
                                            f"**abaixo** da premissa do modelo ({taxa_crescimento_gordon}% a.a.). "
                                            f"O Valor Gordon pode estar **superestimado** — revise a taxa 'g'."
                                        )
                                elif isinstance(cagr_o, (int, float)) and cagr_o <= 0:
                                    cagr_comentario = (
                                        f"Os dividendos apresentaram **queda ou estagnação** nos últimos anos "
                                        f"(CAGR DPA: {cagr_o:.1f}% a.a.). O Modelo de Gordon assume crescimento positivo "
                                        f"— interprete este resultado com cautela adicional."
                                    )
                                else:
                                    cagr_comentario = "CAGR DPA não disponível para este ticker."

                                # Anos consecutivos — texto de confiabilidade
                                if anos_o >= 15:
                                    anos_label = f"**{anos_o} anos consecutivos** de pagamentos — confiabilidade máxima ⭐⭐⭐"
                                elif anos_o >= 10:
                                    anos_label = f"**{anos_o} anos consecutivos** de pagamentos — alta confiabilidade ⭐⭐"
                                elif anos_o >= 5:
                                    anos_label = f"**{anos_o} anos consecutivos** de pagamentos — confiabilidade moderada ⭐"
                                elif anos_o > 0:
                                    anos_label = f"**{anos_o} anos consecutivos** de pagamentos — histórico curto, monitore ⚠️"
                                else:
                                    anos_label = "Histórico de pagamentos consecutivos não identificado ⚠️"

                                # --------------------------------------------------
                                # Bloco 1 — Preço vs Valor (análise técnica central)
                                # --------------------------------------------------
                                st.markdown(
                                    "<div style='"
                                    "border-left: 4px solid #4361ee;"
                                    "background: rgba(67,97,238,0.07);"
                                    "border-radius: 0 8px 8px 0;"
                                    "padding: 14px 18px;"
                                    "margin-bottom: 14px;"
                                    "'>"
                                    "<span style='font-size:13px;font-weight:700;letter-spacing:0.3px;'>"
                                    "📍 1. Preço vs. Valor Intrínseco"
                                    "</span>"
                                    "</div>",
                                    unsafe_allow_html=True
                                )

                                if preco_o > gordon_o:
                                    # Ação acima do valor Gordon
                                    pct_agio = ((preco_o / gordon_o) - 1) * 100
                                    st.markdown(
                                        f"A cotação atual (**R$ {preco_o:.2f}**) está **{pct_agio:.1f}% acima** "
                                        f"do Valor Gordon calculado (**{gordon_str}**), uma diferença de "
                                        f"**R$ {diferenca_abs:.2f}** por ação.\n\n"
                                        f"Isso significa que, sob a ótica da SELIC a {taxa_selic_gordon}% e crescimento "
                                        f"de dividendos de {taxa_crescimento_gordon}% a.a., o mercado está pagando "
                                        f"um **ágio pela previsibilidade da empresa** ou já precificando uma queda "
                                        f"futura nos juros — o que aumentaria o Valor Gordon automaticamente.\n\n"
                                        f"**Yield on Cost atual:** comprando a R$ {preco_o:.2f}, seu DY imediato "
                                        f"seria de aproximadamente **{dy_atual:.2f}% a.a.** — "
                                        + ("abaixo da SELIC, indicando que o investidor aceita retorno menor pela "
                                           "qualidade e previsibilidade dos dividendos."
                                           if dy_atual < taxa_selic_gordon else
                                           "acima da SELIC, o que é positivo para um ativo de renda variável.")
                                    )
                                else:
                                    # Ação abaixo do valor Gordon (desconto)
                                    pct_desconto = ((gordon_o / preco_o) - 1) * 100
                                    st.markdown(
                                        f"A cotação atual (**R$ {preco_o:.2f}**) está **{pct_desconto:.1f}% abaixo** "
                                        f"do Valor Gordon calculado (**{gordon_str}**), uma diferença de "
                                        f"**R$ {abs(diferenca_abs):.2f}** por ação.\n\n"
                                        f"O mercado está oferecendo esta ação com **desconto** em relação ao seu "
                                        f"valor intrínseco pelo modelo, considerando SELIC a {taxa_selic_gordon}% "
                                        f"e crescimento de dividendos de {taxa_crescimento_gordon}% a.a.\n\n"
                                        f"**Yield on Cost atual:** comprando a R$ {preco_o:.2f}, seu DY imediato "
                                        f"seria de aproximadamente **{dy_atual:.2f}% a.a.** — "
                                        + ("acima da SELIC, o que representa uma janela de eficiência importante "
                                           "para aportes em estratégia de longo prazo."
                                           if dy_atual >= taxa_selic_gordon else
                                           "ainda abaixo da SELIC, mas com margem de segurança expressiva no preço.")
                                    )

                                # --------------------------------------------------
                                # Bloco 2 — Estratégia de Carregamento / Acumulação
                                # --------------------------------------------------
                                st.markdown(
                                    "<div style='"
                                    "border-left: 4px solid #4361ee;"
                                    "background: rgba(67,97,238,0.07);"
                                    "border-radius: 0 8px 8px 0;"
                                    "padding: 14px 18px;"
                                    "margin-bottom: 14px;margin-top:10px;"
                                    "'>"
                                    "<span style='font-size:13px;font-weight:700;letter-spacing:0.3px;'>"
                                    "📈 2. Estratégia de Longo Prazo (Carregamento Perpétuo)"
                                    "</span>"
                                    "</div>",
                                    unsafe_allow_html=True
                                )

                                if preco_o > gordon_o:
                                    st.markdown(
                                        f"**A Visão de Carregamento:** Para uma estratégia buy-and-hold perpétua, "
                                        f"o preço de entrada impacta diretamente o seu **Yield on Cost (YoC) futuro**.\n\n"
                                        f"- Comprando a **R$ {preco_o:.2f}**, seu retorno imediato em dividendos seria de "
                                        f"aproximadamente **{dy_atual:.2f}% a.a.**\n"
                                        f"- A **zona de eficiência de aporte** para esta ação, pelo modelo, situa-se "
                                        f"entre **R$ {zona_inferior:.2f}** e **R$ {zona_superior:.2f}** — faixa onde "
                                        f"o preço se aproxima ou fica abaixo do Valor Gordon calculado.\n"
                                        f"- Para quem busca **acumular**, o ideal é aguardar janelas onde o preço "
                                        f"recue para essa zona, maximizando a eficiência dos novos aportes e o "
                                        f"Yield on Cost da posição total."
                                    )
                                else:
                                    st.markdown(
                                        f"**Oportunidade de Acumulação:** O preço atual (**R$ {preco_o:.2f}**) "
                                        f"encontra-se dentro ou abaixo da **zona de eficiência de aporte** "
                                        f"(R$ {zona_inferior:.2f} – R$ {zona_superior:.2f}) calculada pelo modelo.\n\n"
                                        f"- Comprando a **R$ {preco_o:.2f}**, seu retorno imediato em dividendos seria de "
                                        f"aproximadamente **{dy_atual:.2f}% a.a.**\n"
                                        f"- Para estratégia buy-and-hold, este é um ponto favorável para **novos aportes**, "
                                        f"pois o preço oferece desconto sobre o valor intrínseco pelos dividendos.\n"
                                        f"- Se o crescimento de dividendos ({taxa_crescimento_gordon}% a.a.) se confirmar, "
                                        f"o YoC tende a crescer ano a ano, protegendo a renda passiva da inflação."
                                    )

                                # --------------------------------------------------
                                # Bloco 3 — Análise de Crescimento de Dividendos
                                # --------------------------------------------------
                                st.markdown(
                                    "<div style='"
                                    "border-left: 4px solid #4361ee;"
                                    "background: rgba(67,97,238,0.07);"
                                    "border-radius: 0 8px 8px 0;"
                                    "padding: 14px 18px;"
                                    "margin-bottom: 14px;margin-top:10px;"
                                    "'>"
                                    "<span style='font-size:13px;font-weight:700;letter-spacing:0.3px;'>"
                                    "📊 3. Análise do Crescimento de Dividendos"
                                    "</span>"
                                    "</div>",
                                    unsafe_allow_html=True
                                )
                                st.markdown(
                                    f"{cagr_comentario}\n\n"
                                    f"Histórico: {anos_label}\n\n"
                                    f"**Mediana DPA utilizada:** R$ {mediana_o:.4f} "
                                    f"(valor central dos dividendos dos últimos {anos_input} anos — "
                                    f"resistente a pagamentos extraordinários não-recorrentes)."
                                )

                                # --------------------------------------------------
                                # Bloco 4 — Conclusão e Veredito
                                # --------------------------------------------------
                                st.markdown(
                                    "<div style='"
                                    "border-left: 4px solid #4361ee;"
                                    "background: rgba(67,97,238,0.07);"
                                    "border-radius: 0 8px 8px 0;"
                                    "padding: 14px 18px;"
                                    "margin-bottom: 14px;margin-top:10px;"
                                    "'>"
                                    "<span style='font-size:13px;font-weight:700;letter-spacing:0.3px;'>"
                                    "✅ 4. Conclusão e Veredito"
                                    "</span>"
                                    "</div>",
                                    unsafe_allow_html=True
                                )

                                if isinstance(margem_o, (int, float)):
                                    if margem_o >= 20:
                                        conclusao = (
                                            f"**{tk_o} apresenta forte desconto pelo Modelo de Gordon** "
                                            f"(margem de **+{margem_o:.1f}%**). "
                                            f"Com {anos_o} anos consecutivos de dividendos e CAGR DPA de "
                                            f"{cagr_o:.1f}% a.a., a empresa demonstra solidez e histórico robusto. "
                                            f"**Veredito:** Candidata de alta prioridade para análise de compra/aporte — "
                                            f"verifique os fundamentos complementares (endividamento, payout, ROE) "
                                            f"antes da decisão final."
                                        )
                                    elif margem_o >= 5:
                                        conclusao = (
                                            f"**{tk_o} apresenta desconto moderado pelo Modelo de Gordon** "
                                            f"(margem de **+{margem_o:.1f}%**). "
                                            f"Há valor, mas a margem de segurança não é expressiva. "
                                            f"**Veredito:** Monitore o papel — aportes parciais são razoáveis, "
                                            f"mas aguardar um recuo adicional pode maximizar a eficiência da posição."
                                        )
                                    elif margem_o >= -5:
                                        conclusao = (
                                            f"**{tk_o} negocia próximo do preço justo pelo Modelo de Gordon** "
                                            f"(margem de **{margem_o:.1f}%**). "
                                            f"Não há grande desconto, mas também não está cara. "
                                            f"**Veredito:** Para quem já possui posição, é razoável manter. "
                                            f"Para novos aportes, aguardar recuo que ofereça margem de segurança "
                                            f"mais confortável (≥ +10%) é a postura mais conservadora."
                                        )
                                    elif margem_o >= -20:
                                        conclusao = (
                                            f"**{tk_o} está levemente acima do valor estimado pelo Modelo de Gordon** "
                                            f"(margem de **{margem_o:.1f}%**), porém não em patamar crítico. "
                                            f"**Veredito:** Evite novos aportes no preço atual. "
                                            f"Se os fundamentos da empresa são sólidos, a posição existente pode ser "
                                            f"mantida — mas aguarde correção antes de ampliar exposição."
                                        )
                                    else:
                                        conclusao = (
                                            f"**{tk_o} está significativamente acima do valor estimado pelo Modelo de Gordon** "
                                            f"(margem de **{margem_o:.1f}%**). O preço atual já embutia crescimento "
                                            f"elevado de dividendos ou queda expressiva dos juros. "
                                            f"**Veredito:** Risco elevado de rendimento (DY) aquém do esperado ao "
                                            f"preço atual. Reveja a tese de investimento e considere aguardar "
                                            f"correção relevante antes de novos aportes."
                                        )
                                else:
                                    conclusao = (
                                        f"**{tk_o}:** Dados insuficientes para gerar conclusão pelo Modelo de Gordon. "
                                        "Utilize indicadores complementares para esta ação."
                                    )

                                # Caixa de conclusão com cor adaptável a dark/light mode
                                if isinstance(margem_o, (int, float)) and margem_o >= 5:
                                    cor_conclusao_borda = "#2D7D32"
                                    cor_conclusao_bg    = "rgba(45,125,50,0.09)"
                                elif isinstance(margem_o, (int, float)) and margem_o >= -5:
                                    cor_conclusao_borda = "#BA7517"
                                    cor_conclusao_bg    = "rgba(186,117,23,0.09)"
                                else:
                                    cor_conclusao_borda = "#A32D2D"
                                    cor_conclusao_bg    = "rgba(163,45,45,0.09)"

                                st.markdown(
                                    f"<div style='"
                                    f"border:1px solid {cor_conclusao_borda};"
                                    f"border-left:5px solid {cor_conclusao_borda};"
                                    f"background:{cor_conclusao_bg};"
                                    f"border-radius:8px;"
                                    f"padding:16px 18px;"
                                    f"margin-top:4px;"
                                    f"'>"
                                    f"{conclusao}"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )

                    # ---------------------------------------------------------------
                    # Seção Explicativa — Modelo de Gordon e limitações
                    # ---------------------------------------------------------------
                    st.markdown("---")
                    st.markdown('<div class="explicacao-container">', unsafe_allow_html=True)
                    st.subheader("📚 O que é o Modelo de Gordon e como interpretar os resultados?")

                    st.write(
                        "**Modelo de Gordon (Dividend Discount Model — DDM):** "
                        "O Modelo de Gordon calcula o valor intrínseco de uma ação como a soma de todos os "
                        "seus dividendos futuros trazidos ao valor presente. Parte do princípio de que o "
                        "valor real de qualquer ativo é o fluxo de caixa que ele gera para o investidor — "
                        "no caso das ações de dividendos, os proventos recebidos ao longo do tempo. "
                        "É a ferramenta ideal para empresas maduras, estáveis e boas pagadoras de dividendos."
                    )
                    st.latex(
                        r"P = \frac{D_1}{k - g}"
                    )
                    st.write(
                        "Onde: **P** = Preço justo (Valor Intrínseco) | "
                        "**D₁** = Dividendo esperado no próximo período = Mediana DPA × (1 + g) | "
                        f"**k** = Taxa de desconto (SELIC = {taxa_selic_gordon}% a.a.) | "
                        f"**g** = Taxa de crescimento dos dividendos ({taxa_crescimento_gordon}% a.a.)"
                    )
                    st.latex(
                        r"D_1 = Mediana\ DPA \times (1 + g)"
                    )

                    st.write(
                        "**Por que usar a Mediana DPA?** "
                        "A Mediana é usada no lugar da Média pois é resistente a distorções causadas por "
                        "dividendos extraordinários não-recorrentes. Ela representa a capacidade típica e "
                        "recorrente de pagamento da empresa — critério mais conservador e realista para o modelo."
                    )

                    st.write(
                        "**Margem de Segurança Gordon (%):** "
                        "Indica o quanto o Valor Gordon está acima ou abaixo do preço atual de mercado. "
                        "Margem positiva = ação com desconto (oportunidade). "
                        "Margem negativa = ação acima do valor justo (cara pelo modelo)."
                    )
                    st.latex(
                        r"Margem\ Gordon\ (\%) = \left( \frac{Valor\ Gordon}{Preço\ Atual} - 1 \right) \times 100"
                    )

                    st.write(
                        "**Condição fundamental do Modelo de Gordon:** "
                        "A taxa de desconto (k) DEVE ser maior que a taxa de crescimento dos dividendos (g). "
                        "Se k ≤ g, o modelo gera valores absurdos ou infinitos. "
                        "Com SELIC em 14,75% e g de 5%, a diferença (k - g = 9,75%) é o 'spread' que define "
                        "o múltiplo de dividendos no cálculo do valor intrínseco."
                    )

                    st.write(
                        "**Para quais empresas o Modelo de Gordon é mais adequado?**"
                    )
                    st.write(
                        "✅ **Ideal para:** Utilities (elétricas, saneamento), bancos consolidados, "
                        "seguradoras, empresas de concessão, holdings com dividendos estáveis e crescentes. "
                        "Em suma: empresas maduras, lucrativas e com política de dividendos previsível."
                    )
                    st.write(
                        "⚠️ **Use com cuidado:** Setor industrial e de construção civil — podem ser "
                        "parcialmente cíclicos. Avalie o histórico individual antes de confiar no modelo."
                    )
                    st.write(
                        "❌ **Não recomendado para:** "
                        "Empresas cíclicas (petróleo, mineração, materiais básicos) cujos dividendos oscilam "
                        "conforme o ciclo de commodities; startups e empresas de crescimento (tecnologia, saúde) "
                        "que retêm lucro para reinvestir; FIIs e imobiliárias (use FFO Yield); "
                        "empresas com histórico inconsistente ou muito curto de dividendos (< 3 anos); "
                        "empresas com payout acima de 100% (dividendos insustentáveis)."
                    )

                    st.write(
                        "**Limitações importantes:**"
                    )
                    st.write(
                        "• O modelo assume crescimento CONSTANTE dos dividendos para sempre — simplificação da realidade. "
                        "• A escolha de 'g' tem grande impacto no resultado: pequenas variações mudam o valor intrínseco significativamente. "
                        "• O modelo não captura mudanças bruscas na política de dividendos ou eventos extraordinários. "
                        "• Use sempre em conjunto com outros indicadores (P/L, P/VP, Bazin, Graham) para uma análise completa. "
                        "• O resultado é uma estimativa de valor, não uma garantia de preço futuro."
                    )

                    st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning(
            "⚠️ Não foi possível processar os dados. "
            "Verifique os tickers e tente novamente."
        )
