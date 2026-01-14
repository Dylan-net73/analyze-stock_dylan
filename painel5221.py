# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURAÇÃO E ESTILO ---
st.set_page_config(page_title="Análise de Ações e Indicadores Fundamentalistas", layout="wide")

st.markdown("""
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

# --- FUNÇÕES DE COLETA E LÓGICA ---

@st.cache_data
def get_seasonality_and_dividends(ticker_symbol: str, anos: int):
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period=f"{anos}y")
        dividends = ticker.dividends
        if hist.empty: return "N/A", "N/A"
        
        if not dividends.empty:
            div_meses = dividends.index.month.value_counts()
            meses_pagto = div_meses[div_meses >= (anos * 0.3)].index.sort_values()
            nomes_meses_pagto = [datetime(2000, m, 1).strftime('%b') for m in meses_pagto]
            col_pagto = ", ".join(nomes_meses_pagto)
        else: col_pagto = "Sem Div."

        hist['Month'] = hist.index.month
        mensal_media = hist.groupby(['Month', hist.index.year])['Close'].mean().unstack().mean(axis=1)
        meses_baratos = mensal_media.nsmallest(3).index.sort_values()
        nomes_meses_baratos = [datetime(2000, m, 1).strftime('%b') for m in meses_baratos]
        col_sazonalidade = ", ".join(nomes_meses_baratos)
        
        return col_pagto, col_sazonalidade
    except: return "Erro", "Erro"

@st.cache_data
def get_dpa_data(ticker_symbol: str, anos: int) -> tuple:
    fuso_horario = pytz.timezone('America/Sao_Paulo')
    end_date = datetime.now(fuso_horario)
    start_date = end_date - timedelta(days=anos * 365)
    try:
        ticker = yf.Ticker(ticker_symbol)
        dividends = ticker.dividends
        if dividends.empty: return pd.DataFrame(), 0.0, 0.0, ""
        div_filt = dividends.loc[start_date:end_date]
        div_ano = div_filt.resample('YE').sum().reset_index()
        div_ano.columns = ['Ano', 'DPA']
        div_ano['Ano'] = div_ano['Ano'].dt.year
        div_ano['Ticker'] = ticker_symbol 
        div_comp = div_ano[div_ano['Ano'] < datetime.now(fuso_horario).year].copy()
        return div_comp, float(div_comp['DPA'].mean()), float(div_comp['DPA'].median()), ""
    except: return pd.DataFrame(), 0.0, 0.0, ""

@st.cache_data
def get_fundamental_data(ticker_symbol: str) -> dict:
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        return {
            'Ticker': ticker_symbol, 
            'P/L': info.get('trailingPE'), 
            'P/VP': info.get('priceToBook'), 
            'ROE': info.get('returnOnEquity'), 
            'DY': info.get('dividendYield'), 
            'Setor': info.get('sector'),
            'Dívida/PL': float(info.get('debtToEquity', 0))/100 if info.get('debtToEquity') else None, 
            'Dívida/EBITDA': info.get('debtToEbitda'),
            'EV/EBITDA': info.get('enterpriseToEbitda'),
            'LPA': info.get('trailingEps'),
            'VPA': info.get('bookValue')
        }
    except: return {'Ticker': ticker_symbol}

@st.cache_data
def get_cagr_data(ticker_symbol: str, years: int = 5) -> dict:
    try:
        ticker = yf.Ticker(ticker_symbol)
        fin = ticker.financials
        def calc_cagr(df, keys):
            key = next((k for k in keys if k in df.index), None)
            if key is None: return None
            s = df.loc[key].sort_index().iloc[-years-1:].dropna()
            if len(s) < 2: return None
            return float(((s.iloc[-1] / s.iloc[0]) ** (1 / (len(s)-1))) - 1)
        return {'CAGR Receita 5a': calc_cagr(fin, ['Total Revenue']), 'CAGR Lucro 5a': calc_cagr(fin, ['Net Income'])}
    except: return {}

# --- SIDEBAR ---
with st.sidebar:
    st.header("Opções de Análise")
    ticker_input_val = "BBAS3.SA, BBSE3.SA, BRAP4.SA, BRSR6.SA, CMIG4.SA, CXSE3.SA, PETR4.SA, TAEE4.SA, UNIP6.SA"
    st.subheader("Entrada Manual")
    ticker_input = st.text_input("insira Tickers (separados por vírgula)", value=ticker_input_val)
    st.divider()
    st.subheader("Upload de Arquivo CSV")
    uploaded_file = st.file_uploader("Escolha o arquivo", type=['csv'], label_visibility="collapsed")
    if uploaded_file:
        try:
            df_csv = pd.read_csv(uploaded_file)
            if 'Ticker' in df_csv.columns:
                ticker_input = ",".join(df_csv['Ticker'].astype(str).tolist())
                st.success("Tickers carregados!")
        except Exception as e: st.error(f"Erro: {e}")
    st.divider()
    anos_input = st.number_input("Número de Anos para Análise", value=5, min_value=1)
    taxa_retorno_input = st.number_input("Taxa de Retorno Anual Desejada (%)", value=6.0, step=0.5)
    st.divider()
    max_pl_filter = st.number_input("P/L Máximo (0 para Desativar)", value=0.0)
    min_dy_filter = st.number_input("DY Mínimo (%) (0 para Desativar)", value=0.0)
    min_ms_filter = st.number_input("Margem de Segurança Mínima (%)", value=-100.0)

# --- EXECUÇÃO ---
st.title("Análise de Ações e Indicadores Fundamentalistas")

if st.button("Gerar Relatório"):
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    data_val, data_ind, data_stats, data_graf, alertas = [], [], [], [], []
    tax_ret = taxa_retorno_input / 100
    
    with st.spinner("Processando dados..."):
        def process(t):
            df_dpa, media, mediana, _ = get_dpa_data(t, anos_input)
            fund = get_fundamental_data(t)
            fund.update(get_cagr_data(t))
            pagtos, baratos = get_seasonality_and_dividends(t, anos_input)
            p_atual = 0.0
            try: p_atual = float(yf.Ticker(t).info.get('regularMarketPrice') or yf.Ticker(t).info.get('currentPrice') or 0.0)
            except: pass
            
            # Preço Teto Bazin
            teto = media / tax_ret if media > 0 else 0
            
            # Preço Justo de Graham
            lpa = fund.get('LPA')
            vpa = fund.get('VPA')
            if lpa and vpa and lpa > 0 and vpa > 0:
                p_graham = np.sqrt(22.5 * lpa * vpa)
            else:
                p_graham = 0.0

            ms = ((teto/p_atual)-1)*100 if p_atual > 0 and teto > 0 else -100
            alert = f"Oportunidade em {t}: Preço Atual R$ {p_atual:.2f} < Teto R$ {teto:.2f}" if ms > 0 else None
            return {
                'v': {'Ticker': t, 'Preço Atual': p_atual, 'Preço Teto (Bazin)': teto, 'Preço Graham': p_graham, 'Margem Segurança (%)': ms, 'Pagamento Dividendos': pagtos, 'Melhor mês para compra': baratos}, 
                'i': fund, 's': {'Ticker': t, 'Média DPA': media, 'Mediana DPA': mediana}, 'g': df_dpa, 'a': alert
            }
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(process, t) for t in tickers]
            for f in as_completed(futures):
                r = f.result()
                data_val.append(r['v']); data_ind.append(r['i']); data_stats.append(r['s']); alertas.append(r['a'])
                if not r['g'].empty: data_graf.append(r['g'])

    if data_val:
        df_v = pd.DataFrame(data_val)
        df_i = pd.DataFrame(data_ind)
        
        tem_financeiro = any(df_i['Setor'].str.contains('Financial', case=False, na=False))
        if tem_financeiro:
            st.info("ℹ️ **Nota sobre Instituições Financeiras:** Identificámos bancos ou seguradoras na sua análise. Para estas empresas, o indicador **Dívida/EBITDA** não é aplicável, pois o modelo de negócio baseia-se na intermediação financeira. Foque em métricas como ROE e P/VP.")

        tab1, tab2, tab3 = st.tabs(["💎 Valuation", "📊 Indicadores", "📈 Gráfico DPA"])
        
        with tab1:
            st.dataframe(df_v.style.map(lambda x: 'background-color: #e6ffed;' if isinstance(x, (int, float)) and x > 0 else '', subset=['Margem Segurança (%)']).format({'Preço Atual': 'R$ {:.2f}', 'Preço Teto (Bazin)': 'R$ {:.2f}', 'Preço Graham': 'R$ {:.2f}', 'Margem Segurança (%)': '{:.2f}%'}), use_container_width=True, hide_index=True)
        with tab2:
            cols_ind = ['Ticker', 'P/L', 'P/VP', 'ROE', 'DY', 'Dívida/PL', 'Dívida/EBITDA', 'EV/EBITDA', 'CAGR Receita 5a', 'CAGR Lucro 5a']
            st.dataframe(df_i[df_i['Ticker'].isin(df_v['Ticker'])][cols_ind], use_container_width=True, hide_index=True)
        with tab3:
            if data_graf:
                fig = go.Figure()
                for d in data_graf: fig.add_trace(go.Bar(x=d['Ano'], y=d['DPA'], name=d['Ticker'].iloc[0]))
                st.plotly_chart(fig, use_container_width=True)
                
                df_stats = pd.DataFrame(data_stats)
                df_stats['Preço Teto (Mediana)'] = df_stats['Mediana DPA'] / tax_ret if tax_ret > 0 else 0
                st.dataframe(df_stats.style.format({'Média DPA': '{:.2f}', 'Mediana DPA': '{:.2f}', 'Preço Teto (Mediana)': 'R$ {:.2f}'}), use_container_width=True, hide_index=True)

        # --- SEÇÃO DE EXPLICAÇÕES ---
        st.markdown('<div class="explicacao-container">', unsafe_allow_html=True)
        st.subheader("O que significam os indicadores e como são calculados?")
        
        st.write(f"**Sazonalidade (Melhor mês para compra):** O app analisa o histórico de preços dos últimos {anos_input} anos (conforme sua escolha na barra lateral), agrupando as cotações por mês. Ele calcula a média de preço de cada mês e identifica os 3 meses que apresentam os menores valores médios. Isso sugere períodos em que, historicamente, o ativo esteve mais barato, auxiliando na identificação de janelas de oportunidade para compra.")
        
        st.write(f"**Preço Teto (Média):** É o preço máximo que um investidor aceitaria pagar por uma ação para garantir uma rentabilidade mínima desejada em dividendos (baseado na média histórica do DPA). Por exemplo, se uma ação paga R\$ 1,20 de média e você deseja 6% de retorno, seu teto é R\$ 20,00. Se pagar mais que isso, o seu rendimento real será menor que o desejado.")
        st.latex(r"Preço\ Teto = \frac{Média\ DPA\ (" + str(anos_input) + r"\ anos)}{Taxa\ de\ Retorno\ Desejada}")
        
        st.write(f"**Preço Teto (Mediana):** Este cálculo utiliza o valor central dos dividendos pagos no período escolhido. É uma métrica de segurança adicional para evitar distorções causadas por anos onde a empresa pagou dividendos extraordinários (não-recorrentes) que elevam a média artificialmente. A mediana ignora estes extremos.")
        st.latex(r"Preço\ Teto\ (Mediana) = \frac{Mediana\ DPA\ (" + str(anos_input) + r"\ anos)}{Taxa\ de\ Retorno\ Desejada}")
        
        st.write(f"**Margem de Segurança:** Indica o quanto o Preço Teto está acima do preço atual de mercado. Uma margem positiva sugere que a ação está sendo negociada com 'desconto' em relação à sua capacidade de pagar dividendos. Ex: Se o teto é R\$ 20 e o preço é R\$ 15, a sua margem é de 33,3%.")
        st.latex(r"MS = \left( \frac{Preço\ Teto}{Preço\ Atual} - 1 \right) \times 100")
        
        st.write(f"**CAGR (Compound Annual Growth Rate) da Receita/Lucro:** É a taxa de crescimento anual composta de um indicador (Receita ou Lucro) durante um período específico (neste caso, 5 anos). Indica se a empresa tem mantido um crescimento consistente. Valores altos e positivos mostram que a empresa está a expandir as suas operações e resultados de forma saudável.")
        st.latex(r"CAGR = \left( \frac{Valor\ Final}{Valor\ Inicial} \right)^{\frac{1}{n}} - 1")
        
        st.write("**P/L (Preço/Lucro):** Indica quantos anos de lucro a empresa levaria para gerar o seu valor de mercado atual. Um P/L mais baixo pode sugerir que a ação está subvalorizada. Exemplo: Um P/L de 5 indica que em 5 anos o lucro acumulado equivaleria ao preço pago pela ação (payback de 5 anos).")
        
        st.write("**P/VP (Preço/Valor Patrimonial):** Compara o preço de mercado da ação com o valor contábil dos ativos da empresa por ação. Um P/VP menor que 1 pode indicar que a ação está sendo vendida por menos do que valem os seus ativos líquidos. Se for 0,5, está a comprar R\$ 1,00 de património por R\$ 0,50.")
        
        st.write("**ROE (Return on Equity):** Mostra o quanto a empresa consegue gerar de lucro para cada R\$ 1 de património líquido. Um valor mais alto indica uma gestão mais eficiente. Ex: Um ROE de 20% significa que a empresa gerou R\$ 20 de lucro para cada R\$ 100 de capital próprio investido.")
        
        st.write("**DY (Dividend Yield):** É o rendimento de dividendos de uma ação, ou seja, o percentual de retorno que recebe em dividendos em relação ao preço atual da ação. É o indicador preferido de quem procura renda passiva recorrente.")
        
        st.write("**Dívida/PL (Dívida/Património Líquido):** Avalia o nível de endividamento da empresa. Valores menores que 1 indicam uma situação confortável, onde a dívida é menor que o património da companhia. Uma dívida de 0,5 significa que a empresa deve R\$ 0,50 para cada R\$ 1,00 que possui.")

        st.write("**Dívida/EBITDA:** Indica quantos anos de geração de caixa operacional (EBITDA) seriam necessários para a empresa quitar a sua dívida total. Valores abaixo de 2.0x são considerados muito seguros. Se o valor for negativo, significa que a empresa possui mais dinheiro em caixa do que dívidas (Caixa Líquido), oferecendo máxima segurança.")
        
        st.write("**EV/EBITDA (Enterprise Value / EBITDA):** Relação entre o Valor da Empresa (incluindo dívida) e a sua geração de caixa operacional. É como o 'P/L do negócio inteiro'. Valores abaixo de 10 são frequentemente vistos como positivos, indicando uma empresa potencialmente subvalorizada.")

        st.write("**Preço Justo de Graham:** É uma fórmula desenvolvida por Benjamin Graham (mentor de Warren Buffett) para encontrar o valor intrínseco de uma ação equilibrando lucro e patrimônio. A fórmula assume que um investidor não deve pagar mais do que 15 vezes o lucro (P/L) e 1.5 vezes o valor patrimonial (P/VP), resultando no multiplicador de 22.5.")
        st.latex(r"Preço\ Justo\ (Graham) = \sqrt{22.5 \times LPA \times VPA}")
        
        st.markdown('</div>', unsafe_allow_html=True)