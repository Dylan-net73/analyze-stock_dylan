# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
import io
import requests

# --- CONFIGURAÇÃO PARA RENDER (Sessão Global) ---
@st.cache_resource
def get_global_session():
    s = requests.Session()
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return s

def get_safe_ticker(symbol):
    session = get_global_session()
    return yf.Ticker(symbol, session=session)

# --- CONFIGURAÇÕES DA PÁGINA ---
st.set_page_config(page_title="Análise Fundamentalista Pro", layout="wide")

@st.cache_data(ttl=300) # Cache de 5 min para o Render ser mais rápido
def fetch_price(ticker_symbol):
    try:
        t = get_safe_ticker(ticker_symbol)
        df = t.history(period="5d")
        return float(df['Close'].iloc[-1]) if not df.empty else 0.0
    except: return 0.0

@st.cache_data(ttl=3600)
def fetch_fundamental_data(ticker_symbol, anos_ref):
    data = {
        'dpa_df': pd.DataFrame(), 'media': 0.0, 'mediana': 0.0, 'anos_enc': 0,
        'low_m': "N/A", 'pay_m': "N/A",
        'fund': {'Ticker': ticker_symbol, 'P/L': None, 'P/VP': None, 'ROE': None, 'DY': None, 'Dívida/PL': None, 'EV/EBITDA': None}
    }
    try:
        t = get_safe_ticker(ticker_symbol)
        m_names = {1:'Jan', 2:'Fev', 3:'Mar', 4:'Abr', 5:'Mai', 6:'Jun', 7:'Jul', 8:'Ago', 9:'Set', 10:'Out', 11:'Nov', 12:'Dez'}
        
        hist = t.history(period="5y")
        if not hist.empty:
            hist['Month'] = hist.index.month
            hist['Year'] = hist.index.year
            min_prices = hist.groupby(['Year', 'Month'])['Low'].min().reset_index()
            idx = min_prices.groupby('Year')['Low'].idxmin()
            data['low_m'] = ", ".join([m_names[m] for m in min_prices.loc[idx, 'Month'].value_counts().head(3).index])

        divs = t.dividends
        if divs.empty and not hist.empty and 'Dividends' in hist.columns:
            divs = hist['Dividends'][hist['Dividends'] > 0]

        if not divs.empty:
            m_paid = divs.index.month.value_counts()
            data['pay_m'] = ", ".join([m_names[m] for m in sorted(m_paid[m_paid >= m_paid.max()*0.4].index)])
            fuso = pytz.timezone('America/Sao_Paulo')
            limite = datetime.now(fuso) - timedelta(days=anos_ref*365)
            df_divs = divs.loc[limite:].resample('YE').sum().reset_index()
            df_divs.columns = ['Ano', 'DPA']
            df_divs['Ano'] = df_divs['Ano'].dt.year
            df_comp = df_divs[df_divs['Ano'] < datetime.now(fuso).year].copy()
            if not df_comp.empty:
                data['media'], data['mediana'], data['anos_enc'] = df_comp['DPA'].mean(), df_comp['DPA'].median(), len(df_comp)
                df_comp['Ticker'] = ticker_symbol
                data['dpa_df'] = df_comp

        info = t.info
        data['fund'].update({
            'P/L': info.get('trailingPE') or info.get('forwardPE'), 'P/VP': info.get('priceToBook'),
            'ROE': info.get('returnOnEquity'), 'DY': info.get('dividendYield'),
            'Dívida/PL': (info.get('debtToEquity')/100) if info.get('debtToEquity') else None,
            'EV/EBITDA': info.get('enterpriseToEbitda')
        })
    except: pass
    return data

# --- SIDEBAR (TODOS OS FILTROS MANTIDOS) ---
with st.sidebar:
    st.header("⚙️ Configurações")
    tickers_txt = st.text_input("Tickers (Use .SA)", value="PETR4.SA,VALE3.SA,ITUB4.SA,BBAS3.SA,TRPL4.SA")
    anos_sel = st.number_input("Anos DPA (Histórico)", value=5, min_value=1)
    taxa_alvo = st.number_input("Taxa Desejada %", value=6.0, step=0.5)
    if st.button("🧹 Limpar Cache (Resetar Dados)"):
        st.cache_data.clear()
        st.toast("Cache limpo!")
    st.divider()
    st.header("🎯 Filtros de Tela")
    f_pl = st.number_input("P/L Máximo (0 = desativado)", value=0.0)
    f_dy = st.number_input("DY Mínimo % (0 = desativado)", value=0.0)

# --- PROCESSAMENTO ---
if st.button("🚀 Gerar Relatório Completo"):
    list_t = [x.strip().upper() for x in tickers_txt.split(',') if x.strip()]
    final_res = []
    
    with st.spinner("Sincronizando dados..."):
        for symbol in list_t:
            d = fetch_fundamental_data(symbol, anos_sel)
            p = fetch_price(symbol)
            teto = d['media'] / (taxa_alvo/100) if d['media'] > 0 else 0
            ms = ((teto/p)-1)*100 if p > 0 else -100.0
            
            final_res.append({
                'Ticker': symbol, 'Preço Atual': p, 'Média DPA': d['media'], 
                'Mediana DPA': d['mediana'], 'Preço Teto': teto, 'Margem Segurança (%)': ms, 
                'Meses Baixos (Preço)': d['low_m'], 'Meses Pagamento (Div)': d['pay_m'],
                'Anos Dados': d['anos_enc'], # No final da tabela
                'fund_raw': d['fund'], 'dpa_df': d['dpa_df']
            })

    v_df = pd.DataFrame(final_res).drop(columns=['fund_raw', 'dpa_df'])
    f_df = pd.DataFrame([x['fund_raw'] for x in final_res])

    if f_pl > 0: v_df = v_df[v_df['Ticker'].isin(f_df[f_df['P/L'] <= f_pl]['Ticker'].tolist())]
    if f_dy > 0: v_df = v_df[v_df['Ticker'].isin(f_df[f_df['DY']*100 >= f_dy]['Ticker'].tolist())]

    st.title("📊 Painel Fundamentalista Pro")

    # Exportação Excel
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='xlsxwriter') as wr:
        v_df.to_excel(wr, index=False, sheet_name='Valuation')
        f_df.to_excel(wr, index=False, sheet_name='Fundamentalista')
    st.download_button("📥 Baixar Relatório Excel", out.getvalue(), "Relatorio.xlsx")

    # Tabelas e Explicações
    tab1, tab2 = st.tabs(["💎 Valuation & Sazonalidade", "📈 Saúde Financeira"])
    with tab1:
        st.dataframe(v_df.style.map(lambda x: 'background-color: #28a745; color: white' if isinstance(x, (float, int)) and x > 0 else 'background-color: #dc3545; color: white' if isinstance(x, (float, int)) and x < 0 else '', subset=['Margem Segurança (%)'])
                     .format({'Preço Atual':'R$ {:.2f}', 'Média DPA':'R$ {:.2f}', 'Mediana DPA':'R$ {:.2f}', 'Preço Teto':'R$ {:.2f}', 'Margem Segurança (%)':'{:.2f}%'}), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(f_df[['Ticker','P/L','P/VP','ROE','DY','Dívida/PL','EV/EBITDA']], use_container_width=True, hide_index=True)

    # Gráfico
    all_d = [r['dpa_df'] for r in final_res if not r['dpa_df'].empty]
    if all_d:
        st.subheader("📅 Histórico de Proventos")
        df_g = pd.concat(all_d)
        fig = go.Figure()
        for t in df_g['Ticker'].unique():
            d = df_g[df_g['Ticker']==t]; fig.add_trace(go.Bar(x=d['Ano'], y=d['DPA'], name=t))
        fig.update_layout(template='plotly_white', barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    # --- DICIONÁRIO RESTAURADO ---
    st.divider()
    st.subheader("📚 Dicionário de Indicadores e Metodologia")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**P/L (Preço/Lucro):** Anos que o mercado aceita pagar pelo lucro atual.")
        st.markdown("**P/VP:** Preço sobre Valor Patrimonial. Também indica os **anos de retorno do investimento** sobre ativos reais.")
    with c2:
        st.markdown("**Dívida/PL:** Acima de 1 indica que a dívida supera o capital próprio.")
        st.markdown("**EV/EBITDA:** Valor da empresa sobre geração de caixa operacional.")
        st.markdown("**ROE:** Eficiência em gerar lucro com o capital dos sócios.")
    with c3:
        st.markdown("**Mediana DPA:** Valor central, protegendo de dividendos atípicos.")
        st.markdown("**Sazonalidade:** Janelas de preços baixos e meses de pagamento.")

    st.info(r"""
    ### 💡 Metodologia do Preço Teto
    $$Preço \ Teto = \frac{Média \ do \ DPA}{Taxa \ de \ Retorno \ Alvo}$$
    Média e Mediana calculadas com base no parâmetro **Anos DPA** selecionado.
    """)