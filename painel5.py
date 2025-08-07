# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
import base64
import io

# --- Fun\u00e7\u00e3o para Obter e Processar Dados de DPA ---
@st.cache_data
def get_dpa_data(ticker_symbol: str, anos: int) -> tuple:
    fuso_horario = pytz.timezone('America/Sao_Paulo')
    end_date = datetime.now(fuso_horario)
    start_date = end_date - timedelta(days=anos * 365)
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        if not ticker.history(period="1d").empty:
            dividends = ticker.dividends
            
            if dividends.empty:
                return pd.DataFrame(), 0, f"Sem dados de dividendos para {ticker_symbol}."

            dividends_filtered = dividends.loc[start_date:end_date]
            
            if dividends_filtered.empty:
                return pd.DataFrame(), 0, f"Sem dados de dividendos para {ticker_symbol} nos \u00faltimos {anos} anos."

            dividends_by_year = dividends_filtered.resample('YE').sum().reset_index()
            dividends_by_year.columns = ['Ano', 'DPA (R$)']
            dividends_by_year['Ano'] = dividends_by_year['Ano'].dt.year
            dividends_by_year['Ticker'] = ticker_symbol
            
            media_dpa = dividends_by_year['DPA (R$)'].mean()
            
            mensagem = ""
            if len(dividends_by_year) < anos:
                mensagem = f"Aviso: M\u00e9dia de {ticker_symbol} calculada com {len(dividends_by_year)} anos."
            
            return dividends_by_year, media_dpa, mensagem
        
        else:
            return pd.DataFrame(), 0, f"O ticker '{ticker_symbol}' n\u00e3o \u00e9 v\u00e1lido ou n\u00e3o foi encontrado."
            
    except Exception as e:
        return pd.DataFrame(), 0, f"Ocorreu um erro ao processar o ticker '{ticker_symbol}': {e}"

# --- Fun\u00e7\u00e3o para obter os indicadores fundamentalistas ---
@st.cache_data
def get_fundamental_data(ticker_symbol: str) -> dict:
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        pe_ratio = info.get('trailingPE')
        roe = info.get('returnOnEquity')
        dy = info.get('dividendYield')
        debt_equity = info.get('debtToEquity')
        pb_ratio = info.get('priceToBook')
        ev_ebitda = info.get('enterpriseToEbitda')

        pe_ratio = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (float, int)) else 'N/A'
        roe = f"{roe:.2%}" if isinstance(roe, (float, int)) else 'N/A'
        
        if isinstance(dy, (float, int)):
            if dy > 1:
                dy = f"{dy:.2f}%"
            else:
                dy = f"{dy:.2%}"
        else:
            dy = 'N/A'
        
        if isinstance(debt_equity, (float, int)):
            debt_equity = f"{(debt_equity / 100):.2f}"
        else:
            debt_equity = 'N/A'
            
        pb_ratio = f"{pb_ratio:.2f}" if isinstance(pb_ratio, (float, int)) else 'N/A'
        ev_ebitda = f"{ev_ebitda:.2f}" if isinstance(ev_ebitda, (float, int)) else 'N/A'

        return {
            'Ticker': ticker_symbol,
            'P/L': pe_ratio,
            'ROE': roe,
            'DY': dy,
            'D\u00edvida/PL': debt_equity,
            'P/VP': pb_ratio,
            'EV/EBITDA': ev_ebitda
        }

    except Exception:
        return {
            'Ticker': ticker_symbol,
            'P/L': 'Erro',
            'ROE': 'Erro',
            'DY': 'Erro',
            'D\u00edvida/PL': 'Erro',
            'P/VP': 'Erro',
            'EV/EBITDA': 'Erro'
        }

# --- Layout do Streamlit ---
st.set_page_config(layout="wide")
st.title("An\u00e1lise de DPA e Indicadores Fundamentalistas")
st.write("Escolha uma das op\u00e7\u00f5es abaixo para inserir os tickers para an\u00e1lise.")

# Inputs na barra lateral
with st.sidebar:
    st.header("Op\u00e7\u00f5es de An\u00e1lise")
    
    ticker_input = st.text_input(
        "Tickers (separados por v\u00edrgula)",
        value="PETR4.SA,VALE3.SA"
    )
    
    anos_input = st.number_input(
        "N\u00famero de Anos para DPA",
        value=5,
        min_value=1
    )
    
    taxa_retorno_input = st.number_input(
        "Taxa de Retorno Anual Desejada (%)",
        value=6,
        min_value=6
    )

    st.markdown("---")
    st.header("Upload de Arquivo CSV")
    uploaded_file = st.file_uploader(
        "Se preferir, fa\u00e7a o upload de um arquivo CSV",
        type=['csv']
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if 'Ticker' in df_upload.columns:
                ticker_list_upload = df_upload['Ticker'].tolist()
                ticker_input = ",".join(ticker_list_upload)
                st.success("Tickers do arquivo carregados!")
            else:
                st.error("O arquivo CSV deve ter uma coluna chamada 'Ticker'.")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")

# Bot\u00e3o principal para acionar a an\u00e1lise
if st.button("Gerar Relat\u00f3rio"):
    if not ticker_input:
        st.warning("Por favor, insira pelo menos um ticker.")
    else:
        # L\u00f3gica de processamento
        with st.spinner("Gerando relat\u00f3rio... Isso pode levar alguns instantes."):
            tickers = [t.strip().upper() for t in ticker_input.split(',')]
            all_dfs = []
            media_data = []
            fundamental_data = []
            preco_teto_data = [] 
            messages = []
            alerta_compra_msgs = []

            taxa_retorno_decimal = max(6, taxa_retorno_input) / 100

            for ticker in tickers:
                df_dpa, media, mensagem = get_dpa_data(ticker, anos_input)
                fundamental_dict = get_fundamental_data(ticker)
                
                fundamental_data.append(fundamental_dict)
                
                try:
                    current_price = yf.Ticker(ticker).info.get('regularMarketPrice')

                    if media > 0 and taxa_retorno_decimal > 0:
                        preco_teto = media / taxa_retorno_decimal
                        
                        margem_seguranca = None
                        if current_price is not None and current_price > 0:
                            margem_seguranca = ((preco_teto / current_price) - 1) * 100
                        
                        preco_teto_data.append({
                            'Ticker': ticker,
                            'Pre\u00e7o Atual': current_price,
                            'M\u00e9dia do DPA 5 anos': media,
                            'Pre\u00e7o Teto': preco_teto,
                            'Margem Seguran\u00e7a (%)': margem_seguranca
                        })

                        if margem_seguranca is not None and margem_seguranca > 0:
                            alerta_compra_msgs.append(f"Aten\u00e7\u00e3o: O Pre\u00e7o Atual de {ticker} (R$ {current_price:.2f}) est\u00e1 abaixo do Pre\u00e7o Teto (R$ {preco_teto:.2f}). Margem de Seguran\u00e7a de {margem_seguranca:.2f}%.")
                    else:
                        preco_teto_data.append({
                            'Ticker': ticker,
                            'Pre\u00e7o Atual': current_price,
                            'M\u00e9dia do DPA 5 anos': media,
                            'Pre\u00e7o Teto': 0,
                            'Margem Seguran\u00e7a (%)': 0
                        })
                except Exception:
                    messages.append(f"Erro ao obter o pre\u00e7o atual ou calcular o Pre\u00e7o Teto para {ticker}.")

                if not df_dpa.empty:
                    all_dfs.append(df_dpa)
                    media_data.append({'Ticker': ticker, f'M\u00e9dia DPA (R$ - {anos_input} anos)': round(media, 2)})
                    if "Aviso" in mensagem:
                        messages.append(mensagem)
                else:
                    messages.append(mensagem)
            
            if not all_dfs:
                st.error("Nenhum dado foi encontrado para os tickers fornecidos. Verifique a grafia.")
            else:
                # --- Exibi\u00e7\u00e3o do Relat\u00f3rio ---
                
                # Alertas
                if alerta_compra_msgs:
                    st.success("### Alertas de Compra")
                    for msg in alerta_compra_msgs:
                        st.info(msg)

                # Indicadores Fundamentalistas
                st.subheader("Indicadores Fundamentalistas")
                fundamental_df = pd.DataFrame(fundamental_data)
                st.dataframe(fundamental_df, hide_index=True)

                st.write(
                    "Nota: A disponibilidade dos dados fundamentalistas (como P/VP) pode variar dependendo do ticker e da fonte (Yahoo Finance)."
                )

                # M\u00e9dia do DPA
                st.subheader(f"M\u00e9dia do DPA por Empresa (\u00faltimos {anos_input} anos)")
                media_df = pd.DataFrame(media_data)
                st.dataframe(media_df, hide_index=True)
                
                # Mensagens de erro/aviso
                if messages:
                    for msg in messages:
                        st.warning(msg)

                # Gr\u00e1fico de DPA
                st.subheader("DPA Anual por Ticker")
                final_df = pd.concat(all_dfs)
                fig = go.Figure()
                for ticker in final_df['Ticker'].unique():
                    df_ticker = final_df[final_df['Ticker'] == ticker]
                    fig.add_trace(go.Bar(
                        x=df_ticker['Ano'], 
                        y=df_ticker['DPA (R$)'], 
                        name=ticker
                    ))
                fig.update_layout(
                    barmode='group',
                    xaxis_title="Ano",
                    yaxis_title="DPA (R$)",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Tabela de Pre\u00e7o Teto com formata\u00e7\u00e3o condicional
                st.subheader("Pre\u00e7o Teto")
                preco_teto_df = pd.DataFrame(preco_teto_data)
                
                # Fun\u00e7\u00e3o para aplicar a formata\u00e7\u00e3o de cor
                def highlight_margem_seguranca(val):
                    try:
                        val_float = float(val)
                        if val_float > 0:
                            return 'background-color: #28a745; color: white'
                        elif val_float < 0:
                            return 'background-color: #dc3545; color: white'
                        else:
                            return 'background-color: #6c757d; color: white'
                    except (ValueError, TypeError):
                        return ''

                # Aplica a formata\u00e7\u00e3o de cor
                styled_df = preco_teto_df.style.applymap(
                    highlight_margem_seguranca,
                    subset=pd.IndexSlice[:, ['Margem Seguran\u00e7a (%)']]
                )

                # Formata os valores para exibi\u00e7\u00e3o (ap\u00f3s a formata\u00e7\u00e3o de cor)
                styled_df = styled_df.format({
                    'Pre\u00e7o Atual': 'R$ {:.2f}',
                    'M\u00e9dia do DPA 5 anos': 'R$ {:.2f}',
                    'Pre\u00e7o Teto': 'R$ {:.2f}',
                    'Margem Seguran\u00e7a (%)': '{:.2f}%'
                }, na_rep='N/A')

                st.dataframe(styled_df, hide_index=True)
                
                # Bot\u00e3o de download
                csv = preco_teto_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Baixar dados de Pre\u00e7o Teto (.csv)",
                    csv,
                    "analise_preco_teto.csv",
                    "text/csv",
                    key='download-button'
                )

                # Explica\u00e7\u00f5es
                st.markdown("---")
                st.subheader("O que significam os indicadores?")
                st.write("**P/L (Pre\u00e7o/Lucro):** Indica quantos anos de lucro a empresa levaria para gerar o seu valor de mercado atual. Um P/L mais baixo pode sugerir que a a\u00e7\u00e3o est\u00e1 subvalorizada.")
                st.write("**P/VP (Pre\u00e7o/Valor Patrimonial):** Compara o pre\u00e7o de mercado da a\u00e7\u00e3o com o valor cont\u00e1bil dos ativos da empresa por a\u00e7\u00e3o. Um P/VP menor que 1 pode indicar que a a\u00e7\u00e3o est\u00e1 subvalorizada.")
                st.write("**ROE (Return on Equity):** Mostra o quanto a empresa consegue gerar de lucro para cada R$ 1 de patrim\u00f4nio l\u00edquido. Um valor mais alto indica uma gest\u00e3o mais eficiente.")
                st.write("**DY (Dividend Yield):** \u00c9 o rendimento de dividendos de uma a\u00e7\u00e3o, ou seja, o percentual de retorno que voc\u00ea recebe em dividendos em rela\u00e7\u00e3o ao pre\u00e7o da a\u00e7\u00e3o. Um DY consistente \u00e9 atrativo para investidores que buscam renda.")
                st.write("**D\u00edvida/PL (D\u00edvida/Patrim\u00f4nio L\u00edquido):** Avalia o n\u00edvel de endividamento da empresa. Um valor menor indica que a empresa tem uma menor propor\u00e7\u00e3o de d\u00edvidas em rela\u00e7\u00e3o ao seu capital pr\u00f3prio.")
                st.write("**EV/EBITDA (Enterprise Value / EBITDA):** A rela\u00e7\u00e3o EV/EBITDA tende a flutuar em diferentes ind\u00fastrias, mas para empresas dentro do S&P 500, a m\u00e9dia tem variado geralmente entre 13 e 17 nos \u00faltimos anos. Como regra geral, uma rela\u00e7\u00e3o EV/EBITDA abaixo de 10 \u00e9 frequentemente vista positivamente por analistas, indicando uma empresa potencialmente subvalorizada e financeiramente saud\u00e1vel.")

# --- Como executar o Streamlit ---
# Para rodar a aplica\u00e7\u00e3o, abra o terminal no seu ambiente Anaconda e execute o comando:
# streamlit run painel5.py