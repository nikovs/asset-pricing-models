import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import dash_table
import dash_bootstrap_components as dbc

import numpy as np
import pandas as pd
import statsmodels.regression.linear_model as lm
import statsmodels.tools.tools as ct
from scipy import stats
import statsmodels.api as sm

import yfinance as yf
from pandas_datareader import data
import pandas_datareader as pdr
import datetime
import datetime as dt


#==================================================================================================
#data
# Import common databases

Damodaran=pd.ExcelFile('ctryprem.xlsx')
RWA_tab = pd.read_excel(Damodaran, 'Regional Weighted Averages',header=0,index_col=0)
RWA_tab = RWA_tab.dropna(subset=['Equity Risk Premium'], axis=0)

#RWA_tab.head()

# List of GICS sectors
sector_map={'Utilities':'XLU', 'Technology':'XLK', 'Basic Materials':'XLB', 'Industrials':'XLI', 'Healthcare': 'XLV',
           'Financials': 'XLF', 'Financial Services': 'XLF', 'Energy':'XLE', 'Consumer Staples':'XLP', 'Consumer Defensive':'XLP',
           'Consumer Discretionary':'XLY', 'Consumer Cyclical':'XLY',
           'Communication Services':'XLC', 'Real Estate':'XLRE'}

# Risk free interest rate proxy list
Rf_map={'30y':'^TYX','10y':'^TNX','3m':'^IRX'}

#List of countries and econ status
country_status=pd.read_csv('DevelopedCountries.csv',header=0)

#button counter
count = 0

#====================================================================================================================

#functions

def ModifiedCAPM(close, CR, rf):
    
    close = close.resample('M').mean().pct_change().dropna()
    data = pd.DataFrame(close).merge(rf, how='left', left_index=True, right_index=True)
    data['excessReturn'] = data['Adj Close'] - data['rf']
    
    #calculate excess market return
    market=yf.download('SPY', period='5y', interval='1d',progress=False)['Adj Close']
    market=market.resample('M').mean().pct_change().dropna()
    data['excessMarket']=market-data['rf']
    
    
    #Regressing beta
    
    ols = sm.OLS(data.excessReturn,data.excessMarket).fit()
    #pred = ols.predict(data.excessMarket)
    #mse=mean_squared_error(data.excessReturn, pred)
    
    return (data['rf'][-1] + ols.params[0]*data.excessMarket[-1]+CR)*100*12, ols.rsquared*100,ols.mse_resid,ols

def IntlCAPM( sector, beta_iM, alphas, rf):
    
    #calculate return and excess return of the sector

    close=yf.download(sector[1], period='5y', interval='1d',progress=False)['Adj Close']
        
    close = close.resample('M').mean().pct_change().dropna()
    data = pd.DataFrame(close).merge(rf, how='left', left_index=True, right_index=True)
    data['excessSectorReturn'] = data['Adj Close'] - data['rf']
    
    #calculate excess market return
    market=yf.download('SPY', period='5y', interval='1d',progress=False)['Adj Close']
    market=market.resample('M').mean().pct_change().dropna()
    data['excessMarket']=market-data['rf']
    
    #Regressing beta_tM
    
    ols = sm.OLS(data.excessSectorReturn,data.excessMarket).fit()
    #pred = ols.predict(data.excessMarket)
    #mse=mean_squared_error(data.excessSectorReturn, pred)
    
    beta_tM=ols.params[0]
    
    #Calculate beta_tnM and beta_tmM
    beta_tiM=[beta_tM*element for element in beta_iM]
    
    #Generate alphas
    if len(alphas)==0:
        alphas=np.random.dirichlet(np.ones(len(beta_tiM)))
    
    #Calculate beta_p
    beta_p=alphas@np.array(beta_tiM)
    beta_p
    
    #Calculate modified international CAMP
    Int_R=data.rf[-1]+beta_p*data.excessMarket[-1]
    
    
    return  Int_R*100*12,ols.rsquared*100, ols.mse_resid


def apt(close, country, sector, rf):

    #interest rate, industrial production,fed funds, crude oil price
    indicator_list = ['T5YIE', 'INDPRO', 'FEDFUNDS', 'DCOILWTICO'] 
    indicator = pdr.DataReader(indicator_list, 'fred')
    indicator['INDPRO'] = indicator['INDPRO'].pct_change()
    indicator['DCOILWTICO'] = indicator['DCOILWTICO'].pct_change()
    indicator = indicator.resample('M').mean().dropna()
    indicator.columns = ['Inflation', 'IndustrialProduction', 'FedFunds', 'CrudeOilPrice']
    
    sector_data=yf.download(sector[1], period='5y', interval='1d',progress=False)['Adj Close']
    sector_data = sector_data.resample('M').mean().pct_change().dropna()
    indicator[sector[1]] = sector_data
    
    #calculate return and excess return
    close = close.resample('M').mean().pct_change().dropna()
    data = pd.DataFrame(close).merge(rf, how='left', left_index=True, right_index=True)
    #print(data.head())
    data['excessReturn'] = data['Adj Close'] - data['rf']
    data = data.merge(indicator, how='left', left_index=True, right_index=True)
    
    #country risk
    data['CR'] = RWA_tab[RWA_tab.index==country]['Country Risk Premium'][0]
    data = data[['Inflation', 'IndustrialProduction', 'FedFunds', 'CrudeOilPrice', sector[1], 'CR', 'excessReturn']].dropna()
    
    #fit data
    formula =  "excessReturn~ Inflation+IndustrialProduction+FedFunds+CrudeOilPrice+{}+CR".format(sector[1])
    model = lm.OLS.from_formula(formula, data = data).fit()
    
    #evaluate model
    r2 = model.rsquared
    mse = model.mse_resid
    E_return=model.params[1:]@data.iloc[-1,:-1]+rf.iloc[-1]
    return E_return[0]*12*100,r2*100,mse,model

def famafrench(factor, country, close, rf):
    
    enddate = dt.datetime.strptime("2021-12-31", "%Y-%m-%d").date()
    startdate = enddate - dt.timedelta(days=365*5)
    
    market = yf.download(tickers='SPY', period='5y', interval='1d',progress=False)[['Adj Close']]
    market=market.resample('M').mean().pct_change().dropna()
    
    #check factors, return error message if it is not 3 or 5
    if factor==5:
        #print('5 Factor Fama-French\n\n')
        FF = pd.read_csv('F-F_Research_Data_5_Factors_2x3_daily.csv', header=2, index_col=0).dropna()
        FF.index = pd.to_datetime(FF.index, format='%Y%m%d')
        FF.columns = ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        FF = FF[['SMB', 'HML', 'RMW', 'CMA']]
    if factor==3:  
        #print('3 Factor Fama-French\n\n')
        FF=pd.read_csv('F-F_Research_Data_Factors_Daily.csv', header=3, index_col=0).dropna()
        FF.index = pd.to_datetime(FF.index, format='%Y%m%d')
        FF.columns = ['Mkt_RF', 'SMB', 'HML', 'RF']
        FF = FF[['SMB', 'HML']]
    
    #calculate return and excess return
    close = pd.DataFrame(close).pct_change().dropna()
    data = close[['Adj Close']].merge(FF, how='left', left_index=True, right_index=True)
    data['RF'] = rf['rf']
    
    #country risk
    if country not in RWA_tab.index:
        country = 'Global'
    data['CR'] = RWA_tab[RWA_tab.index==country]['Country Risk Premium'][0]
    data = data.resample('M').mean().dropna()
    data['Mkt_RF']=market['Adj Close']-rf['rf']
    data['excessReturn'] = data['Adj Close']*100 - data['RF']
    RF=data['RF'].copy()
    data = data[['Mkt_RF', 'SMB', 'HML', 'CR', 'excessReturn']].dropna()
    
    
    #fit model
    formula =  "excessReturn~ {}".format('+'.join(data.columns[:-1]))
    model = lm.OLS.from_formula(formula, data = data).fit()
    
    #evaluate model
    r2 =  model.rsquared
    mse =  model.mse_resid
    E_return=model.params[1:]@data.iloc[-1,:-1]+RF.iloc[-1]
    
    return E_return*12*100,r2*100,mse,model


#=====================================================================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# Cards

param_card = dbc.Card([
    dbc.CardBody([
        dbc.Row([dbc.Col( #ticker text
                html.H6('Stock of interest for beta calculation:', 
                        className='ticker'),),
                 dbc.Col( #ref_ticker text
                html.H6('For Modified CAPM, reference stock in US market for beta calculation:',
                        className='ref_ticker'),),
                 dbc.Col( #country text
                html.H6('Country of the interested stock:', 
                        className='country'),),
                 ]),
        dbc.Row([dbc.Col(#ticker input
            dcc.Input(
                id="ticker", type="text", placeholder="Input ticker"),
                lg=4, md=6, sm=8, width={"size": 4},),
                 dbc.Col(#ref_ticker input
            dcc.Input(
                id="ref_ticker", type="text", placeholder="Input ticker"),
                lg=4, md=6, sm=8, width={"size": 4},),
                 dbc.Col(#country input
            dcc.Dropdown(id="country", placeholder='Input country',
                         options=list(RWA_tab.index),
                         multi=False,
                         value='10y'),
                lg=4, md=6, sm=8, width={"size": 4},),
                 ]),
        html.Br(),
        
        dbc.Row([dbc.Col( #rf text
                html.H6('Type of risk free interest rate proxy:', 
                        className='rf'),width={"size": 4},),
                 dbc.Col( #ff text
                html.H6('Fama French factors:',
                        className='factor'),width={"size": 4},),
                 dbc.Col( #countries text
                html.H6('Input country(ies) dominating the sector of interest for Modified International CAPM:', 
                        className='rf'),width={"size": 4},),
                 ]),
        dbc.Row([dbc.Col(#rf input
            dcc.Dropdown(id="rf_select", placeholder='Risk free interest rate',
                         options=[
                             {"label": "30 years T-Bond", "value": '30y'},
                             {"label": "10 years T-Bond", "value": '10y'},
                             {"label": "3 months T-Bond", "value": '3m'}],
                         multi=False,
                         value='10y'),
                 lg=4, md=6, sm=8, width={"size": 3},),
                 dbc.Col(#ff input
            dcc.Dropdown(id="ff", placeholder='FF factor',
                         options=[
                             {"label": "3 Factors", "value": 3},
                             {"label": "5 Factors", "value": 5}],
                         multi=False,
                         value=3),
                 lg=4, md=6, sm=8, width={"size": 3},),
                 dbc.Col(#countries input
            dcc.Dropdown(id="countries", placeholder='Input country(ies)',
                         options=list(RWA_tab.index),
                         multi=True),
                 lg=4, md=6, sm=8, width={"size": 3},),
                 ]),
        html.Br(),
        
        dbc.Row([
                 dbc.Col( #yn text
                html.H6('Input sector weight by country for Modified International CAPM:',
                        className='yn'),width={"size": 4},),
                 dbc.Col( #yn text
                html.H6('Input sector weight per country in a list of decimals (eg. [0.3, 0.2, ..., 0.3]):',
                        className='yn'),width={"size": 4},),
                 ]),
        dbc.Row([
                 dbc.Col(#yn input
            dcc.Dropdown(id="yn", placeholder='Yes/No',
                         options=[
                             {"label": "Yes", "value": 'Y'},
                             {"label": "No", "value": 'N'}],
                         multi=False,
                         value='Y'),
                 lg=4, md=6, sm=8, width={"size": 3},),
                 dbc.Col(#weight input
            dcc.Input(
                id="weight", type="text", placeholder="Input weights"),
                lg=4, md=6, sm=8, width={"size": 4},),
            dbc.Col(html.Button('Fit Data', id='calc_button', n_clicks=0), width={"size": 3},)
                 ]),
        html.Br(),
    ]),
], color='light')

capm_card = dbc.Card([
    dbc.CardBody([
        html.H6("Annualized Expected Return (%):"),
        dcc.Loading(
                    id="m1_return",
                    children=[html.Div([html.Div(id="model1_return")])],
                    type="circle",
                ),
        #html.H6(id='model1_return', children=[], className='capm_return'),
        html.H6("R-squared:"),
        html.H6(id='model1_r2', children=[], className='capm_r2'),
        html.H6("MSE:"),
        html.H6(id='model1_mse', children=[], className='capm_mse'),
    ]),
], color='light')


itlcapm_card = dbc.Card([
    dbc.CardBody([
        html.H6("Annualized Expected Return (%):"),
        dcc.Loading(
                    id="m2_return",
                    children=[html.Div([html.Div(id="model2_return")])],
                    type="circle",
                ),
        #html.H6(id='model2_return', children=[], className='itlcapm_return'),
        html.H6("R-squared:"),
        html.H6(id='model2_r2', children=[], className='itlcapm_r2'),
        html.H6("MSE:"),
        html.H6(id='model2_mse', children=[], className='itlcapm_mse'),
    ]),
], color='light')

apt_card = dbc.Card([
    dbc.CardBody([
        html.H6("Annualized Expected Return (%):"),
        dcc.Loading(
                    id="m3_return",
                    children=[html.Div([html.Div(id="model3_return")])],
                    type="circle",
                ),
        #html.H6(id='model3_return', children=[], className='apt_return'),
        html.H6("R-squared:"),
        html.H6(id='model3_r2', children=[], className='apt_r2'),
        html.H6("MSE:"),
        html.H6(id='model3_mse', children=[], className='apt_mse'),
    ]),
], color='light')

ff_card = dbc.Card([
    dbc.CardBody([
        html.H6("Annualized Expected Return (%):"),
        dcc.Loading(
                    id="m4_return",
                    children=[html.Div([html.Div(id="model4_return")])],
                    type="circle",
                ),
        #html.H6(id='model4_return', children=[], className='ff_return'),
        html.H6("R-squared:"),
        html.H6(id='model4_r2', children=[], className='ff_r2'),
        html.H6("MSE:"),
        html.H6(id='model4_mse', children=[], className='ff_mse'),
    ]),
], color='light')

#==========================================================================================================

# App layout
app.layout = html.Div([
    html.Br(),

    dbc.Row(dbc.Col(html.H3("WQU Capstone Project"), width={'size':6, 'offset':1},),),
    dbc.Row(dbc.Col(html.H4("Alfonso Chang, Duy Le, Nikolaus Siauw"), width={'size':6, 'offset':1},),),
    dbc.Row(dbc.Col(html.H6("Insert all parameters and click Fit Data button to process."), width={'size':6, 'offset':1},),),
    
    html.Br(),
    
    dbc.Row([
        dbc.Col(param_card, width={'size':10, 'offset':1, 'order':1}), 
    ],className='rows2'),
    
    html.Br(),
    
    dbc.Row([
        dbc.Col(
            html.H4("Modified CAPM", className="card-title"),
            width={'size':5, 'offset':1, 'order':1}),
        dbc.Col(
            html.H4("Modified International CAPM", className="card-title"),
            width={'size':5, 'order':2}),
    ], className='rows1'),
    
    dbc.Row([
        dbc.Col(
            capm_card,
            width={'size':5, 'offset':1, 'order':1}),
        dbc.Col(
            itlcapm_card,
            width={'size':5, 'order':2}),
    ], className='rows1'),
    
    html.Br(),
    
    dbc.Row([
        dbc.Col(
            html.H4("APT", className="card-title"),
            width={'size':5, 'offset':1, 'order':1}),
        dbc.Col(
            html.H4("Fama-French", className="card-title"),
            width={'size':5, 'order':2}),
    ], className='rows1'),
    
    dbc.Row([
        dbc.Col(
            apt_card,
            width={'size':5, 'offset':1, 'order':1}),
        dbc.Col(
            ff_card,
            width={'size':5, 'order':2}),
    ], className='rows1'),

])



# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='model1_return', component_property='children'),
     Output(component_id='model1_r2', component_property='children'),
     Output(component_id='model1_mse', component_property='children'),
     Output(component_id='model2_return', component_property='children'),
     Output(component_id='model2_r2', component_property='children'),
     Output(component_id='model2_mse', component_property='children'),
     Output(component_id='model3_return', component_property='children'),
     Output(component_id='model3_r2', component_property='children'),
     Output(component_id='model3_mse', component_property='children'),
     Output(component_id='model4_return', component_property='children'),
     Output(component_id='model4_r2', component_property='children'),
     Output(component_id='model4_mse', component_property='children')],
    [Input(component_id='ticker', component_property='value'),
     Input(component_id='ref_ticker', component_property='value'),
     Input(component_id='country', component_property='value'),
     Input(component_id='rf_select', component_property='value'),
     Input(component_id='ff', component_property='value'),
     Input(component_id='countries', component_property='value'),
     Input(component_id='yn', component_property='value'),
     Input(component_id='weight', component_property='value'),
    Input(component_id='calc_button', component_property='n_clicks')]
)

def update_graph(ticker, ref_ticker, country, rf_select, ff, countries, yn, weight, calc_button):
    
    global count
    factor = ff

    #Economic status of the country
    if country_status[country_status.country==country]['hdi2019'].values >= 0.80:
        econ_status = 'developed market'
    else:
        econ_status = 'emerging market'

    #calculate return and excess return
    if econ_status == 'developed market':
        close=yf.download(ticker, period='5y', interval='1d',progress=False)['Adj Close']
    else:
        close=yf.download(ref_ticker, period='5y', interval='1d',progress=False)['Adj Close']
        
    #Sector - automatically identified based on Ticker
    if econ_status == 'developed market':
        sector = yf.Ticker(ticker).info['sector']
    else:
        sector=yf.Ticker(ref_ticker).info['sector']
    sector = [sector,sector_map[sector]]
    
    countries = list(countries)
    beta_iM = []
    for i in countries:
        beta_iM.append(1+RWA_tab.loc[i,'Equity Risk Premium'])
        
    alphas = (list(eval(weight)))

    CR=RWA_tab.loc[country,'Country Risk Premium']
    
    # load risk free interest rate
    rf = yf.download(tickers=Rf_map[rf_select], period='5y', interval='1d',progress=False)[['Adj Close']]
    rf = rf[['Adj Close']].resample('M').mean()
    rf.columns = ['rf']
    
    if calc_button>count:
        model1 = ModifiedCAPM(close, CR, rf)
        model2 = IntlCAPM( sector, beta_iM, alphas, rf)
        model3 = apt(close, country, sector, rf)
        model4 = famafrench(factor, country, close, rf)
    
        o1 = [round(x, 4) for x in model1[:3]]
        o2 = [round(x, 4) for x in model2[:3]]
        o3 = [round(x, 4) for x in model3[:3]]
        o4 = [round(x, 4) for x in model4[:3]]
        
        count += 1
    
    #return str(ticker), str(ref_ticker), str(country), str(rf), str(ff), str(countries), str(yn), str(weight), str(beta_iM), str(alphas), str(sector), str(econ_status)
    
    return o1[0], o1[1], o1[2], o2[0], o2[1], o2[2], o3[0], o3[1], o3[2], o4[0], o4[1], o4[2]
    
    
if __name__ == '__main__':
    app.run_server()