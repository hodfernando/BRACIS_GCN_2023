import os
from datetime import datetime, timedelta
import dash
import numpy as np
from dash import dcc, html
import pandas as pd
from sklearn.metrics import r2_score

# Path do diretório atual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Caminho do diretório 'results/Prophet/'
results_prophet = os.path.join(os.path.join(current_dir, "results"), 'LSTM_Prophet')

# Caminho do diretório 'results/GCLSTM/'
results_gclstm = os.path.join(os.path.join(current_dir, "results"), 'GCLSTM')

# Caminho do diretório 'results/GCRN/'
results_gcrn = os.path.join(os.path.join(current_dir, "results"), 'GCRN')

# Carregando os dados de predição dos modelos
pred_gclstm = np.load(results_gclstm + f'\\2020-2022\\y_pred_no_norm_2020-2022.npy').mean(axis=0)
pred_gcrn = np.load(results_gcrn + f'\\2020-2022\\y_pred_no_norm_2020-2022.npy').mean(axis=0)
pred_prophet = np.load(results_prophet + f'\\pred_prophet.npy')[13::14, :]
pred_lstm = np.load(results_prophet + f'\\pred_lstm.npy')

# Carregando os dados reais
y_real = np.load(results_gclstm + f'\\2020-2022\\y_real_no_norm_2020-2022.npy').mean(axis=0)

# RMSE do modelo GCLSTM
rmse_gclstm = np.load(results_gclstm + f'\\2020-2022\\metric_RMSE_by_city_2020-2022.npy')

# RMSE do modelo GCRN
rmse_gcrn = np.load(results_gcrn + f'\\2020-2022\\metric_RMSE_by_city_2020-2022.npy')

# R2 e RMSE do modelo Prophet
df_city = pd.read_csv(results_prophet + '\\R2_RMSE_LSTM_Prophet.csv', sep=';')
df_city.drop(['LSTM R2', 'Prophet R2'], axis=1, inplace=True)

# Completa o DataFrame com os resultados dos modelos GCLSTM e GCRN
df_city['GCLSTM RMSE'] = np.mean(rmse_gclstm, axis=0)
df_city['GCRN RMSE'] = np.mean(rmse_gcrn, axis=0)

# Insere os resultados no DataFrame df_city
df_city['Avg. Yhat GCLSTM'] = np.mean(pred_gclstm, axis=0)
df_city['Avg. Yhat GCRN'] = np.mean(pred_gcrn, axis=0)
df_city['Avg. Yhat Prophet'] = np.mean(pred_prophet, axis=0)
df_city['Avg. Yhat LSTM'] = np.mean(pred_lstm, axis=0)
df_city['Y'] = np.mean(y_real, axis=0)

# Número de lags
lags = 14

# Criar DataFrame com a coluna 'Days'
df_time = pd.DataFrame(
    {'Days': [datetime.strptime('03/05/2022', '%d/%m/%Y') + timedelta(days=i) for i in
              list(range(lags - 1, lags * y_real.shape[0], lags))]})

# Adicionar colunas de R2 para cada modelo
modelos = ['GCLSTM', 'GCRN', 'Prophet']
for modelo in modelos:
    pred_modelo = globals()[f'pred_{modelo.lower()}']  # Obtém a variável pred_gclstm, pred_gcrn ou pred_prophet
    r2_modelo = [r2_score(y_real[i, :], pred_modelo[i, :]) for i in range(y_real.shape[0])]
    df_time[f'R2_{modelo}'] = r2_modelo

# Inicie o aplicativo Dash
app = dash.Dash(__name__)

# Defina o layout do seu dashboard
app.layout = html.Div([
    html.H1("Dashboard de Resultados de Predição"),

    # Gráfico para RMSE
    dcc.Graph(
        id='resultados-rmse',
        figure={
            'data': [
                {'x': df_city.NOMENUM, 'y': df_city[f'{modelo} RMSE'], 'type': 'bar', 'name': f'{modelo} RMSE'} for
                modelo in modelos
            ],
            'layout': {
                'title': 'Resultados de RMSE por Cidade',
                'barmode': 'group',
                'yaxis': {'title': 'RMSE'},
                'margin': {'l': 50, 'r': 50, 't': 30, 'b': 30},  # Adiciona um espaço vertical menor
            }
        }
    ),

    # Adiciona espaço vertical entre os gráficos
    html.Div(style={'height': '30px'}),

    # Gráfico para Avg. Yhat
    dcc.Graph(
        id='resultados-yhat',
        figure={
            'data': [
                        {'x': df_city.NOMENUM, 'y': df_city[f'Avg. Yhat {modelo}'], 'type': 'bar',
                         'name': f'Avg. Yhat {modelo}'}
                        for modelo in modelos
                    ] + [
                        {'x': df_city.NOMENUM, 'y': df_city['Y'], 'type': 'bar', 'name': 'Y'}
                    ],
            'layout': {
                'title': 'Resultados de Avg. Yhat por Cidade',
                'barmode': 'group',
                'yaxis': {'title': 'Valor'},
                'margin': {'l': 50, 'r': 50, 't': 30, 'b': 30},  # Adiciona um espaço vertical menor
            }
        }
    ),

    # Adiciona espaço vertical entre os gráficos
    html.Div(style={'height': '30px'}),

    # Gráfico para R2
    dcc.Graph(
        id='resultados-r2',
        figure={
            'data': [
                {'x': df_time['Days'], 'y': df_time[f'R2_{modelo}'], 'type': 'line', 'name': f'R2_{modelo}'}
                for modelo in modelos
            ],
            'layout': {
                'title': 'Resultados de R2 ao longo do tempo',
                'yaxis': {'title': 'R2'},
                'margin': {'l': 50, 'r': 50, 't': 30, 'b': 30},  # Adiciona um espaço vertical menor
            }
        }
    ),
])

# Execute o aplicativo
if __name__ == '__main__':
    app.run_server(debug=True)
