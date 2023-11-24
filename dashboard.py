import os
from datetime import datetime, timedelta
import dash
import numpy as np
from dash import dcc, html
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

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

# Criando um DataFrame
df_stats_pred = pd.DataFrame({
    'Model': ['GCLSTM', 'GCRN', 'Prophet', 'LSTM'],
    'Max': [np.max(pred_gclstm), np.max(pred_gcrn), np.max(pred_prophet), np.max(pred_lstm)],
    'Min': [np.min(pred_gclstm), np.min(pred_gcrn), np.min(pred_prophet), np.min(pred_lstm)],
    'Mean': [np.mean(pred_gclstm), np.mean(pred_gcrn), np.mean(pred_prophet), np.mean(pred_lstm)],
    'Std': [np.std(pred_gclstm), np.std(pred_gcrn), np.std(pred_prophet), np.std(pred_lstm)],
})

# Exibindo o DataFrame de estatísticas
print(df_stats_pred)

# Número de lags
lags = 14

# Criar DataFrame com a coluna 'Days'
df_time = pd.DataFrame(
    {'Days': [datetime.strptime('03/05/2022', '%d/%m/%Y') + timedelta(days=i) for i in
              list(range(lags - 1, lags * y_real.shape[0], lags))]})

# Adicionar colunas de R2 para cada modelo
models = ['GCLSTM', 'GCRN', 'LSTM', 'Prophet']
for model in models:
    pred_modelo = globals()[f'pred_{model.lower()}']  # Obtém a variável pred_gclstm, pred_gcrn ou pred_prophet
    r2_modelo = [r2_score(y_real[i, :], pred_modelo[i, :]) for i in range(y_real.shape[0])]
    df_time[f'R2_{model}'] = r2_modelo

# R2 e RMSE do modelo Prophet
df_city = pd.read_csv(results_prophet + '\\R2_RMSE_LSTM_Prophet.csv', sep=';')
df_city.drop(['LSTM R2', 'Prophet R2'], axis=1, inplace=True)

# Completa o DataFrame com os resultados dos modelos GCLSTM e GCRN
df_city['GCLSTM RMSE'] = np.array(
    [mean_squared_error(y_real[:, ind], pred_gclstm[:, ind], squared=False) for ind in range(y_real.shape[1])])
df_city['GCRN RMSE'] = np.array(
    [mean_squared_error(y_real[:, ind], pred_gcrn[:, ind], squared=False) for ind in range(y_real.shape[1])])

# Insere os resultados no DataFrame df_city
df_city['Avg. Yhat GCLSTM'] = np.mean(pred_gclstm, axis=0)
df_city['Avg. Yhat GCRN'] = np.mean(pred_gcrn, axis=0)
df_city['Avg. Yhat Prophet'] = np.mean(pred_prophet, axis=0)
df_city['Avg. Yhat LSTM'] = np.mean(pred_lstm, axis=0)
df_city['Y'] = np.mean(y_real, axis=0)

# Calculando o vetor rmse para cada modelo
dfs = []
for i, model in enumerate(models):
    predictions = globals()[f'pred_{model.lower()}']
    rmse_model = np.array(
        [mean_squared_error(y_real[ind], predictions[ind], squared=False) for ind in range(y_real.shape[0])])
    dfs.append(pd.DataFrame({
        'Model': [model] * len(rmse_model),
        'RMSE': rmse_model,
    }))

# Concatenando os DataFrames de cada modelo
df_rmse = pd.concat(dfs, ignore_index=True)

# Criando o DataFrame de estatísticas
df_stats_rmse = pd.DataFrame({
    'Model': models,
    'Max': df_rmse.groupby('Model')['RMSE'].max(),
    'Min': df_rmse.groupby('Model')['RMSE'].min(),
    'Mean': df_rmse.groupby('Model')['RMSE'].mean(),
    'Std': df_rmse.groupby('Model')['RMSE'].std(),
})

# Exibindo o DataFrame de estatísticas
print(df_stats_rmse)

# Calculando o vetor rmse para cada modelo
dfs = []
for i, model in enumerate(models):
    predictions = globals()[f'pred_{model.lower()}']
    rmse_model = np.array(
        [mean_squared_error(y_real[:, ind], predictions[:, ind], squared=False) for ind in range(y_real.shape[1])])
    dfs.append(pd.DataFrame({
        'Model': [model] * len(rmse_model),
        'RMSE': rmse_model,
    }))

# Concatenando os DataFrames de cada modelo
df_rmse_city = pd.concat(dfs, ignore_index=True)

# Criando o DataFrame de estatísticas
df_stats_rmse_city = pd.DataFrame({
    'Model': models,
    'Max': df_rmse_city.groupby('Model')['RMSE'].max(),
    'Min': df_rmse_city.groupby('Model')['RMSE'].min(),
    'Mean': df_rmse_city.groupby('Model')['RMSE'].mean(),
    'Std': df_rmse_city.groupby('Model')['RMSE'].std(),
})

# Exibindo o DataFrame de estatísticas
print(df_stats_rmse_city)

# rmse_gclstm_all = np.load(results_gclstm + f'\\2020-2022\\metric_RMSE_all_2020-2022.npy')
#
# rmse_gcrn_all = np.load(results_gcrn + f'\\2020-2022\\metric_RMSE_all_2020-2022.npy')
#
# rmse_lstm_all = df_city['LSTM RMSE']
#
# rmse_prophet_all = df_city['Prophet RMSE']

# models = ['GCLSTM', 'GCRN', 'LSTM', 'Prophet']
# dfs = []
#
# for model in models:
#     predictions = globals()[f'rmse_{model.lower()}_all']
#     rmse_model = predictions
#     model_stats = {
#         'Model': model,
#         'Max': np.max(rmse_model),
#         'Min': np.min(rmse_model),
#         'Mean': np.mean(rmse_model),
#         'Std': np.std(rmse_model),
#     }
#     dfs.append(model_stats)
#
# # Criando o DataFrame de estatísticas
# df_stats_error = pd.DataFrame(dfs)
#
# # Exibindo o DataFrame de estatísticas
# print(df_stats_error)


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
                {'x': df_city.NOMENUM, 'y': df_city[f'{model} RMSE'], 'type': 'bar', 'name': f'{model} RMSE'} for
                model in models
            ],
            'layout': {
                'title': 'Resultados de RMSE por Cidade',
                'barmode': 'group',
                'yaxis': {'title': 'RMSE'},
                'margin': {'l': 50, 'r': 50, 't': 30, 'b': 150},
                'xaxis': {'tickangle': -45},
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
                        {'x': df_city.NOMENUM, 'y': df_city[f'Avg. Yhat {model}'], 'type': 'bar',
                         'name': f'Avg. Yhat {model}'}
                        for modelo in models
                    ] + [
                        {'x': df_city.NOMENUM, 'y': df_city['Y'], 'type': 'bar', 'name': 'Y'}
                    ],
            'layout': {
                'title': 'Resultados de Avg. Yhat por Cidade',
                'barmode': 'group',
                'yaxis': {'title': 'Valor'},
                'margin': {'l': 50, 'r': 50, 't': 30, 'b': 150},
                'xaxis': {'tickangle': -45},
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
                {'x': df_time['Days'], 'y': df_time[f'R2_{model}'], 'type': 'line', 'name': f'R2_{model}'}
                for model in models
            ],
            'layout': {
                'title': 'Resultados de R2 ao longo do tempo',
                'yaxis': {'title': 'R2'},
                'margin': {'l': 50, 'r': 50, 't': 30, 'b': 80},
            }
        }
    ),
])

# Execute o aplicativo
if __name__ == '__main__':
    app.run_server(debug=True)
