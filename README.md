# Projeto para a 12ª Conferência BRACIS 2023

Este projeto é uma publicação para a 12ª Conferência BRACIS do ano de 2023.

## Resumo do Artigo

Neste estudo, examinamos o impacto da mobilidade humana na transmissão da COVID-19, uma doença altamente contagiosa que se espalhou rapidamente em todo o mundo. Para investigar isso, construímos uma rede de mobilidade que captura os padrões de movimento entre cidades brasileiras e a integramos com dados de séries temporais de registros de infecção por COVID-19. Nossa abordagem considera a interação entre os movimentos das pessoas e a propagação do vírus. Utilizamos duas redes neurais baseadas em Graph Convolutional Network (GCN), que aproveitam entradas de dados espaciais e temporais, para prever séries temporais em cada cidade, levando em conta a influência das cidades vizinhas. Em comparação, avaliamos modelos LSTM e Prophet que não capturam dependências de séries temporais. Entre os modelos, o Prophet alcança o melhor RMSE médio de 482,95 com um mínimo de 1,49, enquanto o LSTM tem o desempenho mais baixo, apesar de ter um RMSE mínimo baixo. Os modelos GCRN e GCLSTM apresentam valores médios de erro RMSE de 3059,5 e 3583,88, respectivamente, com os menores valores de desvio padrão para erros RMSE em 500,39 e 452,59. Embora o modelo Prophet demonstre um desempenho superior, seu valor máximo de RMSE de 52.058,21 é dez vezes maior do que o maior valor observado nos modelos de Graph Convolutional Networks (GCNs). Com base em nossas descobertas, concluímos que os modelos GCNs produzem resultados mais estáveis em comparação com os modelos avaliados.

## Instalação

A instalação deve ser feita utilizando o arquivo `requirements.txt` fornecido. É recomendado utilizar um interpretador pelo conda, pela facilidade de instalação dos recursos e minimizar problemas que possam ocorrer.

## Uso

Para utilizar, basta executar o código `compute_predictions.py`, que irá gerar os resultados para os modelos GCNs. E o código `forecast_lstm_prophet.py`, que gera os resultados para os modelos LSTM e Prophet.

## Resultados

Os resultados podem ser encontrados na pasta "results" nas subpastas particionadas para os modelos. 