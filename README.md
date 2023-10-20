# Projeto para a 12ª Conferência BRACIS 2023

Este projeto é uma publicação para a 12ª Conferência BRACIS do ano de 2023.

## Resumo do Artigo

Neste estudo, examinamos o impacto da mobilidade humana na transmissão da COVID-19, uma doença altamente contagiosa que se espalhou rapidamente em todo o mundo. Para investigar isso, construímos uma rede de mobilidade que captura os padrões de movimento entre cidades brasileiras e a integramos com dados de séries temporais de registros de infecção por COVID-19. Nossa abordagem considera a interação entre os movimentos das pessoas e a propagação do vírus. Utilizamos duas redes neurais baseadas em Graph Convolutional Network (GCN), que aproveitam entradas de dados espaciais e temporais, para prever séries temporais em cada cidade, levando em conta a influência das cidades vizinhas. Em comparação, avaliamos modelos LSTM e Prophet que não capturam dependências de séries temporais. Entre os modelos, o Prophet alcança o melhor RMSE médio de 482,95 com um mínimo de 1,49, enquanto o LSTM tem o desempenho mais baixo, apesar de ter um RMSE mínimo baixo. Os modelos GCRN e GCLSTM apresentam valores médios de erro RMSE de 3059,5 e 3583,88, respectivamente, com os menores valores de desvio padrão para erros RMSE em 500,39 e 452,59. Embora o modelo Prophet demonstre um desempenho superior, seu valor máximo de RMSE de 52.058,21 é dez vezes maior do que o maior valor observado nos modelos de Graph Convolutional Networks (GCNs). Com base em nossas descobertas, concluímos que os modelos GCNs produzem resultados mais estáveis em comparação com os modelos avaliados.

**Link para o artigo publicado:** [Artigo BRACIS 2023](https://link.springer.com/chapter/10.1007/978-3-031-45392-2_24)

**Resumo em Inglês:**
In this study, we examine the impact of human mobility on the transmission of COVID-19, a highly contagious disease that has rapidly spread worldwide. To investigate this, we construct a mobility network that captures movement patterns between Brazilian cities and integrate it with time series data of COVID-19 infection records. Our approach considers the interplay between people’s movements and the spread of the virus. We employ two neural networks based on Graph Convolutional Network (GCN), which leverage spatial and temporal data inputs, to predict time series at each city while accounting for the influence of neighboring cities. In comparison, we evaluate LSTM and Prophet models that do not capture time series dependencies. By utilizing RMSE (Root Mean Square Error), we quantify the discrepancy between the actual number of COVID-19 cases and the predicted number of cases by the model among the models. Prophet achieves the best average RMSE of 482.95 with a minimum of 1.49, while LSTM performs the least despite having a low minimum RMSE. The GCRN and GCLSTM models exhibit mean RMSE error values of 3059.5 and 3583.88, respectively, with the lowest standard deviation values for RMSE errors at 500.39 and 452.59. Although the Prophet model demonstrates superior performance, its maximum RMSE value of 52,058.21 is ten times higher than the highest value observed in the Graph Convolutional Networks (GCNs) models. Based on our findings, we conclude that GCNs models yield more stable results compared to the evaluated models.

**Palavras-chave:**
- Previsão de Séries Temporais
- Redes Neurais Baseadas em Grafos
- Redes de Mobilidade
- COVID-19

**Link para o GitHub do CSILAB-UFOP:** [GitHub CSILAB-UFOP](https://github.com/ufopcsilab/projects)

**Site CSILAB-UFOP:** [Site CSILAB-UFOP](https://csilab.ufop.br/)

**CSI Lab:**
O Laboratório de Computação de Sistemas Inteligentes (CSI Lab) é um espaço reservado para atividades de pesquisa de bolsistas de graduação da UFOP e estudantes de doutorado do PPGCC/DECOM.

## Instalação

A instalação deve ser feita utilizando o arquivo `requirements.txt` fornecido. É recomendado utilizar um interpretador pelo conda, pela facilidade de instalação dos recursos e minimizar problemas que possam ocorrer.

## Uso

Para utilizar, executando o código `compute_predictions.py` gera os resultados para os modelos based-GCNs. O código `forecast_lstm_prophet.py`, que gera os resultados para os modelos LSTM e Prophet.

O script `create_networks.py` cria as redes de mobilidade com diversos pesos.

## Pastas

- `/raw_data`: Contém os dados originais utilizados no projeto.
- `/models`: Armazena os códigos dos modelos baseados em GCNs.
- `/networks`: Inclui os arquivos `.GraphML`, que representam os grafos do mapa do Brasil, com diferentes pesos em suas arestas.
- `/save`: Contém os datasets preprocessados.
- `/results`: Armazena resultados, incluindo figuras e arquivos gerados.

