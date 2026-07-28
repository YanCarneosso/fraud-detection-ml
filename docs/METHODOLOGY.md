# Metodologia

## Formulação

Classificação binária com forte desbalanceamento. `Class=1` representa fraude.

## Experimentos

O baseline majoritário é comparado com Regressão Logística, Random Forest e XGBoost.
As estratégias incluem dados originais, pesos de classe, RandomUnderSampler e SMOTE.

Scaler e reamostragem permanecem dentro do pipeline. Dessa forma, em validação cruzada
cada transformação aprende apenas com o fold de treino.

## Seleção

Hiperparâmetros usam validação cruzada estratificada no treino. O threshold é escolhido
na validação por custo configurável. PR-AUC e recall servem como critérios adicionais.
O conjunto de teste é consultado uma única vez.

## Avaliação

As métricas principais são precisão, recall, F1 da fraude, PR-AUC, ROC-AUC, falsos
positivos, falsos negativos e custo estimado. Probabilidades são usadas nas métricas de
ranking e curvas.

## Reprodutibilidade

Operações estocásticas usam `random_state=42`. O pipeline completo, ordem das features,
threshold e métricas finais são persistidos.

