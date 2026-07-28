# Registro de decisões técnicas

## Remover `Amount` e `Time` originais

`Amount` é substituído por `log1p(Amount)`, reduzindo assimetria. `Time` dá origem a hora
cíclica (`sin` e `cos`). As colunas originais são removidas para reduzir redundância.

## Manter SMOTE dentro do pipeline

SMOTE antes da divisão permitiria que observações sintéticas relacionadas aparecessem
em treino e avaliação. No pipeline, a reamostragem ocorre somente sobre dados de treino.

## Não selecionar por acurácia

Com aproximadamente 0,17% de fraudes, um classificador majoritário pode ter acurácia
elevada sem detectar uma única fraude. A seleção privilegia custo, PR-AUC e recall.

## Threshold separado do modelo

O threshold operacional é escolhido na validação e armazenado nos metadados. Isso
separa capacidade de ranking da decisão de negócio e permite ajustar custos.

## Não versionar dataset e modelos

Dados e binários grandes ficam fora do Git. O repositório contém código, testes,
documentação e um relatório demonstrativo autocontido.

