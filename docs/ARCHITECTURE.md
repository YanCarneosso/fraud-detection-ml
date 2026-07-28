# Arquitetura

```mermaid
flowchart LR
    A["creditcard.csv"] --> B["Validação de esquema e qualidade"]
    B --> C["Divisão estratificada 70/15/15"]
    C --> D["Treino"]
    C --> E["Validação"]
    C --> F["Teste isolado"]
    D --> G["Feature engineering"]
    G --> H["Scaler / balanceamento"]
    H --> I["Modelos e CV"]
    I --> E
    E --> J["Seleção de modelo e threshold"]
    J --> F
    F --> K["Métricas finais"]
    K --> L["Pipeline + metadados"]
    K --> M["Relatório HTML"]
    K --> N["Explicações SHAP"]
```

## Componentes

- `data_processing.py`: entrada, validação, divisão e EDA;
- `feature_engineering.py`: transformações determinísticas reutilizáveis;
- `train.py`: candidatos, balanceamento, CV, comparação e persistência;
- `evaluate.py`: métricas, custo, thresholds e gráficos;
- `explain.py`: explicabilidade global e local;
- `inference.py`: contrato de predição para novas transações;
- `report.py`: relatório executivo baseado nos artefatos reais;
- `main.py`: orquestração do fluxo.

O teste é acessado somente depois da seleção do modelo e do threshold.

