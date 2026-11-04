# Модель данных

Каждая исходная таблица сводится до одной строки на клиента своим SQL-файлом, затем всё собирается
в витрину `data_mart`: 307 511 клиентов, 66 столбцов.

```mermaid
flowchart LR
    subgraph raw["Raw tables (DuckDB)"]
        APP["application_train<br/>307 511 applications"]
        BUR["bureau<br/>1.7M bureau credits"]
        BB["bureau_balance<br/>27.3M months"]
        PREV["previous_application<br/>1.7M applications"]
        INST["installments_payments<br/>13.6M payments"]
        POS["pos_cash_balance<br/>10.0M months"]
        CC["credit_card_balance<br/>3.8M months"]
    end

    subgraph feat["Feature tables (one row per client)"]
        FA["feat_application<br/>10_feat_application.sql"]
        FB["feat_bureau<br/>20_feat_bureau.sql"]
        FP["feat_prev<br/>30_feat_prev.sql"]
        FI["feat_installments<br/>40_feat_installments.sql"]
        FPO["feat_pos<br/>50_feat_pos.sql"]
        FC["feat_cc<br/>60_feat_cc.sql"]
    end

    APP --> FA
    BB -->|month → credit| BUR
    BUR -->|credit → client| FB
    PREV --> FP
    INST -->|payment → installment → client| FI
    POS -->|month → loan → client| FPO
    CC -->|month → card → client| FC

    FA --> DM
    FB --> DM
    FP --> DM
    FI --> DM
    FPO --> DM
    FC --> DM
    DM["data_mart<br/>70_data_mart.sql<br/>307 511 rows, 66 columns<br/>+ history flags, sample split"]
    DM --> PQ[("data/processed/<br/>data_mart.parquet")]
```

| Таблица признаков | Клиентов | Доля от всех |
|---|---|---|
| feat_application | 307 511 | 100% |
| feat_bureau | 263 491 | 85.7% |
| feat_prev | 291 057 | 94.7% |
| feat_installments | 291 643 | 94.8% |
| feat_pos | 289 444 | 94.1% |
| feat_cc | 86 905 | 28.3% |

Разбиение: train 184 506, valid 61 502, test 61 503 (60 / 20 / 20), доля дефолтов 8.07% в каждой части.
