# Модель данных

Raw-таблицы Home Credit агрегируются до зерна клиента (`SK_ID_CURR`) отдельными SQL-файлами,
затем соединяются LEFT JOIN в витрину `data_mart` (307 511 строк, одна строка на заявку из
`application_train`). Каждая таблица признаков уже на зерне клиента, поэтому джойн не размножает строки.

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

## Таблицы признаков

| Таблица | Клиентов | Покрытие | Сверка с источником |
|---|---|---|---|
| feat_application | 307 511 | 100% | — |
| feat_bureau | 263 491 | 85.7% | 1 465 325 кредитов |
| feat_prev | 291 057 | 94.7% | 1 413 701 заявка |
| feat_installments | 291 643 | 94.8% | 11 026 627 взносов |
| feat_pos | 289 444 | 94.1% | 8 543 375 месяцев |
| feat_cc | 86 905 | 28.3% | 3 227 965 месяцев |

## Разбиение выборки

Столбец `sample`: train 184 506 (60%), valid 61 502 (20%), test 61 503 (20%).
Bad rate 0.0807 во всех трёх частях (стратификация по `TARGET`).
