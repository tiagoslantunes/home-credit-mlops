# Pré-processamento — Home Credit (application_train)

Resumo de todo o pré-processamento: ordem das operações, o que é feito a cada
variável, e **onde acontece o split e como é usado**. Tudo o que aqui está é o
que está efetivamente implementado nas pipelines Kedro.

---

## 1. Visão geral — ordem das pipelines

```
01_raw/application_train.csv
        │
        ▼
[data_quality]   ──► application_train_validated      (gate Great Expectations; dados inalterados)
        │
        ▼
[data_split]     ──► application_train_split           (80%)  ┐ split ESTRATIFICADO por TARGET
        │            application_validation_split       (20%)  ┘ (preserva o imbalance ~11.4:1)
        ▼
[data_cleaning]  ──► cleaning_params  (fit SÓ no train)
        │            application_train_cleaned
        │            application_validation_cleaned
        │            application_test_cleaned        (holdout/produção, SEM passar pelo split)
        ▼
[data_feat_engineering] → [model_train] → ...           (ainda por implementar)
```

`application_test` (o batch de produção/holdout) **não passa pelo split** — é
limpo diretamente do raw com o **mesmo** `cleaning_params` (fit no treino), tal
como qualquer batch de serving.

**Princípio central:** o **split é feito ANTES da limpeza**. Assim todas as
estatísticas de imputação (medianas, modas) e todas as decisões de remoção de
colunas são aprendidas **apenas na partição de treino**, nunca vendo a validação
nem dados futuros de produção → **sem data leakage**. (Corrige a ordem do
exemplo da week_04, onde o pré-processamento via os dados todos.)

---

## 2. Onde acontece o split e como é usado

| Momento | O quê | Vê que dados? |
|---|---|---|
| `data_split` | Parte o **total validado** (`application_train_validated`) em treino (80%) / validação (20%), estratificado por `TARGET` | dados validados pela GX |
| `data_cleaning › fit_cleaning` | **Aprende** drop-lists + medianas/modas + bounds | **só `application_train_split`** |
| `data_cleaning › apply_cleaning` | **Aplica** (replay) a mesma transformação | treino, validação, **test/produção** e qualquer batch futuro |

O artefacto **`cleaning_params.pkl`** é a ponte: é produzido pelo `fit` (no treino)
e consumido pelo `apply` para limpar treino, validação e produção **de forma
idêntica**. Em serving, basta voltar a chamar `apply_cleaning` com este artefacto.

---

## 3. Relação nó ↔ pipeline ↔ catálogo

### Pipeline `data_split`
`src/home_credit_mlops/pipelines/data_split/`

| Nó | Função | Inputs | Outputs |
|---|---|---|---|
| `split_application_train_node` | `split_data` | `application_train`, `params:data_split` | `application_train_split`, `application_validation_split` |

### Pipeline `data_cleaning`
`src/home_credit_mlops/pipelines/data_cleaning/`

| Nó | Função | Inputs | Outputs |
|---|---|---|---|
| `fit_cleaning_node` | `fit_cleaning` | `application_train_split`, `params:data_cleaning`, `params:numerical_rules` | `cleaning_params` |
| `clean_train_node` | `apply_cleaning` | `application_train_split`, `cleaning_params`, `params:data_cleaning` | `application_train_cleaned` |
| `clean_validation_node` | `apply_cleaning` | `application_validation_split`, `cleaning_params`, `params:data_cleaning` | `application_validation_cleaned` |
| `clean_test_node` | `apply_cleaning` | `application_test`, `cleaning_params`, `params:data_cleaning` | `application_test_cleaned` |

Parâmetros em `conf/base/parameters_data_split.yml` e
`conf/base/parameters_data_cleaning.yml`; bounds reutilizam
`params:numerical_rules` (de `parameters_data_quality.yml`). Datasets em
`conf/base/catalog.yml`.

---

## 4. `data_split` — o que faz

- `train_test_split` com `test_size=0.2`, `random_state=42`, `shuffle=True`.
- `stratify=TARGET` → a taxa de positivos (0.0807) é idêntica em raw / treino / validação.
- Reset de índice em ambas as partições.

---

## 5. `data_cleaning` — ordem das operações

### `fit_cleaning` (aprende, só no treino)
1. **Sentinel `DAYS_EMPLOYED`** → NaN + flag (ver §6).
2. **Drop > 45% missing** (colunas com mais de 45% de nulos).
3. **Drop alta correlação** `|r| > 0.9` (remove a 2ª coluna de cada par numérico).
4. **Drop near-constant** (valor dominante ≥ 95% dos não-nulos).
5. Calcula **mediana** de cada coluna numérica e **moda** de cada categórica.
6. Calcula **bounds** (min/max) a partir de `params:numerical_rules`.

> Colunas protegidas dos drops: `SK_ID_CURR`, `TARGET`, e todas as que têm
> decisão de imputação explícita (assim o conjunto de features é estável e as
> imputações têm sempre coluna onde atuar).
>
> Resultado: **122 → 48 colunas**, 75 removidas (49 high-missing, 2 high-corr, 24 near-constant).

### `apply_cleaning` (replay, em treino/val/test/batch)
1. Sentinel `DAYS_EMPLOYED` → NaN + flag.
2. Drop das colunas aprendidas no fit (`errors="ignore"`).
3. **Zero-fill** semântico (bureau/social-circle).
4. **Constant-fill** categórico (`OCCUPATION_TYPE → "Unknown"`).
5. **Mediana-inteira** (colunas de contagem `CNT_FAM_MEMBERS`, `DAYS_LAST_PHONE_CHANGE`).
6. **Fallback genérico**: numérica → mediana do treino; categórica → moda do treino.
7. **Capping** aos bounds das regras de qualidade (salvaguarda de drift).
8. **Cast de tipos**: colunas de contagem/dias (ver §6) arredondadas e convertidas para `int64`.

Garantias no fim: **0 NaN** e tipos coerentes (ids/flags/contagens/dias em `int64`,
montantes/scores em `float64`, categorias em `string`). Aplica-se de forma
idêntica a treino, validação e test/produção.

---

## 6. O que é feito a cada variável

### Remoção estrutural (aprendida no treino)
| Regra | Ação | Exemplos removidos |
|---|---|---|
| > 45% missing | remover coluna | `EXT_SOURCE_1`, colunas de habitação (`*_AVG/_MEDI/_MODE`), `OWN_CAR_AGE`, … |
| `\|r\| > 0.9` | remover a 2ª coluna do par | redundâncias numéricas correlacionadas |
| dominante ≥ 95% | remover coluna | maioria dos `FLAG_DOCUMENT_*`, `AMT_REQ_CREDIT_BUREAU_HOUR/DAY/WEEK`, flags quase-constantes |

### Tratamento de sentinel (decisão de metodologia, não estava no analises.ipynb)
| Variável | Ação |
|---|---|
| `DAYS_EMPLOYED` | valor `365243` ("não empregado", ~18%) → **NaN** + cria flag `DAYS_EMPLOYED_ANOM` (1/0); depois imputado pela mediana do treino |

### Imputação (decisões do analises.ipynb)
| Variável(eis) | Estratégia | Porquê |
|---|---|---|
| `AMT_ANNUITY` | mediana (treino) | residual (12 missing), rácio não fiável p/ Cash loans |
| `EXT_SOURCE_2`, `EXT_SOURCE_3` | mediana (treino) | fórmula opaca; fontes ~independentes |
| `DAYS_EMPLOYED` | mediana (treino) | após sentinel→NaN |
| `AMT_REQ_CREDIT_BUREAU_MON` / `_QRT` / `_YEAR` | **0** | ausência de registo = 0 consultas (mesma fonte) |
| `OBS_30_CNT_SOCIAL_CIRCLE`, `DEF_30_CNT_SOCIAL_CIRCLE`, `DEF_60_CNT_SOCIAL_CIRCLE` | **0** | ausência = 0 eventos no círculo social |
| `CNT_FAM_MEMBERS`, `DAYS_LAST_PHONE_CHANGE` | mediana arredondada a **inteiro** | são contagens/dias inteiros |
| `OCCUPATION_TYPE` | nova categoria **"Unknown"** | missingness (31%) é sinal preditivo |
| `NAME_TYPE_SUITE` | moda (treino) → "Unaccompanied" | residual (0.4%), sem significado especial |

### Fallback de produção (qualquer outra variável)
| Tipo | Estratégia |
|---|---|
| numérica não listada acima | mediana do treino |
| categórica não listada acima | moda do treino |

> Garante que um batch de serving nunca fica com NaN, mesmo em colunas não enumeradas.

### Capping defensivo (salvaguarda de drift)
Cada coluna numérica é "clipped" aos limites min/max das regras de qualidade
(`params:numerical_rules`), ex.: `EXT_SOURCE_* ∈ [0,1]`, `CNT_CHILDREN ≥ 0`,
`DAYS_* ≤ 0`. No treino (já validado) nunca corta; só atua em valores
drifted de validação/produção.

### Cast de tipos (consistência dtype)
Colunas conceptualmente inteiras que o pandas tinha promovido a `float` por
terem NaN/sentinel antes da imputação são, no fim (já sem NaN), arredondadas e
convertidas para `int64` (`cast_int_cols`):
`DAYS_EMPLOYED`, `DAYS_REGISTRATION`, `OBS_30/DEF_30/OBS_60/DEF_60_CNT_SOCIAL_CIRCLE`,
`AMT_REQ_CREDIT_BUREAU_MON/QRT/YEAR`. As 6 colunas genuinamente contínuas
(`AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `REGION_POPULATION_RELATIVE`,
`EXT_SOURCE_2/3`) ficam `float64`.

---

## 7. Outputs

| Layer | Ficheiro | Shape |
|---|---|---|
| `02_intermediate` | `application_train_validated.csv` | (307511, 122) |
| `02_intermediate` | `application_train_split.csv` | (246008, 122) |
| `02_intermediate` | `application_validation_split.csv` | (61503, 122) |
| `03_primary` | `application_train_cleaned.csv` | (246008, 48) — 0 NaN |
| `03_primary` | `application_validation_cleaned.csv` | (61503, 48) — 0 NaN |
| `03_primary` | `application_test_cleaned.csv` | (48744, 47) — 0 NaN, sem `TARGET` |
| `04_feature` | `cleaning_params.pkl` | artefacto fit (drop-lists + medianas/modas + bounds) |
