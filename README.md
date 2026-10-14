# Recomendação de freelancers para vagas da UmFreela

Modelo de Machine Learning que estima a chance de cada freelancer dar certo em uma vaga e, a partir disso, **recomenda quem convidar** e **qual ação tomar** em cada vaga publicada na UmFreela, plataforma que conecta restaurantes, bares e hotéis a freelancers do setor (garçom, cozinheiro, bartender e outras funções).

**Aluno:** Jonathan Gonçalves Possa · Especialização em Ciência de Dados e Inteligência Artificial (AMF)
**Trabalho:** Módulo 8, Machine Learning Aplicado · Alternativa A (Machine Learning tradicional)

**Entrega principal:** [`recomendacao_freelancers.ipynb`](recomendacao_freelancers.ipynb). O mesmo relatório, já executado, também está em [`reports/recomendacao_freelancers.html`](reports/recomendacao_freelancers.html).

[Abrir no Google Colab](https://colab.research.google.com/github/Jonathanpossa/umfreela-recomendacao-freelancers/blob/main/recomendacao_freelancers.ipynb)

## Problema

Quando um restaurante publica uma vaga, recebe em média 7 candidaturas e precisa escolher rápido. Hoje a lista é ordenada pela nota média, o que ignora distância, faltas anteriores, disponibilidade e o histórico do freelancer com aquele restaurante. Algumas vagas também ficam sem ninguém porque os freelancers certos não chegam a vê-las.

**Problema de ML:** classificação binária. Para cada par (vaga, freelancer), prever a probabilidade de um **match bem-sucedido**: o freelancer é contratado, comparece ao turno e recebe nota 4 ou 5.

**Recomendação construída sobre o modelo**, em dois momentos:

1. **Na publicação:** todos os freelancers elegíveis (mesma cidade, função principal ou secundária) são avaliados, e os primeiros do ranking recebem convite.
2. **Depois das candidaturas:** os candidatos são ordenados para o restaurante, e a maior probabilidade entre eles define a ação.

| Faixa (maior probabilidade entre os candidatos) | Ação recomendada |
|---|---|
| Boa opção (≥ 0,33) | destacar o 1º do ranking para o restaurante contratar |
| Medianos (0,15 a 0,33) | mostrar o ranking e convidar mais 3 freelancers elegíveis |
| Risco (< 0,15) | nova rodada de convites aos 5 melhores elegíveis + aviso ao restaurante |

## Dataset

Os dados reais da plataforma têm informações pessoais de freelancers (LGPD). Por isso, o conjunto foi **gerado por simulação** com as regras do negócio (`src/gerar_dados.py`, semente 42), incluindo problemas de qualidade parecidos com os de produção. Os arquivos estão em [`data/`](data/):

| Arquivo | Linhas | Conteúdo |
|---|---|---|
| `freelancers.csv` | 1.800 | cidade, localização, funções, experiência, turnos concluídos, nota, taxas de falta e cancelamento, pretensão de diária, disponibilidade, curso, tempo de resposta, perfil |
| `restaurantes.csv` | 320 | cidade, localização, segmento |
| `vagas.csv` | 7.000 | função, datas de publicação e do turno, período, valor da diária, antecedência |
| `candidaturas.csv` | 49.727 | par (vaga, freelancer), distância, contratação e alvo `match_sucesso` |

Período simulado: 01/10/2025 a 30/09/2026, em seis cidades do RS (Santa Maria, Porto Alegre, Santiago, Uruguaiana, Itaqui e Quarta Colônia).

## Metodologia

| Etapa | O que foi feito |
|---|---|
| Análise exploratória | tipos, faltantes, duplicados (735 candidaturas), outliers (distância 999 km, pretensão R$ 0 ou R$ 9.999), cidades fora do padrão, desbalanceamento do alvo (9,4% positivos) |
| Limpeza | remoção de duplicados, padronização de cidades, distância recalculada pelas coordenadas, pretensão inválida substituída pela mediana da função |
| Atributos | 22 atributos, entre eles: função principal ou secundária, diferença entre diária e pretensão, disponibilidade no turno, taxas de falta e cancelamento, e **matches anteriores no mesmo restaurante** (calculado só com vagas anteriores, sem vazamento) |
| Codificação e padronização | One-Hot Encoding (função, cidade, período, segmento), StandardScaler e imputação pela mediana, tudo dentro de um `Pipeline` |
| Divisão | **temporal**: treino de out/2025 a jun/2026 e teste de jul a set/2026; validação cruzada de 5 dobras **agrupada por vaga** (`GroupKFold`) no treino |
| v1 | regressão logística com 7 atributos (`C=1`, `class_weight="balanced"`) |
| Refinamento (v2) | inclusão de 15 atributos; comparação entre regressão logística, Random Forest e Gradient Boosting; ajuste de hiperparâmetros (grade na logística, busca aleatória no Gradient Boosting) |
| Modelo final | regressão logística v2 com `C=0,01` e sem peso de classes, escolhida pelo maior PR-AUC na validação cruzada |

## Resultados (teste: julho a setembro de 2026)

**Classificação** (limiar 0,16, escolhido pelo maior F1 na validação)

| Métrica | v1 | v2 |
|---|---|---|
| Acurácia | 0,814 | 0,829 |
| Precisão | 0,264 | 0,300 |
| Revocação | 0,518 | 0,579 |
| F1 | 0,350 | 0,395 |
| ROC-AUC | 0,785 | 0,825 |
| PR-AUC | 0,274 | 0,358 |

**Ranking por vaga** (vagas com pelo menos um match bem-sucedido)

| Métrica | Regra atual (nota) | v1 | v2 |
|---|---|---|---|
| HitRate@1 | 41,9% | 45,0% | **54,7%** |
| HitRate@3 | 80,0% | 81,1% | **87,2%** |
| MRR | 0,625 | 0,649 | **0,719** |
| NDCG@5 | 0,695 | 0,716 | **0,775** |

**Ação por faixa:** vagas do teste com match bem-sucedido foram 51,6% na faixa de risco, 66,6% na de candidatos medianos e 79,0% na de boa opção. As faixas separam bem as vagas que precisam de atenção.

**Overfitting:** treino, validação e teste ficam muito próximos na v2 (ROC-AUC 0,812 / 0,810 / 0,825). A v1 mostra sinais de underfitting. O Gradient Boosting teve PR-AUC de treino bem acima do de validação e não superou o modelo linear.

## Como executar

**Google Colab (recomendado):** abrir o link acima e executar todas as células (Ambiente de execução > Executar tudo). O notebook clona o repositório sozinho. Leva cerca de 5 minutos.

**Localmente** (Python 3.10 ou superior):

```bash
git clone https://github.com/Jonathanpossa/umfreela-recomendacao-freelancers.git
cd umfreela-recomendacao-freelancers
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest               # testes automatizados
jupyter notebook recomendacao_freelancers.ipynb
```

Para gerar os dados de novo: `python src/gerar_dados.py`. A semente fixa garante o mesmo resultado.

## Estrutura

```
.
├── recomendacao_freelancers.ipynb   # relatório técnico (entrega principal)
├── src/
│   ├── gerar_dados.py               # simulação dos dados da plataforma
│   ├── preparo.py                   # diagnóstico, limpeza e engenharia de atributos
│   ├── modelos.py                   # pipelines, ajuste de hiperparâmetros e métricas
│   └── recomendador.py              # convites, ranking dos candidatos e ação por vaga
├── data/                            # dataset (CSV)
├── models/                          # modelo final (joblib) e métricas (JSON)
├── reports/
│   ├── recomendacao_freelancers.html  # notebook executado
│   └── figuras/                     # gráficos usados no relatório
├── tests/test_projeto.py            # testes: limpeza, vazamento, métricas e recomendação
└── requirements.txt
```

## Limitações e evolução

- **Dados simulados:** as relações seguem a lógica do negócio, mas os números não são da operação real. O próximo passo é treinar com o histórico real anonimizado.
- **Viés de seleção:** o modelo aprende só com quem se candidatou; ao convidar quem não se candidatou, a probabilidade pode estar superestimada.
- **Um contratado por vaga:** bons candidatos não escolhidos recebem rótulo 0, o que limita a precisão máxima.
- **Evolução:** teste A/B contra a regra atual, modelo de ranking dedicado (Learning to Rank), limite de convites por freelancer para distribuir oportunidades, calibração e retreino mensal.

O arquivo `models/modelo_v2.joblib` foi salvo com scikit-learn 1.9. Em outras versões, basta executar o notebook para treinar o modelo de novo.
