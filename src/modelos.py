"""Pipelines, ajuste de hiperparâmetros e métricas de avaliação."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEMENTE = 42


def preprocessador(numericas, categoricas, escalar=True):
    num = [("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        num.append(("padronizar", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
    ])


def pipeline_v1(numericas, categoricas):
    """Versão base: regressão logística com poucos atributos."""
    return Pipeline([
        ("prep", preprocessador(numericas, categoricas)),
        ("modelo", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
    ])


def pipeline_floresta(numericas, categoricas):
    return Pipeline([
        ("prep", preprocessador(numericas, categoricas, escalar=False)),
        ("modelo", RandomForestClassifier(n_estimators=300, min_samples_leaf=20, class_weight="balanced",
                                          n_jobs=-1, random_state=SEMENTE)),
    ])


def pipeline_boosting(numericas, categoricas):
    return Pipeline([
        ("prep", preprocessador(numericas, categoricas, escalar=False)),
        ("modelo", HistGradientBoostingClassifier(random_state=SEMENTE)),
    ])


ESPACO_BOOSTING = {
    "modelo__learning_rate": [0.03, 0.05, 0.08, 0.12],
    "modelo__max_iter": [150, 250, 400],
    "modelo__max_leaf_nodes": [15, 31, 63],
    "modelo__min_samples_leaf": [20, 50, 100],
    "modelo__l2_regularization": [0.0, 0.5, 1.0],
}


def ajustar_boosting(pipe, X, y, grupos, n_iter=20):
    """Busca aleatória com validação cruzada agrupada por vaga (a mesma vaga nunca
    aparece no treino e na validação ao mesmo tempo)."""
    busca = RandomizedSearchCV(pipe, ESPACO_BOOSTING, n_iter=n_iter, scoring="average_precision",
                               cv=GroupKFold(n_splits=5), random_state=SEMENTE, n_jobs=-1,
                               return_train_score=True)
    busca.fit(X, y, groups=grupos)
    return busca


ESPACO_LOGISTICA = {
    "modelo__C": [0.01, 0.1, 1.0, 10.0],
    "modelo__class_weight": [None, "balanced"],
}


def ajustar_logistica(pipe, X, y, grupos):
    """Busca em grade para a regressão logística (força da regularização e peso das classes)."""
    busca = GridSearchCV(pipe, ESPACO_LOGISTICA, scoring="average_precision", cv=GroupKFold(n_splits=5),
                         n_jobs=-1, return_train_score=True)
    busca.fit(X, y, groups=grupos)
    return busca


def prever_oof(pipe, X, y, grupos):
    """Probabilidades fora da amostra no treino (validação cruzada por vaga)."""
    return cross_val_predict(pipe, X, y, groups=grupos, cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1]


def melhor_limiar(y, p):
    """Limiar que maximiza o F1 nas previsões de validação."""
    limiares = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y, p >= t, zero_division=0) for t in limiares]
    return float(round(limiares[int(np.argmax(f1s))], 2))


def metricas_classificacao(y, p, limiar):
    pred = (p >= limiar).astype(int)
    return {
        "acuracia": accuracy_score(y, pred),
        "precisao": precision_score(y, pred, zero_division=0),
        "revocacao": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
    }


def matriz_confusao(y, p, limiar):
    return pd.DataFrame(confusion_matrix(y, (p >= limiar).astype(int)),
                        index=["real: não", "real: sim"], columns=["previsto: não", "previsto: sim"])


def metricas_ranking(df, coluna_score, ks=(1, 3, 5)):
    """Qualidade do ranking por vaga, considerando só vagas com pelo menos um match.

    HitRate@k: % de vagas em que o freelancer que deu certo estava entre os k primeiros.
    MRR: média de 1/posição do primeiro freelancer que deu certo.
    NDCG@5: ganho acumulado descontado, normalizado pelo ranking ideal.
    """
    res = {f"hit_rate@{k}": [] for k in ks}
    res["mrr"], res["ndcg@5"] = [], []
    for _, g in df.groupby("id_vaga"):
        if g.match_sucesso.sum() == 0:
            continue
        rel = g.sort_values(coluna_score, ascending=False).match_sucesso.to_numpy()
        pos = int(np.argmax(rel)) + 1
        for k in ks:
            res[f"hit_rate@{k}"].append(pos <= k)
        res["mrr"].append(1 / pos)
        desc = 1 / np.log2(np.arange(2, 7))
        dcg = float((rel[:5] * desc[:len(rel[:5])]).sum())
        ideal = np.sort(rel)[::-1][:5]
        res["ndcg@5"].append(dcg / float((ideal * desc[:len(ideal)]).sum()))
    return {k: float(np.mean(v)) for k, v in res.items()}


def score_heuristico(df):
    """Regra usada antes do modelo: ordenar por nota e, no empate, pela distância."""
    return df.nota_media.fillna(0) * 10 - df.distancia_km.fillna(df.distancia_km.median()) * 0.01
