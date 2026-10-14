"""Limpeza dos dados e engenharia de atributos.

Cada linha da base final é um par (vaga, freelancer candidato) com atributos
que existem no momento em que a vaga é publicada, e o alvo `match_sucesso`.
"""
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

PASTA_DADOS = Path(__file__).resolve().parents[1] / "data"
CIDADES = ["Santa Maria", "Porto Alegre", "Santiago", "Uruguaiana", "Itaqui", "Quarta Colônia"]
DATA_CORTE_TESTE = "2026-07-01"  # vagas publicadas a partir daqui formam o teste

CATEGORICAS = ["funcao", "cidade", "periodo", "segmento"]
NUMERICAS_V1 = ["distancia_km", "funcao_principal_igual", "nota_media",
                "anos_experiencia", "diferenca_diaria_pct"]
NUMERICAS_V2 = NUMERICAS_V1 + [
    "funcao_secundaria_igual", "disponivel_no_turno", "turno_fim_semana",
    "log_turnos_concluidos", "sem_historico", "taxa_no_show", "taxa_cancelamento",
    "matches_previos_restaurante", "possui_curso", "tempo_resposta_min",
    "perfil_completo_pct", "valor_diaria", "horas_antecedencia",
]
FEATURES_V1 = NUMERICAS_V1 + ["funcao", "cidade"]
FEATURES_V2 = NUMERICAS_V2 + CATEGORICAS


def _sem_acento(t):
    return "".join(c for c in unicodedata.normalize("NFD", str(t)) if unicodedata.category(c) != "Mn")


MAPA_CIDADES = {_sem_acento(c).strip().lower(): c for c in CIDADES}


def padronizar_cidade(serie):
    return serie.map(lambda c: MAPA_CIDADES.get(_sem_acento(c).strip().lower(), np.nan))


def carregar(pasta=PASTA_DADOS):
    pasta = Path(pasta)
    return {n: pd.read_csv(pasta / f"{n}.csv") for n in ["freelancers", "restaurantes", "vagas", "candidaturas"]}


def diagnostico(dados):
    """Faltantes, duplicados e valores fora da faixa esperada em cada tabela."""
    fre, cand = dados["freelancers"], dados["candidaturas"]
    return pd.DataFrame([
        {"tabela": "freelancers", "problema": "cidade fora do padrão",
         "registros": int((~fre.cidade.isin(CIDADES)).sum())},
        {"tabela": "freelancers", "problema": "pretensão de diária inválida (< R$ 50 ou > R$ 1.000)",
         "registros": int((~fre.pretensao_diaria.between(50, 1000)).sum())},
        {"tabela": "freelancers", "problema": "tempo de resposta faltante",
         "registros": int(fre.tempo_resposta_min.isna().sum())},
        {"tabela": "freelancers", "problema": "sem histórico (nota e taxas vazias)",
         "registros": int(fre.nota_media.isna().sum())},
        {"tabela": "candidaturas", "problema": "candidatura duplicada (mesma vaga e freelancer)",
         "registros": int(cand.duplicated(["id_vaga", "id_freela"]).sum())},
        {"tabela": "candidaturas", "problema": "distância faltante",
         "registros": int(cand.distancia_km.isna().sum())},
        {"tabela": "candidaturas", "problema": "distância fora da faixa (> 50 km, erro de GPS)",
         "registros": int((cand.distancia_km > 50).sum())},
    ])


def limpar(dados):
    fre = dados["freelancers"].copy()
    rest = dados["restaurantes"].copy()
    vagas = dados["vagas"].copy()
    cand = dados["candidaturas"].copy()

    fre["cidade"] = padronizar_cidade(fre.cidade)
    fre.loc[~fre.pretensao_diaria.between(50, 1000), "pretensao_diaria"] = np.nan
    # pretensão inválida: substitui pela mediana da mesma função
    fre["pretensao_diaria"] = fre.pretensao_diaria.fillna(
        fre.groupby("funcao_principal").pretensao_diaria.transform("median"))

    cand = cand.sort_values("data_candidatura").drop_duplicates(["id_vaga", "id_freela"], keep="first")
    cand.loc[cand.distancia_km > 50, "distancia_km"] = np.nan
    # distância faltante ou com erro: recalculada pelas coordenadas
    pos = (cand.merge(vagas[["id_vaga", "id_restaurante"]], on="id_vaga")
               .merge(rest[["id_restaurante", "x_km", "y_km"]], on="id_restaurante")
               .merge(fre[["id_freela", "x_km", "y_km"]], on="id_freela", suffixes=("_r", "_f")))
    calc = np.hypot(pos.x_km_r - pos.x_km_f, pos.y_km_r - pos.y_km_f).round(2)
    cand = pos.assign(distancia_km=pos.distancia_km.fillna(calc))[cand.columns]

    vagas["data_publicacao"] = pd.to_datetime(vagas.data_publicacao)
    vagas["data_turno"] = pd.to_datetime(vagas.data_turno)
    return {"freelancers": fre, "restaurantes": rest, "vagas": vagas, "candidaturas": cand.reset_index(drop=True)}


def atributos_par(vagas_rest, fre):
    """Atributos de um par (vaga, freelancer). Usado no treino e na recomendação."""
    df = vagas_rest.merge(fre.drop(columns=["cidade", "x_km", "y_km"]), on="id_freela", how="left")
    df["funcao_principal_igual"] = (df.funcao_principal == df.funcao).astype(int)
    df["funcao_secundaria_igual"] = (df.funcao_secundaria == df.funcao).astype(int)
    df["diferenca_diaria_pct"] = ((df.valor_diaria - df.pretensao_diaria) / df.valor_diaria).round(4)
    df["turno_fim_semana"] = (df.data_turno.dt.dayofweek >= 5).astype(int)
    disp_periodo = np.select([df.periodo == "manha", df.periodo == "tarde"],
                             [df.disp_manha, df.disp_tarde], df.disp_noite)
    df["disponivel_no_turno"] = (disp_periodo * np.where(df.turno_fim_semana == 1, df.disp_fim_semana, 1)).astype(int)
    df["log_turnos_concluidos"] = np.log1p(df.turnos_concluidos)
    df["sem_historico"] = df.nota_media.isna().astype(int)
    return df


def construir_base(dados_limpos):
    fre, rest, vagas, cand = (dados_limpos[k] for k in ["freelancers", "restaurantes", "vagas", "candidaturas"])
    df = (cand.merge(vagas, on="id_vaga")
              .merge(rest[["id_restaurante", "cidade", "segmento"]], on="id_restaurante"))
    # histórico com o restaurante: só matches de vagas publicadas ANTES desta (sem vazamento)
    df = df.sort_values(["data_publicacao", "id_vaga"]).reset_index(drop=True)
    df["matches_previos_restaurante"] = (df.groupby(["id_restaurante", "id_freela"]).match_sucesso
                                           .transform(lambda s: s.shift(fill_value=0).cumsum()))
    df = atributos_par(df, fre)
    df["conjunto"] = np.where(df.data_publicacao < DATA_CORTE_TESTE, "treino", "teste")
    colunas = (["id_vaga", "id_freela", "id_restaurante", "data_publicacao", "conjunto"]
               + NUMERICAS_V2 + CATEGORICAS + ["contratado", "match_sucesso"])
    return df[colunas]
