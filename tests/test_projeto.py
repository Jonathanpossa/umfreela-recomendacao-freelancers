"""Testes automatizados: limpeza, ausência de vazamento, métricas e recomendação."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import modelos as M  # noqa: E402
import preparo as P  # noqa: E402
import recomendador as R  # noqa: E402


@pytest.fixture(scope="module")
def dados():
    brutos = P.carregar()
    limpos = P.limpar(brutos)
    return brutos, limpos, P.construir_base(limpos)


def test_padroniza_cidade():
    s = pd.Series(["SANTA MARIA", " porto alegre ", "quarta colonia", "Cidade X"])
    assert list(P.padronizar_cidade(s)[:3]) == ["Santa Maria", "Porto Alegre", "Quarta Colônia"]
    assert pd.isna(P.padronizar_cidade(s)[3])


def test_limpeza_remove_problemas(dados):
    _, limpos, _ = dados
    fre, cand = limpos["freelancers"], limpos["candidaturas"]
    assert fre.cidade.isin(P.CIDADES).all()
    assert fre.pretensao_diaria.between(50, 1000).all()
    assert not cand.duplicated(["id_vaga", "id_freela"]).any()
    assert cand.distancia_km.notna().all() and (cand.distancia_km <= 50).all()


def test_historico_sem_vazamento(dados):
    _, _, base = dados
    # o primeiro encontro de cada par nunca pode ter histórico
    primeiro = base.sort_values("data_publicacao").groupby(["id_restaurante", "id_freela"]).head(1)
    assert (primeiro.matches_previos_restaurante == 0).all()


def test_divisao_temporal(dados):
    _, _, base = dados
    assert base[base.conjunto == "treino"].data_publicacao.max() < base[base.conjunto == "teste"].data_publicacao.min()
    assert not set(base[base.conjunto == "treino"].id_vaga) & set(base[base.conjunto == "teste"].id_vaga)


def test_metricas_ranking():
    df = pd.DataFrame({"id_vaga": ["a"] * 3 + ["b"] * 3,
                       "match_sucesso": [0, 1, 0, 1, 0, 0],
                       "s": [0.9, 0.5, 0.1, 0.8, 0.2, 0.1]})
    m = M.metricas_ranking(df, "s")
    assert m["hit_rate@1"] == 0.5 and m["hit_rate@3"] == 1.0
    assert m["mrr"] == pytest.approx((1 / 2 + 1) / 2)


def test_recomendacao(dados):
    brutos, limpos, base = dados
    tr = base[base.conjunto == "treino"].sample(5000, random_state=1)
    modelo = M.pipeline_v1(P.NUMERICAS_V2, P.CATEGORICAS).fit(tr[P.FEATURES_V2], tr.match_sucesso)
    rest = limpos["restaurantes"].iloc[0]
    vaga = {"id_vaga": "NOVA", "id_restaurante": rest.id_restaurante, "cidade": rest.cidade,
            "segmento": rest.segmento, "x_km": rest.x_km, "y_km": rest.y_km, "funcao": "Garçom",
            "periodo": "noite", "data_turno": "2026-10-17 19:00", "valor_diaria": 140.0,
            "horas_antecedencia": 48.0}
    tabela, n = R.convidar(vaga, modelo, limpos["freelancers"], k=5)
    assert len(tabela) == 5 and n >= 5
    assert tabela.prob_match.is_monotonic_decreasing
    eleg = R.elegiveis(vaga, limpos["freelancers"])
    assert set(tabela.id_freela) <= set(eleg.id_freela)
    cands = base[base.id_vaga == base.id_vaga.iloc[-1]]
    ranking, resumo = R.ranquear_candidatos(cands, modelo)
    assert len(ranking) == len(cands) and ranking.prob_match.is_monotonic_decreasing
    assert resumo["situacao"] in {"Boa opção entre os candidatos", "Candidatos medianos", "Risco de não preencher"}
