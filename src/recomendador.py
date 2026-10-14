"""Recomendação de freelancers para uma vaga e ação sugerida para o restaurante.

A recomendação acontece em dois momentos:
1. Na publicação: todos os freelancers da mesma cidade que trabalham na função
   (principal ou secundária) são avaliados pelo modelo, e os primeiros do
   ranking recebem convite.
2. Depois das candidaturas: os candidatos são ordenados para o restaurante e a
   maior probabilidade entre eles define a ação recomendada para a vaga.
"""
import numpy as np
import pandas as pd

from preparo import FEATURES_V2, atributos_par

# faixas da maior probabilidade entre os candidatos (quartis da validação no treino)
LIMIARES_PADRAO = (0.15, 0.33)


def elegiveis(vaga, freelancers):
    f = freelancers
    return f[(f.cidade == vaga["cidade"]) &
             ((f.funcao_principal == vaga["funcao"]) | (f.funcao_secundaria == vaga["funcao"]))]


def montar_pares(vaga, freelancers, historico=None):
    """Monta os atributos de todos os pares (vaga, freelancer elegível)."""
    cand = elegiveis(vaga, freelancers)
    base = pd.DataFrame({k: [v] * len(cand) for k, v in vaga.items()})
    base["id_freela"] = cand.id_freela.values
    base["data_turno"] = pd.to_datetime(base.data_turno)
    base["distancia_km"] = np.hypot(cand.x_km.values - vaga["x_km"], cand.y_km.values - vaga["y_km"]).round(2)
    hist = historico or {}
    base["matches_previos_restaurante"] = [hist.get((vaga["id_restaurante"], f), 0) for f in base.id_freela]
    return atributos_par(base, freelancers)


def motivos(linha):
    m = []
    if linha.matches_previos_restaurante > 0:
        m.append(f"já trabalhou {int(linha.matches_previos_restaurante)}x aqui")
    if linha.distancia_km <= 3:
        m.append(f"mora perto ({linha.distancia_km:.1f} km)")
    if not np.isnan(linha.nota_media) and linha.nota_media >= 4.6:
        m.append(f"nota {linha.nota_media:.1f}")
    if not np.isnan(linha.taxa_no_show) and linha.taxa_no_show <= 0.03:
        m.append("quase nunca falta")
    if linha.funcao_principal_igual == 0:
        m.append("função secundária")
    if linha.disponivel_no_turno == 0:
        m.append("disponibilidade não confirmada")
    return ", ".join(m) if m else "-"


def acao(melhor_prob, limiares=LIMIARES_PADRAO):
    baixo, alto = limiares
    if melhor_prob >= alto:
        return ("Boa opção entre os candidatos",
                "Destacar o 1º do ranking para o restaurante contratar.")
    if melhor_prob >= baixo:
        return ("Candidatos medianos",
                "Mostrar o ranking ao restaurante e convidar mais 3 freelancers elegíveis.")
    return ("Risco de não preencher",
            "Nova rodada de convites aos 5 melhores elegíveis que não se candidataram e aviso ao restaurante, "
            "sugerindo rever a diária ou o horário.")


def convidar(vaga, modelo, freelancers, historico=None, k=5):
    """Momento 1: ranking de todos os freelancers elegíveis para enviar convites."""
    pares = montar_pares(vaga, freelancers, historico)
    pares["prob_match"] = modelo.predict_proba(pares[FEATURES_V2])[:, 1]
    top = pares.sort_values("prob_match", ascending=False).head(k).copy()
    top["motivos"] = top.apply(motivos, axis=1)
    top.insert(0, "posicao", range(1, len(top) + 1))
    return top[["posicao", "id_freela", "prob_match", "distancia_km", "nota_media",
                "matches_previos_restaurante", "motivos"]].reset_index(drop=True), len(pares)


def ranquear_candidatos(candidatos, modelo, limiares=LIMIARES_PADRAO):
    """Momento 2: ordena quem se candidatou e define a ação pela maior probabilidade."""
    c = candidatos.copy()
    c["prob_match"] = modelo.predict_proba(c[FEATURES_V2])[:, 1]
    c = c.sort_values("prob_match", ascending=False)
    c["motivos"] = c.apply(motivos, axis=1)
    c.insert(0, "posicao", range(1, len(c) + 1))
    situacao, sugestao = acao(c.prob_match.iloc[0], limiares)
    return c.reset_index(drop=True), {"situacao": situacao, "acao": sugestao,
                                      "melhor_prob": float(c.prob_match.iloc[0])}


def simular_diaria(vaga, modelo, freelancers, historico=None, aumentos=(0, 0.1, 0.2, 0.3)):
    """Como a melhor probabilidade de match muda se o restaurante aumentar a diária."""
    linhas = []
    for a in aumentos:
        v = dict(vaga, valor_diaria=round(vaga["valor_diaria"] * (1 + a) / 5) * 5)
        pares = montar_pares(v, freelancers, historico)
        p = modelo.predict_proba(pares[FEATURES_V2])[:, 1]
        linhas.append({"aumento": f"{int(a * 100)}%", "valor_diaria": v["valor_diaria"],
                       "melhor_prob": p.max(), "media_top5": np.sort(p)[-5:].mean()})
    return pd.DataFrame(linhas)
