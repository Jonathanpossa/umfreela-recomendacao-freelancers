"""Gerador do conjunto de dados sintético da UmFreela.

Simula 12 meses de operação da plataforma em seis cidades do RS: freelancers,
restaurantes, vagas publicadas e candidaturas. Para cada candidatura fica
registrado se ela terminou em um "match bem-sucedido" (o freelancer foi
contratado, compareceu ao turno e recebeu nota 4 ou 5), que é o alvo do modelo.

Os dados são sintéticos para não expor dados pessoais (LGPD), mas seguem as
regras do negócio: o restaurante escolhe entre os candidatos, freelancers mais
próximos, mais bem avaliados e mais confiáveis têm mais chance de dar certo, e
parte dos registros chega com problemas de qualidade (faltantes, duplicados,
outliers e textos fora do padrão), tratados na análise exploratória.

Uso:  python src/gerar_dados.py            (grava os CSVs em data/)
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEMENTE = 42
PASTA_DADOS = Path(__file__).resolve().parents[1] / "data"

CIDADES = {  # peso na plataforma, raio da área atendida (km)
    "Santa Maria": (0.32, 7.0),
    "Porto Alegre": (0.30, 12.0),
    "Santiago": (0.10, 4.0),
    "Uruguaiana": (0.11, 5.0),
    "Itaqui": (0.07, 3.5),
    "Quarta Colônia": (0.10, 9.0),
}
FUNCOES = {  # peso nas vagas, diária base (R$)
    "Garçom": (0.30, 140),
    "Cozinheiro": (0.13, 190),
    "Auxiliar de Cozinha": (0.15, 130),
    "Bartender": (0.09, 170),
    "Recepcionista": (0.07, 135),
    "Chapeiro": (0.08, 150),
    "Copeiro": (0.08, 120),
    "Auxiliar de Limpeza": (0.10, 115),
}
FUNCOES_PROXIMAS = {  # função secundária comum para quem tem a principal
    "Garçom": ["Bartender", "Copeiro", "Recepcionista"],
    "Cozinheiro": ["Auxiliar de Cozinha", "Chapeiro"],
    "Auxiliar de Cozinha": ["Cozinheiro", "Copeiro", "Auxiliar de Limpeza"],
    "Bartender": ["Garçom", "Copeiro"],
    "Recepcionista": ["Garçom"],
    "Chapeiro": ["Cozinheiro", "Auxiliar de Cozinha"],
    "Copeiro": ["Auxiliar de Cozinha", "Garçom", "Auxiliar de Limpeza"],
    "Auxiliar de Limpeza": ["Copeiro", "Auxiliar de Cozinha"],
}
SEGMENTOS = ["Restaurante", "Bar", "Pizzaria", "Lancheria", "Hotel", "Buffet", "Churrascaria"]
PERIODOS = {"manha": 0.25, "tarde": 0.30, "noite": 0.45}
INICIO, FIM = pd.Timestamp("2025-10-01"), pd.Timestamp("2026-09-30")

N_FREELAS, N_RESTAURANTES, N_VAGAS = 1800, 320, 7000


def _ponto(rng, raio, n):
    """Coordenadas (km) de n pontos dentro da área da cidade, mais densos no centro."""
    r = raio * np.sqrt(rng.uniform(0, 1, n)) * rng.uniform(0.4, 1.0, n)
    a = rng.uniform(0, 2 * np.pi, n)
    return r * np.cos(a), r * np.sin(a)


def gerar_freelancers(rng):
    cid = rng.choice(list(CIDADES), N_FREELAS, p=[v[0] for v in CIDADES.values()])
    fun = rng.choice(list(FUNCOES), N_FREELAS, p=[v[0] for v in FUNCOES.values()])
    x, y = np.zeros(N_FREELAS), np.zeros(N_FREELAS)
    for c, (_, raio) in CIDADES.items():
        m = cid == c
        x[m], y[m] = _ponto(rng, raio, m.sum())
    sec = [rng.choice(FUNCOES_PROXIMAS[f]) if rng.random() < 0.6 else None for f in fun]

    qualidade = rng.normal(0, 1, N_FREELAS)       # latente: qualidade do trabalho
    confiab = rng.normal(0, 1, N_FREELAS)         # latente: costuma aparecer?
    engaj = rng.normal(0, 1, N_FREELAS)           # latente: responde rápido, usa o app
    exp = np.round(rng.gamma(2.0, 1.6, N_FREELAS), 1)
    turnos = rng.poisson(np.exp(1.2 + 0.35 * np.log1p(exp) + 0.4 * engaj).clip(0, 80))
    nota = np.clip(4.25 + 0.38 * qualidade + rng.normal(0, 0.15, N_FREELAS), 1, 5).round(2)
    nota = np.where(turnos == 0, np.nan, nota)
    no_show = np.clip(rng.beta(1.2, 18, N_FREELAS) * np.exp(-0.6 * confiab), 0, 0.6).round(3)
    cancel = np.clip(rng.beta(1.3, 14, N_FREELAS) * np.exp(-0.4 * confiab), 0, 0.6).round(3)
    no_show = np.where(turnos == 0, np.nan, no_show)
    cancel = np.where(turnos == 0, np.nan, cancel)
    base = np.array([FUNCOES[f][1] for f in fun])
    pretensao = np.round(base * (0.9 + 0.025 * exp + rng.normal(0, 0.07, N_FREELAS)) / 5) * 5
    resposta = np.round(np.exp(3.2 - 0.6 * engaj + rng.normal(0, 0.5, N_FREELAS)), 0)
    perfil = np.clip(np.round(70 + 12 * engaj + rng.normal(0, 10, N_FREELAS)), 20, 100)
    cadastro = INICIO - pd.to_timedelta(rng.integers(0, 900, N_FREELAS), unit="D")

    df = pd.DataFrame({
        "id_freela": [f"F{i:05d}" for i in range(1, N_FREELAS + 1)],
        "cidade": cid, "x_km": x.round(3), "y_km": y.round(3),
        "funcao_principal": fun, "funcao_secundaria": sec,
        "anos_experiencia": exp, "turnos_concluidos": turnos, "nota_media": nota,
        "taxa_no_show": no_show, "taxa_cancelamento": cancel,
        "pretensao_diaria": pretensao,
        "disp_manha": (rng.random(N_FREELAS) < 0.55).astype(int),
        "disp_tarde": (rng.random(N_FREELAS) < 0.70).astype(int),
        "disp_noite": (rng.random(N_FREELAS) < 0.75).astype(int),
        "disp_fim_semana": (rng.random(N_FREELAS) < 0.80).astype(int),
        "possui_curso": (rng.random(N_FREELAS) < 0.25 + 0.1 * (qualidade > 0)).astype(int),
        "tempo_resposta_min": resposta, "perfil_completo_pct": perfil,
        "data_cadastro": cadastro.strftime("%Y-%m-%d"),
    })
    latentes = pd.DataFrame({"id_freela": df.id_freela, "q": qualidade, "r": confiab, "e": engaj})
    return df, latentes


def gerar_restaurantes(rng):
    cid = rng.choice(list(CIDADES), N_RESTAURANTES, p=[v[0] for v in CIDADES.values()])
    x, y = np.zeros(N_RESTAURANTES), np.zeros(N_RESTAURANTES)
    for c, (_, raio) in CIDADES.items():
        m = cid == c
        x[m], y[m] = _ponto(rng, raio * 0.7, m.sum())
    return pd.DataFrame({
        "id_restaurante": [f"R{i:04d}" for i in range(1, N_RESTAURANTES + 1)],
        "cidade": cid, "x_km": x.round(3), "y_km": y.round(3),
        "segmento": rng.choice(SEGMENTOS, N_RESTAURANTES,
                               p=[0.3, 0.15, 0.12, 0.13, 0.1, 0.1, 0.1]),
    })


def gerar_vagas(rng, rest):
    atividade = rng.gamma(1.5, 1.0, len(rest))
    idx = rng.choice(len(rest), N_VAGAS, p=atividade / atividade.sum())
    dias = (FIM - INICIO).days + 1
    # mais vagas nos meses recentes (crescimento) e no fim de semana
    pub = INICIO + pd.to_timedelta(np.sort((rng.beta(1.3, 1.0, N_VAGAS) * dias).astype(int)), unit="D")
    pub = pub + pd.to_timedelta(rng.integers(8 * 60, 22 * 60, N_VAGAS), unit="m")
    periodo = rng.choice(list(PERIODOS), N_VAGAS, p=list(PERIODOS.values()))
    # o turno começa em um horário coerente com o período, de 3 h a ~5 dias após a publicação
    inicio_periodo = {"manha": (7, 11), "tarde": (12, 17), "noite": (18, 23)}
    hora = np.array([rng.integers(*inicio_periodo[p_]) for p_ in periodo])
    alvo = pub + pd.to_timedelta(rng.gamma(2.0, 30, N_VAGAS) + 3, unit="h")
    turno = alvo.normalize() + pd.to_timedelta(hora, unit="h")
    turno = turno.where(turno > pub + pd.Timedelta(hours=3), turno + pd.Timedelta(days=1))
    antecedencia_h = np.round((turno - pub).total_seconds() / 3600, 1)
    fun = rng.choice(list(FUNCOES), N_VAGAS, p=[v[0] for v in FUNCOES.values()])
    base = np.array([FUNCOES[f][1] for f in fun])
    urgencia = np.where(antecedencia_h < 24, 1.10, 1.0)
    valor = np.round(base * urgencia * rng.normal(1.0, 0.09, N_VAGAS) / 5) * 5
    return pd.DataFrame({
        "id_vaga": [f"V{i:05d}" for i in range(1, N_VAGAS + 1)],
        "id_restaurante": rest.id_restaurante.values[idx],
        "funcao": fun,
        "data_publicacao": pub.strftime("%Y-%m-%d %H:%M"),
        "data_turno": turno.strftime("%Y-%m-%d %H:%M"),
        "periodo": periodo,
        "valor_diaria": valor,
        "horas_antecedencia": antecedencia_h,
    })


def _softmax(u):
    e = np.exp(u - u.max())
    return e / e.sum()


def gerar_candidaturas(rng, vagas, rest, fre, lat):
    rest_i = rest.set_index("id_restaurante")
    fre = fre.merge(lat, on="id_freela")
    por_cidade = {c: fre[fre.cidade == c].reset_index(drop=True) for c in CIDADES}
    historico = {}  # (restaurante, freela) -> matches bem-sucedidos anteriores
    linhas = []
    for v in vagas.itertuples(index=False):
        r = rest_i.loc[v.id_restaurante]
        pool = por_cidade[r.cidade]
        compat = (pool.funcao_principal == v.funcao) | (pool.funcao_secundaria == v.funcao)
        outros = pool[~compat].sample(min(3, (~compat).sum()), random_state=int(rng.integers(1e9)))
        pool = pd.concat([pool[compat], outros])
        dist = np.hypot(pool.x_km - r.x_km, pool.y_km - r.y_km).values
        disp = pool[f"disp_{v.periodo}"].values
        fds = pd.Timestamp(v.data_turno).dayofweek >= 5
        if fds:
            disp = disp * pool.disp_fim_semana.values
        principal = (pool.funcao_principal == v.funcao).values
        dif = (v.valor_diaria - pool.pretensao_diaria.values) / v.valor_diaria
        # propensão a se candidatar
        prop = (1.2 * principal + 0.6 * (~principal & (pool.funcao_secundaria == v.funcao).values)
                - 0.18 * dist + 2.5 * dif + 0.6 * pool.e.values + 1.0 * disp - 1.6)
        p = 1 / (1 + np.exp(-prop))
        n = int(np.clip(rng.poisson(7), 2, 14))
        n = min(n, len(pool))
        esc = rng.choice(len(pool), n, replace=False, p=p / p.sum())
        cand = pool.iloc[esc].reset_index(drop=True)
        d, dp, pr = dist[esc], disp[esc], principal[esc]
        dif_c = dif[esc]
        hist = np.array([historico.get((v.id_restaurante, f), 0) for f in cand.id_freela])
        nota_obs = cand.nota_media.fillna(3.8).values
        # escolha do restaurante entre os candidatos
        u = (1.6 * (nota_obs - 4.2) + 1.1 * pr + 0.9 * np.minimum(hist, 3) - 0.10 * d
             + 0.08 * np.log1p(cand.turnos_concluidos.values) * 3 - 0.6 * (1 - dp)
             + 0.5 * cand.possui_curso.values + 0.01 * (cand.perfil_completo_pct.values - 70)
             - 1.2 * cand.taxa_no_show.fillna(0.08).values * 5 + rng.gumbel(0, 0.6, n))
        contratado = np.zeros(n, dtype=int)
        if rng.random() > 0.08 + 0.25 * (u.max() < -1.5):  # às vezes ninguém é contratado
            contratado[rng.choice(n, p=_softmax(u * 1.4))] = 1
        # o turno dá certo? (comparece, não cancela, recebe nota 4 ou 5)
        p_aparece = 1 / (1 + np.exp(-(2.6 + 0.9 * cand.r.values - 0.07 * d + 0.6 * dp + 0.3 * dif_c)))
        nota_turno = 3.9 + 0.65 * cand.q.values + 0.35 * pr + 0.15 * np.minimum(hist, 3) + rng.normal(0, 0.55, n)
        sucesso = contratado * (rng.random(n) < p_aparece) * (nota_turno >= 3.9)
        for f, s in zip(cand.id_freela, sucesso):
            if s:
                historico[(v.id_restaurante, f)] = historico.get((v.id_restaurante, f), 0) + 1
        pub = pd.Timestamp(v.data_publicacao)
        atraso = pd.to_timedelta(np.round(cand.tempo_resposta_min.values * rng.uniform(0.5, 3, n)), unit="m")
        for k in range(n):
            linhas.append((v.id_vaga, cand.id_freela[k], (pub + atraso[k]).strftime("%Y-%m-%d %H:%M"),
                           round(float(d[k]), 2), int(contratado[k]), int(sucesso[k])))
    return pd.DataFrame(linhas, columns=["id_vaga", "id_freela", "data_candidatura", "distancia_km",
                                         "contratado", "match_sucesso"])


def sujar(rng, fre, cand):
    """Insere problemas de qualidade parecidos com os de produção."""
    fre, cand = fre.copy(), cand.copy()
    m = rng.random(len(fre)) < 0.05  # cidade digitada fora do padrão
    fre.loc[m, "cidade"] = [rng.choice([c.upper(), c.lower(), f" {c} "]) for c in fre.loc[m, "cidade"]]
    m = rng.random(len(fre)) < 0.006  # pretensão inválida
    fre.loc[m, "pretensao_diaria"] = rng.choice([0, 9999], m.sum())
    m = rng.random(len(fre)) < 0.04  # tempo de resposta não registrado
    fre.loc[m, "tempo_resposta_min"] = np.nan
    m = rng.random(len(cand)) < 0.003  # erro de GPS
    cand.loc[m, "distancia_km"] = 999.0
    m = rng.random(len(cand)) < 0.01  # distância não calculada
    cand.loc[m, "distancia_km"] = np.nan
    dup = cand.sample(frac=0.015, random_state=SEMENTE)  # candidatura reenviada pelo app
    cand = pd.concat([cand, dup]).sort_values(["id_vaga", "data_candidatura"]).reset_index(drop=True)
    return fre, cand


def gerar(pasta=PASTA_DADOS):
    rng = np.random.default_rng(SEMENTE)
    fre, lat = gerar_freelancers(rng)
    rest = gerar_restaurantes(rng)
    vagas = gerar_vagas(rng, rest)
    cand = gerar_candidaturas(rng, vagas, rest, fre, lat)
    fre, cand = sujar(rng, fre, cand)
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    for nome, df in [("freelancers", fre), ("restaurantes", rest), ("vagas", vagas), ("candidaturas", cand)]:
        df.to_csv(pasta / f"{nome}.csv", index=False)
    return {"freelancers": len(fre), "restaurantes": len(rest), "vagas": len(vagas),
            "candidaturas": len(cand), "taxa_match": round(cand.match_sucesso.mean(), 4)}


if __name__ == "__main__":
    print(gerar())
