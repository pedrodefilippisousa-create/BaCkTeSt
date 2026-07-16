"""
Diversificação Setorial da B3 — Backtest Unificado
====================================================
Junta o MOTOR DE CÁLCULO do Backtest-Rafael (backtest walk-forward
out-of-sample entre setores sintéticos da B3, com 4 estratégias de alocação)
com as FUNÇÕES DE DASHBOARD do Backtest-Pedro (navegação, resumo executivo,
textos explicativos por gráfico, simulador interativo, glossário e
botão "voltar ao topo").

Estratégias comparadas (out-of-sample):
    • Markowitz — Máximo Sharpe
    • Markowitz — Mínima Variância
    • Equal-Weight (1/N)
    • Risk Parity
    • Ibovespa (benchmark)

Cada setor é um índice sintético = cesta de ações líderes ponderada pelo
market cap atual. A cada N meses os pesos de cada estratégia são reestimados
numa janela de LOOKBACK_YEARS e aplicados no período seguinte (out-of-sample).

Saída: janela NATIVA (via pywebview) quando disponível; caso contrário
gera/abre um dashboard .html interativo (Plotly, tema escuro). O .html é
sempre salvo em disco para compartilhamento. Dados 100% via Yahoo Finance.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from datetime import datetime, timedelta
import os
import json

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURAÇÃO — edite aqui antes de rodar
# ─────────────────────────────────────────────────────────────────────────────
SECTORS = {
    "Financeiro":                    ["ITUB4", "BBDC4", "BBAS3", "B3SA3", "SANB11", "BPAC11", "ITSA4", "PSSA3", "BBSE3", "CXSE3"],
    "Materiais/Mineração":           ["VALE3", "GGBR4", "CSNA3", "USIM5", "BRAP4", "GOAU4", "CBAV3"],
    "Papel e Celulose":              ["SUZB3", "KLBN11", "RANI3"],
    "Petróleo e Gás":                ["PETR4", "PETR3", "PRIO3", "UGPA3", "VBBR3", "RECV3", "CSAN3"],
    "Energia Elétrica":              ["EQTL3", "ENGI11", "CMIG4", "CPLE3", "TAEE11", "EGIE3", "CPFE3", "NEOE3", "AURE3"],
    "Saneamento":                    ["SBSP3", "SAPR11", "CSMG3"],
    "Consumo Cíclico/Varejo":        ["LREN3", "MGLU3", "PCAR3", "ASAI3", "VIVA3", "SBFG3", "SMFT3", "CVCB3", "GMAT3", "BHIA3"],
    "Consumo Não-Cíclico/Alimentos": ["ABEV3", "BEEF3", "JBSS32", "MBRF3"],
    "Saúde":                         ["RDOR3", "HAPV3", "FLRY3", "HYPE3", "RADL3"],
    "Bens Industriais":              ["WEGE3", "RAIL3", "MOTV3", "POMO4", "TUPY3", "KEPL3", "RAPT4", "VAMO3"],
    "Imobiliário/Construção":        ["MRVE3", "CYRE3", "EZTC3", "MULT3", "JHSF3", "ALOS3", "DXCO3"],
    "Tecnologia/Telecom":            ["VIVT3", "TIMS3", "TOTS3", "LWSA3"],
    "Agro":                          ["SLCE3", "SMTO3", "AGRO3", "TTEN3"],
}
BENCHMARK        = "^BVSP"     # Ibovespa
BENCHMARK_NAME   = "Ibovespa"
RISK_FREE_RATE   = 0.1425      # Selic anual aprox. (CDI/livre de risco)
YEARS_HISTORY    = 10          # histórico total baixado
LOOKBACK_YEARS   = 3           # janela de estimação no walk-forward
REBALANCE_MONTHS = 1           # rebalanceia a cada N meses
MAX_WEIGHT       = 0.30        # teto por setor na otimização (evita cantos)
N_PORTFOLIOS     = 8_000       # nuvem Monte Carlo p/ fronteira eficiente

USE_NATIVE_WINDOW = False      # True: tenta janela nativa (pywebview); False: abre no navegador
OUTPUT_DIR        = "."
OUTPUT_NAME       = "setores_b3_dashboard"
# ─────────────────────────────────────────────────────────────────────────────

TRADING_DAYS = 252

# Estratégias e suas cores/labels (ordem de exibição)
STRATEGIES = ["Max Sharpe", "Min Variância", "Equal-Weight", "Risk Parity"]

COLORS = {
    "max_sharpe":  "#FF6F00",
    "min_vol":     "#26A69A",
    "equal":       "#2196F3",
    "risk_parity": "#AB47BC",
    "benchmark":   "#EF5350",
    "positive":    "#26A69A",
    "negative":    "#EF5350",
    "grid":        "#2D3748",
}

STRAT_COLORS = {
    "Max Sharpe":    COLORS["max_sharpe"],
    "Min Variância": COLORS["min_vol"],
    "Equal-Weight":  COLORS["equal"],
    "Risk Parity":   COLORS["risk_parity"],
    BENCHMARK_NAME:  COLORS["benchmark"],
}

THEME = dict(
    paper_bgcolor="#0D1117",
    plot_bgcolor="#161B22",
    font_color="#E6EDF3",
    font_family="Inter, Arial, sans-serif",
    font_size=13,
    title_font_size=18,
)

PERIODS = {
    "6M":  timedelta(days=183),
    "1A":  timedelta(days=365),
    "3A":  timedelta(days=1095),
    "5A":  timedelta(days=1825),
    "10A": timedelta(days=3650),
}
PERIOD_LABELS = {"6M": "6 meses", "1A": "1 ano", "3A": "3 anos",
                 "5A": "5 anos", "10A": "10 anos"}


# ═════════════════════════════════════════════════════════════════════════════
#  1. DOWNLOAD E CONSTRUÇÃO DOS ÍNDICES SETORIAIS   (motor Rafael)
# ═════════════════════════════════════════════════════════════════════════════
def _sa(ticker):
    """Sufixo .SA do Yahoo para ações da B3 (índices, ex.: ^BVSP, ficam como estão)."""
    return ticker if ticker.startswith("^") else f"{ticker}.SA"


def download_prices(sectors, benchmark, years):
    all_stocks = sorted({t for lst in sectors.values() for t in lst})
    yahoo = [_sa(t) for t in all_stocks] + [benchmark]
    start = datetime.today() - timedelta(days=int(years * 365.25) + 10)
    data = yf.download(yahoo, start=start, progress=False, auto_adjust=True)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    # descarta linhas e colunas totalmente vazias (tickers que falharam no download)
    return close.dropna(how="all").dropna(how="all", axis=1)


def fetch_market_caps(stocks):
    """Market cap atual de cada ação (peso por 'market share' dentro do setor).
    yfinance.info é instável; usa fast_info -> info -> fallback equal-weight."""
    caps = {}
    for t in stocks:
        cap = None
        tk = yf.Ticker(_sa(t))
        try:
            cap = tk.fast_info.get("market_cap")
        except Exception:
            cap = None
        if not cap:
            try:
                cap = tk.info.get("marketCap")
            except Exception:
                cap = None
        caps[t] = float(cap) if cap else np.nan
    return caps


def build_sector_indices(prices, sectors, caps):
    """Para cada setor: retorno diário = média dos retornos das ações ponderada por
    market cap (renormalizada pelos tickers efetivamente disponíveis). Acumula em
    índice base 100. Retorna (sector_prices_df, sector_returns_df, composition_dict)."""
    rets = prices.pct_change()
    sector_returns = {}
    composition = {}
    for sector, stocks in sectors.items():
        cols = [_sa(t) for t in stocks if _sa(t) in prices.columns]
        if not cols:
            print(f"  ! Setor '{sector}' sem dados — ignorado.")
            continue
        plain = [c[:-3] for c in cols]
        w = np.array([caps.get(t, np.nan) for t in plain], dtype=float)
        if np.isnan(w).all() or np.nansum(w) == 0:
            w = np.ones(len(cols))  # fallback equal-weight
            print(f"  ! Setor '{sector}': sem market cap — usando equal-weight.")
        else:
            w = np.where(np.isnan(w), np.nanmin(w), w)  # ações sem cap herdam o menor
        w = w / w.sum()
        sec_ret = (rets[cols] * w).sum(axis=1, min_count=1)
        sector_returns[sector] = sec_ret
        composition[sector] = dict(zip(plain, w))
    df = pd.DataFrame(sector_returns).dropna(how="all")
    sector_prices = (1 + df.fillna(0)).cumprod() * 100
    return sector_prices, df, composition


# ═════════════════════════════════════════════════════════════════════════════
#  2. ESTRATÉGIAS DE ALOCAÇÃO   (motor Rafael)
# ═════════════════════════════════════════════════════════════════════════════
def port_stats(w, mu, cov):
    r = float(np.dot(w, mu)) * TRADING_DAYS
    v = float(np.sqrt(w @ cov @ w)) * np.sqrt(TRADING_DAYS)
    s = (r - RISK_FREE_RATE) / v if v else np.nan
    return r, v, s


def _bounds_constraints(n):
    bounds = [(0, MAX_WEIGHT)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    return bounds, constraints


def w_max_sharpe(mu, cov):
    n = len(mu)
    bounds, constraints = _bounds_constraints(n)
    res = minimize(lambda w: -port_stats(w, mu, cov)[2], np.ones(n) / n,
                   bounds=bounds, constraints=constraints, method="SLSQP")
    return res.x


def w_min_variance(mu, cov):
    n = len(mu)
    bounds, constraints = _bounds_constraints(n)
    res = minimize(lambda w: port_stats(w, mu, cov)[1], np.ones(n) / n,
                   bounds=bounds, constraints=constraints, method="SLSQP")
    return res.x


def w_equal_weight(mu, cov):
    n = len(mu)
    return np.ones(n) / n


def w_risk_parity(mu, cov):
    """Paridade de risco: iguala as contribuições marginais de risco.
    Fallback inverse-vol se a otimização não convergir."""
    n = len(mu)
    cov = np.asarray(cov)

    def obj(w):
        port_var = w @ cov @ w
        mrc = cov @ w
        rc = w * mrc / port_var          # contribuições fracionárias de risco (somam 1)
        return np.sum((rc - 1.0 / n) ** 2)

    bounds = [(1e-4, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    res = minimize(obj, np.ones(n) / n, bounds=bounds,
                   constraints=constraints, method="SLSQP")
    if res.success:
        return res.x / res.x.sum()
    inv_vol = 1 / np.sqrt(np.diag(cov))
    return inv_vol / inv_vol.sum()


STRATEGY_FUNCS = {
    "Max Sharpe":    w_max_sharpe,
    "Min Variância": w_min_variance,
    "Equal-Weight":  w_equal_weight,
    "Risk Parity":   w_risk_parity,
}


# ═════════════════════════════════════════════════════════════════════════════
#  3. BACKTEST OUT-OF-SAMPLE (WALK-FORWARD)   (motor Rafael)
# ═════════════════════════════════════════════════════════════════════════════
def run_walk_forward(sector_rets, bench_rets):
    """Em cada data de rebalanceamento: estima mu/cov na janela passada
    (LOOKBACK_YEARS), calcula pesos de cada estratégia e os aplica no período
    seguinte usando retornos realizados. Retorna séries oos + últimos pesos."""
    sectors = list(sector_rets.columns)
    idx = sector_rets.index
    lookback_days = int(LOOKBACK_YEARS * 365.25)
    rebal_delta = pd.DateOffset(months=REBALANCE_MONTHS)

    start = idx[0] + timedelta(days=lookback_days)
    rebal_dates = []
    d = start
    while d <= idx[-1]:
        rebal_dates.append(d)
        d = d + rebal_delta

    oos = {s: [] for s in STRATEGIES}
    last_weights = {s: None for s in STRATEGIES}

    for i, rd in enumerate(rebal_dates):
        win = sector_rets.loc[rd - timedelta(days=lookback_days):rd]
        if len(win) < 60:
            continue
        mu, cov = win.mean(), win.cov()
        end = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else idx[-1] + timedelta(days=1)
        hold = sector_rets.loc[rd:end].iloc[1:] if rd in sector_rets.index else sector_rets.loc[rd:end]
        for s in STRATEGIES:
            w = STRATEGY_FUNCS[s](mu.values, cov.values)
            last_weights[s] = pd.Series(w, index=sectors)
            if len(hold):
                oos[s].append((hold * w).sum(axis=1))

    oos_series = {s: pd.concat(parts).sort_index() for s, parts in oos.items() if parts}
    # benchmark alinhado ao período out-of-sample
    if oos_series:
        oos_start = min(v.index[0] for v in oos_series.values())
        oos_series[BENCHMARK_NAME] = bench_rets.loc[oos_start:]
    return oos_series, last_weights, rebal_dates


# ═════════════════════════════════════════════════════════════════════════════
#  4. MÉTRICAS
# ═════════════════════════════════════════════════════════════════════════════
def cumulative(returns):
    return (1 + returns).cumprod()


def annualized_stats(returns):
    n = len(returns)
    total = cumulative(returns).iloc[-1] - 1
    ann_r = (1 + total) ** (TRADING_DAYS / n) - 1
    ann_v = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_r - RISK_FREE_RATE) / ann_v if ann_v else np.nan
    return ann_r, ann_v, sharpe


def max_drawdown(cum):
    return ((cum - cum.cummax()) / cum.cummax()).min()


def sortino(returns):
    mu = returns.mean() * TRADING_DAYS
    downside_std = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    return (mu - RISK_FREE_RATE) / downside_std if downside_std else np.nan


def var_cvar(returns, confidence=0.95):
    s = np.sort(returns.values)
    idx = int((1 - confidence) * len(s))
    return -s[idx], -s[:idx].mean()


def build_metrics(oos_series):
    rows = []
    for name, ret in oos_series.items():
        ret = ret.dropna()
        if len(ret) < 20:
            continue
        cum = cumulative(ret)
        a_r, a_v, sh = annualized_stats(ret)
        mdd = max_drawdown(cum)
        var, cvar = var_cvar(ret)
        rows.append({
            "Estratégia":         name,
            "Retorno Total":      cum.iloc[-1] - 1,
            "Retorno Anualiz.":   a_r,
            "Volatilidade Anual": a_v,
            "Sharpe":             sh,
            "Sortino":            sortino(ret),
            "Max Drawdown":       mdd,
            "VaR 95%":            var,
            "CVaR 95%":           cvar,
            "Calmar":             a_r / abs(mdd) if mdd else np.nan,
        })
    return pd.DataFrame(rows)


def build_period_summary(oos_series):
    """Retorno total acumulado por período (fatias do fim da série OOS) para cada
    estratégia e o benchmark. Alimenta a tabela do Resumo Executivo."""
    rows = []
    end = max(v.index[-1] for v in oos_series.values())
    for label, delta in PERIODS.items():
        cutoff = end - delta
        entry = {"Período": label}
        enough = False
        for name, ret in oos_series.items():
            r = ret.loc[cutoff:].dropna()
            if len(r) >= 20:
                entry[name] = cumulative(r).iloc[-1] - 1
                enough = True
            else:
                entry[name] = None
        if enough:
            rows.append(entry)
    return rows


# ═════════════════════════════════════════════════════════════════════════════
#  5. FRONTEIRA EFICIENTE (in-sample, contexto)   (motor Rafael)
# ═════════════════════════════════════════════════════════════════════════════
def run_frontier(sector_rets):
    mu, cov = sector_rets.mean(), sector_rets.cov()
    sectors = list(sector_rets.columns)
    n = len(sectors)
    mc_r, mc_v, mc_s = [], [], []
    for _ in range(N_PORTFOLIOS):
        w = np.random.dirichlet(np.ones(n))
        r, v, s = port_stats(w, mu.values, cov.values)
        mc_r.append(r); mc_v.append(v); mc_s.append(s)
    ws = w_max_sharpe(mu.values, cov.values)
    wv = w_min_variance(mu.values, cov.values)
    # setores individuais
    sec_pts = [(s, *port_stats(np.eye(n)[i], mu.values, cov.values)) for i, s in enumerate(sectors)]
    return {
        "mc": {"ret": np.array(mc_r), "vol": np.array(mc_v), "sharpe": np.array(mc_s)},
        "max_sharpe": port_stats(ws, mu.values, cov.values),
        "min_vol":    port_stats(wv, mu.values, cov.values),
        "sectors":    sec_pts,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  6. GRÁFICOS   (motor Rafael)
# ═════════════════════════════════════════════════════════════════════════════
def _hex_to_rgba(hex_color, alpha=0.2):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def fig_frontier(front):
    mc = front["mc"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mc["vol"] * 100, y=mc["ret"] * 100, mode="markers",
        marker=dict(size=4, color=mc["sharpe"], colorscale="Plasma", showscale=True,
                    opacity=0.55,
                    colorbar=dict(title=dict(text="Sharpe", font=dict(color="#E6EDF3")),
                                  tickfont=dict(color="#E6EDF3"))),
        name="Carteiras simuladas", hoverinfo="skip"))
    for name, r, v, s in front["sectors"]:
        fig.add_trace(go.Scatter(
            x=[v * 100], y=[r * 100], mode="markers+text",
            marker=dict(size=9, color="#8B949E", symbol="circle",
                        line=dict(color="#E6EDF3", width=1)),
            text=[name], textposition="top center", textfont=dict(size=9, color="#8B949E"),
            name=name, showlegend=False,
            hovertemplate=f"{name}<br>Retorno: {r*100:.1f}%<br>Vol: {v*100:.1f}%<extra></extra>"))
    for label, (r, v, s), color, symbol in [
        ("Max Sharpe", front["max_sharpe"], COLORS["max_sharpe"], "star"),
        ("Min Vol",    front["min_vol"],    COLORS["min_vol"],    "triangle-up"),
    ]:
        fig.add_trace(go.Scatter(
            x=[v * 100], y=[r * 100], mode="markers",
            marker=dict(size=18, color=color, symbol=symbol, line=dict(color="white", width=1.5)),
            name=label,
            hovertemplate=f"{label}<br>Retorno: {r*100:.1f}%<br>Vol: {v*100:.1f}%<br>Sharpe: {s:.2f}<extra></extra>"))
    fig.update_layout(title="Fronteira Eficiente entre Setores (in-sample)",
        xaxis_title="Volatilidade Anual (%)", yaxis_title="Retorno Anual Esperado (%)",
        **THEME, height=520, legend=dict(bgcolor="#161B22", bordercolor="#2D3748"),
        xaxis=dict(gridcolor=COLORS["grid"]), yaxis=dict(gridcolor=COLORS["grid"]))
    return fig


def fig_equity(oos_series):
    fig = go.Figure()
    for name, ret in oos_series.items():
        cum = (cumulative(ret.dropna()) - 1) * 100
        dash = "dash" if name == BENCHMARK_NAME else "solid"
        width = 1.8 if name == BENCHMARK_NAME else 2.4
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values, mode="lines", name=name,
            line=dict(color=STRAT_COLORS[name], width=width, dash=dash),
            hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1f}%<extra>" + name + "</extra>"))
    fig.add_hline(y=0, line_color=COLORS["grid"])
    fig.update_layout(title="Curva de Equity — Backtest Out-of-Sample",
        yaxis_title="Retorno Acumulado (%)", **THEME, height=480, hovermode="x unified",
        legend=dict(bgcolor="#161B22", bordercolor="#2D3748"),
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], ticksuffix="%"))
    return fig


def fig_drawdown(oos_series):
    fig = go.Figure()
    for name, ret in oos_series.items():
        cum = cumulative(ret.dropna())
        dd = ((cum - cum.cummax()) / cum.cummax()) * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, mode="lines", name=name,
            line=dict(color=STRAT_COLORS[name], width=1.8),
            hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1f}%<extra>" + name + "</extra>"))
    fig.update_layout(title="Drawdown — Out-of-Sample", yaxis_title="Drawdown (%)",
        **THEME, height=380, hovermode="x unified",
        legend=dict(bgcolor="#161B22", bordercolor="#2D3748"),
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], ticksuffix="%"))
    return fig


def fig_rolling_sharpe(oos_series):
    fig = go.Figure()
    for name, ret in oos_series.items():
        ret = ret.dropna()
        roll_mu = ret.rolling(TRADING_DAYS).mean() * TRADING_DAYS
        roll_vol = ret.rolling(TRADING_DAYS).std() * np.sqrt(TRADING_DAYS)
        roll_sh = (roll_mu - RISK_FREE_RATE) / roll_vol
        fig.add_trace(go.Scatter(
            x=roll_sh.index, y=roll_sh.values, mode="lines", name=name,
            line=dict(color=STRAT_COLORS[name], width=1.8),
            hovertemplate="%{x|%d/%m/%Y}<br>Sharpe: %{y:.2f}<extra>" + name + "</extra>"))
    fig.add_hline(y=1, line_dash="dot", line_color="#E6EDF3",
                  annotation_text="Sharpe = 1", annotation_font_color="#E6EDF3")
    fig.add_hline(y=0, line_color=COLORS["grid"])
    fig.update_layout(title="Rolling Sharpe (252 dias) — Out-of-Sample",
        yaxis_title="Sharpe", **THEME, height=380,
        legend=dict(bgcolor="#161B22", bordercolor="#2D3748"),
        xaxis=dict(gridcolor=COLORS["grid"]), yaxis=dict(gridcolor=COLORS["grid"]))
    return fig


def fig_correlation(sector_rets):
    corr = sector_rets.corr()
    labels = list(corr.columns)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale=[[0, "#EF5350"], [0.5, "#161B22"], [1, "#26A69A"]],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}", textfont=dict(size=9),
        colorbar=dict(title=dict(text="Correlação", font=dict(color="#E6EDF3")),
                      tickfont=dict(color="#E6EDF3")),
        hovertemplate="%{y} × %{x}<br>Correlação: %{z:.2f}<extra></extra>"))
    fig.update_layout(title="Matriz de Correlação entre Setores", **THEME, height=560,
        xaxis=dict(tickfont=dict(color="#E6EDF3"), tickangle=-45),
        yaxis=dict(tickfont=dict(color="#E6EDF3")))
    return fig


def fig_allocation(last_weights):
    sectors = list(next(iter(last_weights.values())).index)
    palette = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    fig = go.Figure()
    for i, sec in enumerate(sectors):
        fig.add_trace(go.Bar(
            name=sec, x=STRATEGIES, y=[last_weights[s][sec] * 100 for s in STRATEGIES],
            marker_color=palette[i % len(palette)],
            hovertemplate="%{x}<br>" + sec + ": %{y:.1f}%<extra></extra>"))
    fig.update_layout(title="Alocação Setorial Recomendada (última janela) por Estratégia",
        barmode="stack", yaxis_title="Peso (%)", **THEME, height=480,
        legend=dict(bgcolor="#161B22", bordercolor="#2D3748", font=dict(size=10)),
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], ticksuffix="%"))
    return fig


def fig_bar_returns(metrics_df):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics_df["Estratégia"], y=metrics_df["Retorno Anualiz."] * 100,
        marker_color=[STRAT_COLORS.get(s, "#8B949E") for s in metrics_df["Estratégia"]],
        text=[f"{v*100:+.1f}%" for v in metrics_df["Retorno Anualiz."]],
        textposition="outside", textfont=dict(color="#E6EDF3"),
        hovertemplate="%{x}<br>Ret. anualiz.: %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=0, line_color=COLORS["grid"])
    fig.update_layout(title="Retorno Anualizado Out-of-Sample por Estratégia",
        yaxis_title="Retorno (%)", **THEME, height=400,
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], ticksuffix="%"))
    return fig


def fig_composition(composition):
    """Heatmap setor × ação com o peso (market cap) de cada ação no setor."""
    sectors = list(composition.keys())
    stocks = sorted({s for d in composition.values() for s in d})
    z = [[composition[sec].get(stk, np.nan) * 100 for stk in stocks] for sec in sectors]
    fig = go.Figure(go.Heatmap(
        z=z, x=stocks, y=sectors, colorscale="Viridis",
        hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
        colorbar=dict(title=dict(text="Peso %", font=dict(color="#E6EDF3")),
                      tickfont=dict(color="#E6EDF3"))))
    fig.update_layout(title="Composição dos Setores — peso de cada ação por market cap",
        **THEME, height=520,
        xaxis=dict(tickfont=dict(color="#E6EDF3", size=9), tickangle=-90),
        yaxis=dict(tickfont=dict(color="#E6EDF3", size=10)))
    return fig


def fig_metrics_table(metrics_df):
    fmt = {
        "Retorno Total":      lambda v: f"{v:+.1%}",
        "Retorno Anualiz.":   lambda v: f"{v:+.1%}",
        "Volatilidade Anual": lambda v: f"{v:.1%}",
        "Sharpe":             lambda v: f"{v:.2f}",
        "Sortino":            lambda v: f"{v:.2f}",
        "Max Drawdown":       lambda v: f"{v:.1%}",
        "VaR 95%":            lambda v: f"{v:.2%}",
        "CVaR 95%":           lambda v: f"{v:.2%}",
        "Calmar":             lambda v: f"{v:.2f}",
    }
    display = metrics_df.copy()
    for col, fn in fmt.items():
        if col in display.columns:
            display[col] = display[col].apply(fn)
    cols = list(display.columns)
    cell_vals = [display[c].tolist() for c in cols]
    fig = go.Figure(go.Table(
        header=dict(values=[f"<b>{c}</b>" for c in cols], fill_color="#21262D",
                    font=dict(color="#E6EDF3", size=11), align="center", height=32,
                    line=dict(color="#2D3748")),
        cells=dict(values=cell_vals, fill_color="#161B22",
                   font=dict(color="#E6EDF3", size=11), align="center", height=30,
                   line=dict(color="#2D3748"))))
    fig.update_layout(title="Métricas Out-of-Sample por Estratégia", **THEME, height=320)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  7. TEXTOS DO DASHBOARD   (funções Pedro)
# ═════════════════════════════════════════════════════════════════════════════
# Título + texto explicativo exibido acima de cada gráfico (chave = id do fig)
CHART_INTROS = {
    "equity": ("Curva de Equity — Out-of-Sample",
        "Retorno acumulado (%) de cada estratégia ao longo do backtest walk-forward, comparado ao "
        "Ibovespa. Diferente de um backtest in-sample, os pesos aqui foram estimados só com dados "
        "passados e aplicados no período seguinte — reflete melhor o desempenho realista."),
    "bar_returns": ("Retorno Anualizado por Estratégia",
        "Retorno anualizado out-of-sample de cada estratégia de alocação entre setores. Facilita "
        "comparar rapidamente qual abordagem entregou mais retorno no período."),
    "metrics_table": ("Métricas por Estratégia",
        "Consolidado numérico out-of-sample (retorno, volatilidade, Sharpe, Sortino, drawdown, VaR/CVaR, "
        "Calmar) de cada estratégia, para leitura rápida sem precisar interpretar os gráficos."),
    "frontier": ("Fronteira Eficiente entre Setores",
        "Cada ponto é uma carteira de setores simulada (Monte Carlo); o eixo X é a volatilidade anual "
        "(risco) e o eixo Y o retorno esperado. Os marcadores destacados são as carteiras ótimas (Máximo "
        "Sharpe e Mínima Volatilidade). Os pontos cinza são os setores individuais. Contexto in-sample."),
    "allocation": ("Alocação Setorial Recomendada",
        "Distribuição dos pesos entre os setores na última janela de estimação, comparando as quatro "
        "estratégias lado a lado. Mostra em quais setores cada abordagem concentra ou dilui a posição."),
    "drawdown": ("Drawdown — Out-of-Sample",
        "Queda percentual em relação ao topo histórico mais recente (pico-a-vale) de cada estratégia. "
        "Quanto mais negativo e prolongado, maior o risco de perda e o tempo de recuperação."),
    "rolling_sharpe": ("Rolling Sharpe (252 dias)",
        "Sharpe Ratio calculado em janelas móveis de 1 ano, mostrando se a relação risco-retorno de cada "
        "estratégia foi consistente ao longo do tempo ou concentrada em poucos períodos bons."),
    "correlation": ("Matriz de Correlação entre Setores",
        "Correlação histórica entre os retornos dos setores. Valores próximos de 1 indicam setores que se "
        "movem juntos (pouca diversificação); valores baixos ou negativos ajudam a reduzir o risco total."),
    "composition": ("Composição dos Setores",
        "Peso de cada ação dentro do seu setor, definido pelo market cap atual (snapshot). Mostra quais "
        "papéis dominam cada índice setorial sintético usado no backtest."),
}

# Rótulo curto de cada gráfico no menu de navegação (chave = id do fig)
NAV_LABELS = {
    "equity": "Equity", "bar_returns": "Retorno/Estratégia", "metrics_table": "Métricas",
    "frontier": "Fronteira Eficiente", "allocation": "Alocação", "drawdown": "Drawdown",
    "rolling_sharpe": "Rolling Sharpe", "correlation": "Correlação", "composition": "Composição",
}

GLOSSARY = [
    ("Backtest Out-of-Sample", "Teste em que os pesos são estimados apenas com dados passados (janela de "
                               "lookback) e aplicados no período seguinte, sem 'espiar o futuro' — mais "
                               "realista que o in-sample."),
    ("Walk-Forward", "Repetição do processo out-of-sample ao longo do tempo: reestima os pesos a cada "
                     "rebalanceamento e avança, emendando os trechos numa curva contínua."),
    ("Markowitz / Máx. Sharpe", "Otimização que busca a carteira com a melhor relação retorno-risco "
                                "(maior Sharpe) dentro da fronteira eficiente."),
    ("Risk Parity", "Aloca de forma que cada setor contribua com a mesma parcela do risco total da "
                    "carteira, em vez de igualar os pesos financeiros."),
    ("Sharpe Ratio", "Retorno em excesso ao ativo livre de risco, dividido pela volatilidade. Quanto "
                     "maior, melhor a relação risco-retorno."),
    ("Volatilidade", "Desvio-padrão anualizado dos retornos; mede a intensidade das oscilações, usada "
                     "como proxy de risco."),
    ("Drawdown", "Maior queda percentual entre um pico e o vale seguinte antes de uma nova máxima ser "
                 "atingida."),
    ("Fronteira Eficiente", "Conjunto de carteiras que oferecem o maior retorno esperado para cada nível "
                            "de risco, segundo a teoria de Markowitz."),
    ("Correlação", "Mede o quanto dois setores se movem juntos, de -1 (opostos) a +1 (idênticos); "
                   "correlações baixas melhoram a diversificação."),
]

BACK_TO_TOP = """<button id="back-to-top" aria-label="Voltar ao topo" onclick="window.scrollTo({top:0,behavior:'smooth'})">&uarr;</button>
<script>
  (function(){
    var btn = document.getElementById('back-to-top');
    window.addEventListener('scroll', function(){
      btn.classList.toggle('show', window.scrollY > 600);
    }, {passive:true});
  })();
</script>"""


def _pct(v, decimals=1, signed=True):
    """Formata fração como percentual pt-BR (vírgula decimal). None vira '—'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    fmt = f"{{:+.{decimals}f}}" if signed else f"{{:.{decimals}f}}"
    return fmt.format(v * 100).replace(".", ",") + "%"


def _css_id(t):
    """ID seguro para HTML/CSS a partir de um rótulo. Usa só ASCII alfanumérico
    para casar exatamente com o regex [^a-zA-Z0-9] do JS do simulador."""
    return "".join(c if (c.isascii() and c.isalnum()) else "_" for c in t)


def build_nav(fig_keys, has_sim=False, has_quotes=False):
    links = ['<a href="#resumo">Resumo</a>', '<a href="#tabela-alocacao">Alocação Ótima</a>']
    if has_sim:
        links.append('<a href="#simulador">🎚️ Simulador</a>')
    links += [f'<a href="#fig-{k}">{NAV_LABELS.get(k, k)}</a>' for k in fig_keys]
    if has_quotes:
        links.append('<a href="#cotacoes">💰 Cotações</a>')
    links.append('<a href="#glossario">Glossário</a>')
    return ('<nav class="toc" aria-label="Navegação do relatório">\n  '
            + "\n  ".join(links) + "\n</nav>")


def build_glossary():
    items = "".join(f"<div><dt>{term}</dt><dd>{desc}</dd></div>" for term, desc in GLOSSARY)
    return (f'<div class="glossary" id="glossario"><div class="container">'
            f'<h2>Glossário</h2><dl>{items}</dl></div></div>')


def build_summary_section(summary_rows, best, last_weights):
    """Resumo executivo: texto dinâmico + tabela de retornos OOS por estratégia/benchmark."""
    names = STRATEGIES + [BENCHMARK_NAME]
    header = "<tr><th>Período</th>" + "".join(f"<th>{n}</th>" for n in names) + "</tr>"
    body = ""
    for row in summary_rows:
        cols = [row.get(n) for n in names]
        vals = [v for v in cols if v is not None]
        top = max(vals) if vals else None
        tds = ""
        for v in cols:
            if v is not None and top is not None and abs(v - top) < 1e-12:
                tds += f'<td style="color:#26A69A"><b>{_pct(v, decimals=1)}</b></td>'
            else:
                tds += f"<td>{_pct(v, decimals=1)}</td>"
        label = PERIOD_LABELS.get(row["Período"], row["Período"])
        body += f"<tr><td>{label}</td>{tds}</tr>"

    b_sh = best["sharpe"]
    best_name = b_sh["Estratégia"]
    top_str = "—"
    if last_weights.get(best_name) is not None:
        pairs = sorted(last_weights[best_name].items(), key=lambda x: -x[1])
        top = [(s, w) for s, w in pairs if w >= 0.05][:5] or pairs[:3]
        top_str = ", ".join(f"{s} ({_pct(w, signed=False)})" for s, w in top)
    period_names = ", ".join(PERIOD_LABELS.get(r["Período"], r["Período"]) for r in summary_rows)

    return f"""<section class="summary-section" id="resumo">
  <h2>Resumo Executivo</h2>
  <p>Este relatório compara quatro estratégias de alocação <b>entre setores da B3</b>
  (Máximo Sharpe, Mínima Variância, Equal-Weight e Risk Parity) num backtest
  <b>walk-forward out-of-sample</b>: a cada {REBALANCE_MONTHS} mês(es) os pesos são reestimados numa
  janela de {LOOKBACK_YEARS} anos e aplicados no período seguinte, medindo o retorno acumulado contra o
  <b>{BENCHMARK_NAME}</b> em janelas de {period_names}.</p>
  <p>No período completo, a estratégia com melhor relação risco-retorno foi
  <b style="color:#FF6F00">{best_name}</b> (Sharpe {b_sh['Sharpe']:.2f}), que na última janela concentra
  posição em {top_str}. Estratégias otimizadas são ajustadas com dados históricos e não garantem repetição
  futura desse desempenho — vale conferir a <a href="#fig-correlation">matriz de correlação</a> e a
  <a href="#fig-allocation">alocação recomendada</a> antes de aplicar os pesos na prática.</p>
  <table class="summary-table">
    <thead>{header}</thead>
    <tbody>{body}</tbody>
  </table>
  <p class="summary-note">Retornos acumulados out-of-sample (fatias do fim da série). O período mais longo
  pode ser menor que a janela nominal porque os primeiros {LOOKBACK_YEARS} anos do histórico são consumidos
  pela estimação inicial.</p>
</section>"""


def build_allocation_table(last_weights):
    """Tabela compacta: peso recomendado de cada setor por estratégia (última janela)."""
    sectors = list(next(iter(last_weights.values())).index)
    ordered = sorted(sectors, key=lambda s: -last_weights["Max Sharpe"][s])
    header = "<tr><th>Setor</th>" + "".join(f"<th>{s}</th>" for s in STRATEGIES) + "</tr>"
    body = ""
    for sec in ordered:
        tds = ""
        for s in STRATEGIES:
            w = last_weights[s][sec]
            color = "#26A69A" if w >= 0.15 else ("#8B949E" if w < 0.005 else "#E6EDF3")
            tds += f'<td style="color:{color}">{w*100:.1f}%</td>'
        body += f'<tr><td style="text-align:left">{sec}</td>{tds}</tr>'
    return f"""<div class="alloc-section" id="tabela-alocacao">
    <h2>Alocação Recomendada por Estratégia (última janela)</h2>
    <table>
      <thead>{header}</thead>
      <tbody>{body}</tbody>
    </table>
  </div>"""


# ═════════════════════════════════════════════════════════════════════════════
#  7b. COTAÇÕES — preço atual + faixa de 52 semanas   (função Pedro, adaptada à B3)
# ═════════════════════════════════════════════════════════════════════════════
def _money_brl(v):
    """Formata em Reais no padrão pt-BR: R$ 1.234,56."""
    s = f"{v:,.2f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return f"R$ {s}"


def fetch_native_prices(stocks, days=400):
    """Preços de fechamento em moeda nativa (R$) das ações B3 p/ a tabela de cotações."""
    yahoo = [_sa(t) for t in dict.fromkeys(stocks)]
    start = datetime.today() - timedelta(days=days)
    data = yf.download(yahoo, start=start, progress=False, auto_adjust=False)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    return close.dropna(how="all")


def _quote_from_series(s):
    """(atual, mín52, máx52, posição 0-1, vs. máx) a partir de uma série de preços."""
    corte = datetime.today() - timedelta(days=365)
    janela = s.loc[corte:]
    if janela.empty:
        janela = s
    last = float(s.iloc[-1])
    hi, lo = float(janela.max()), float(janela.min())
    pos = (last - lo) / (hi - lo) if hi > lo else 0.5
    return last, lo, hi, max(0.0, min(1.0, pos)), (last / hi - 1 if hi else 0.0)


def build_quotes_section(native_prices, composition):
    """Preço atual e faixa de 52 semanas de cada ação, agrupado por setor (R$)."""
    if native_prices is None or native_prices.empty:
        return ""
    body = ""
    n_ativos = 0
    for setor, comp in composition.items():
        # ações do setor ordenadas por peso (market cap) desc
        stocks = sorted(comp.items(), key=lambda x: -x[1])
        linhas = ""
        for tk, _w in stocks:
            col = _sa(tk)
            if col not in native_prices.columns:
                continue
            s = native_prices[col].dropna()
            if s.empty:
                continue
            last, lo, hi, pos, vsmax = _quote_from_series(s)
            posp = pos * 100
            n_ativos += 1
            linhas += f"""
      <tr>
        <td style="text-align:left"><b>{tk}</b></td>
        <td><b>{_money_brl(last)}</b></td>
        <td>{_money_brl(lo)}</td>
        <td>{_money_brl(hi)}</td>
        <td class="range-cell">
          <div class="range-bar"><div class="range-dot" style="left:{posp:.0f}%"></div></div>
          <span class="range-txt">{posp:.0f}% da faixa · {_pct(vsmax)} vs. máx</span>
        </td>
      </tr>"""
        if linhas:
            body += (f'<tr class="sector-head"><td colspan="5" style="text-align:left">'
                     f'🏭 {setor}</td></tr>{linhas}')
    if not body:
        return ""
    return f"""<section class="summary-section" id="cotacoes">
  <h2>💰 Cotações — Preço Atual e Faixa de 52 Semanas</h2>
  <p>Preço de fechamento mais recente de cada ação da B3 (em R$), com a <b>máxima e a mínima dos últimos
  12 meses</b>, agrupadas pelos {len(composition)} setores. A barra mostra onde o preço atual está na faixa:
  à <span style="color:#EF5350">esquerda (vermelho)</span> = perto da mínima;
  à <span style="color:#26A69A">direita (verde)</span> = perto da máxima. São as mesmas ações que compõem
  os índices setoriais deste relatório.</p>
  <table class="summary-table">
    <thead><tr>
      <th style="text-align:left">Ação</th><th>Preço Atual</th><th>Mín. 52s</th><th>Máx. 52s</th>
      <th style="text-align:left">Posição na faixa (52 semanas)</th>
    </tr></thead>
    <tbody>{body}
    </tbody>
  </table>
  <p class="summary-note">Atualizado a cada execução. "vs. máx" = variação em relação à máxima de 52 semanas
  (quanto o preço está abaixo do topo do período). {n_ativos} ações listadas.</p>
</section>"""


# ═════════════════════════════════════════════════════════════════════════════
#  8. SIMULADOR INTERATIVO   (função Pedro, adaptada a setores/estratégias)
# ═════════════════════════════════════════════════════════════════════════════
SIMULATOR_JS = r"""
<script>
(function(){
  const SIM = __SIM_JSON__;
  const DATEOBJ = SIM.dates.map(d => new Date(d));
  const LAST = DATEOBJ[DATEOBJ.length - 1];
  const sliders = {};
  let curPeriod = "3A";

  const cssId = t => t.replace(/[^a-zA-Z0-9]/g, "_");
  const fmtPct = (v, dec=1, signed=true) => {
    if (!isFinite(v)) return "—";
    let s = (signed && v > 0 ? "+" : "") + (v*100).toFixed(dec);
    return s.replace(".", ",") + "%";
  };
  function currentWeights(){
    const raw = SIM.sectors.map(t => sliders[t] ? +sliders[t].value : 0);
    const sum = raw.reduce((a,b) => a+b, 0);
    return sum <= 0 ? raw.map(() => 0) : raw.map(x => x/sum);
  }
  function startIndex(days){
    const cutoff = new Date(LAST.getTime() - days*86400000);
    for (let i=0; i<DATEOBJ.length; i++) if (DATEOBJ[i] >= cutoff) return i;
    return 0;
  }
  function cumSeries(daily){
    let c = 1; const out = [];
    for (const r of daily){ c *= (1+r); out.push((c-1)*100); }
    return out;
  }
  function stats(daily){
    const n = daily.length;
    if (!n) return {ret:NaN, vol:NaN, sharpe:NaN, mdd:NaN, total:NaN};
    let c=1, peak=1, mdd=0, sum=0;
    for (const r of daily){ c*=(1+r); if(c>peak)peak=c; const dd=(c-peak)/peak; if(dd<mdd)mdd=dd; sum+=r; }
    const total = c-1, mean = sum/n;
    let vs=0; for (const r of daily) vs += (r-mean)*(r-mean);
    const std = Math.sqrt(vs/((n-1)||1));
    const annR = Math.pow(1+total, 252/n)-1;
    const annV = std*Math.sqrt(252);
    return {ret:annR, vol:annV, sharpe: annV ? (annR-SIM.rf)/annV : NaN, mdd:mdd, total:total};
  }
  function setCard(id, val, color){
    const el = document.getElementById(id);
    if (el){ el.textContent = val; if (color) el.style.color = color; }
  }
  function compute(){
    const w = currentWeights();
    const days = SIM.periods[curPeriod];
    const start = startIndex(days);
    const dates = SIM.dates.slice(start);
    const N = dates.length;
    const daily = new Array(N).fill(0);
    SIM.sectors.forEach((t, ti) => {
      const wt = w[ti]; if (!wt) return;
      const arr = SIM.returns[t];
      for (let i=0; i<N; i++) daily[i] += wt*arr[start+i];
    });
    SIM.sectors.forEach((t, ti) => {
      const el = document.getElementById("w-"+cssId(t));
      if (el) el.textContent = fmtPct(w[ti], 1, false);
    });
    const s = stats(daily);
    setCard("sim-ret", fmtPct(s.ret), s.ret>=0 ? "#26A69A" : "#EF5350");
    setCard("sim-vol", fmtPct(s.vol,1,false), "#E6EDF3");
    setCard("sim-sharpe", isFinite(s.sharpe) ? s.sharpe.toFixed(2) : "—",
            s.sharpe>=1 ? "#26A69A" : (s.sharpe>=0 ? "#FFA726" : "#EF5350"));
    setCard("sim-mdd", fmtPct(s.mdd), "#EF5350");
    setCard("sim-total", fmtPct(s.total), s.total>=0 ? "#26A69A" : "#EF5350");
    if (!window.Plotly) return;
    const traces = [{x:dates, y:cumSeries(daily), mode:"lines", name:"Sua carteira",
                     line:{color:"#FF6F00", width:3},
                     hovertemplate:"%{x|%d/%m/%Y}<br>%{y:.1f}%<extra>Sua carteira</extra>"}];
    Object.keys(SIM.benchmarks).forEach(bn => {
      const arr = SIM.benchmarks[bn].slice(start);
      traces.push({x:dates, y:cumSeries(arr), mode:"lines", name:bn,
        line:{color:"#EF5350", width:1.5, dash:"dash"},
        hovertemplate:"%{x|%d/%m/%Y}<br>%{y:.1f}%<extra>"+bn+"</extra>"});
    });
    Plotly.react("sim-chart", traces, {
      paper_bgcolor:"#0D1117", plot_bgcolor:"#161B22",
      font:{color:"#E6EDF3", family:"Inter, Arial, sans-serif"},
      margin:{t:12, r:12, b:40, l:52}, height:440, hovermode:"x unified",
      legend:{orientation:"h", y:1.12, bgcolor:"rgba(0,0,0,0)"},
      xaxis:{gridcolor:"#2D3748"}, yaxis:{gridcolor:"#2D3748", ticksuffix:"%", title:"Retorno acumulado (%)"}
    }, {responsive:true, displayModeBar:false});
  }
  function setWeights(arr){
    SIM.sectors.forEach((t,i) => { if (sliders[t]) sliders[t].value = Math.round(arr[i]*100); });
    compute();
  }
  function applyPreset(name){
    if (name === "__equal__") setWeights(SIM.sectors.map(() => 1/SIM.sectors.length));
    else if (name === "__zero__") setWeights(SIM.sectors.map(() => 0));
    else if (SIM.presets[name]) setWeights(SIM.presets[name]);
  }
  function waitPlotly(){ if (window.Plotly) compute(); else setTimeout(waitPlotly, 120); }
  function init(){
    SIM.sectors.forEach(t => {
      sliders[t] = document.getElementById("slider-"+cssId(t));
      if (sliders[t]) sliders[t].addEventListener("input", compute);
    });
    document.querySelectorAll(".sim-periods button").forEach(btn =>
      btn.addEventListener("click", () => {
        curPeriod = btn.dataset.period;
        document.querySelectorAll(".sim-periods button").forEach(b => b.classList.remove("active"));
        btn.classList.add("active"); compute();
      }));
    document.querySelectorAll(".sim-preset").forEach(btn =>
      btn.addEventListener("click", () => applyPreset(btn.dataset.preset)));
    waitPlotly();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
</script>
"""


def build_simulator(sector_rets, bench_rets, last_weights):
    """Painel interativo: sliders de peso por setor recalculando retorno/risco/Sharpe/curva
    no navegador. Presets carregam os pesos de cada estratégia (última janela)."""
    sectors = list(sector_rets.columns)
    rets = sector_rets.dropna()
    if len(rets) < 60:
        return ""
    bench = bench_rets.reindex(rets.index).fillna(0)

    presets = {}
    for s in STRATEGIES:
        if last_weights.get(s) is not None:
            presets[s] = [round(float(last_weights[s][sec]), 4) for sec in sectors]
    default_w = presets.get("Max Sharpe", [1.0 / len(sectors)] * len(sectors))

    data = {
        "dates":      [d.strftime("%Y-%m-%d") for d in rets.index],
        "returns":    {s: [round(float(x), 5) for x in rets[s].values] for s in sectors},
        "benchmarks": {BENCHMARK_NAME: [round(float(x), 5) for x in bench.values]},
        "sectors":    sectors,
        "presets":    presets,
        "rf":         RISK_FREE_RATE,
        "periods":    {"6M": 183, "1A": 365, "3A": 1095, "5A": 1825, "10A": 3650},
    }
    sim_json = json.dumps(data, ensure_ascii=False)

    rows = ""
    for sec, wi in zip(sectors, default_w):
        cid = _css_id(sec)
        rows += f"""
        <div class="sim-row">
          <span class="sim-name">{sec}</span>
          <input type="range" min="0" max="100" step="1" value="{round(wi*100)}"
                 id="slider-{cid}" class="sim-slider" aria-label="Peso de {sec}">
          <span class="sim-weight" id="w-{cid}">{wi*100:.1f}%</span>
        </div>"""

    preset_btns = "".join(
        f'<button class="sim-preset" type="button" data-preset="{s}">{s}</button>'
        for s in presets
    )
    preset_btns += ('<button class="sim-preset" type="button" data-preset="__equal__">Igualar (1/N)</button>'
                    '<button class="sim-preset" type="button" data-preset="__zero__">Zerar</button>')

    def _pbtn(k):
        cls = ' class="active"' if k == "3A" else ''
        return f'<button data-period="{k}"{cls}>{k}</button>'
    periods_html = "".join(_pbtn(k) for k in ["6M", "1A", "3A", "5A", "10A"])

    section = f"""<section class="summary-section" id="simulador">
  <h2>🎚️ Simulador de Carteira Setorial</h2>
  <p>Arraste os controles para mudar o peso de cada setor — retorno, risco, Sharpe e a curva abaixo
  recalculam <b>na hora</b>. Os pesos são normalizados automaticamente para somar 100%. Os botões de preset
  carregam a alocação de cada estratégia (última janela). <i>Esta é uma simulação estática (pesos fixos no
  período todo), útil como "e se…"; o resultado rigoroso é o backtest walk-forward das seções abaixo.</i></p>
  <div class="sim-periods">Período:&nbsp; {periods_html}</div>
  <div class="cards" style="margin:18px 0">
    <div class="card"><div class="card-label">Retorno anualizado</div>
        <div class="card-value" id="sim-ret">—</div></div>
    <div class="card"><div class="card-label">Retorno total (período)</div>
        <div class="card-value" id="sim-total">—</div></div>
    <div class="card"><div class="card-label">Volatilidade anual</div>
        <div class="card-value" id="sim-vol">—</div></div>
    <div class="card"><div class="card-label">Sharpe Ratio</div>
        <div class="card-value" id="sim-sharpe">—</div></div>
    <div class="card"><div class="card-label">Máx. Drawdown</div>
        <div class="card-value" id="sim-mdd">—</div></div>
  </div>
  <div class="sim-layout">
    <div class="sim-controls">
      <div class="sim-buttons">
        {preset_btns}
      </div>
      {rows}
    </div>
    <div class="chart-wrap sim-chartwrap"><div id="sim-chart" style="height:440px;width:100%"></div></div>
  </div>
</section>
{SIMULATOR_JS.replace("__SIM_JSON__", sim_json)}"""
    return section


# ═════════════════════════════════════════════════════════════════════════════
#  9. HTML   (estrutura Pedro, adaptada)
# ═════════════════════════════════════════════════════════════════════════════
def build_html(figures_html, metrics_df, best, n_sectors, oos_start, oos_end,
               summary_html, alloc_html, simulator_html="", quotes_html=""):
    b_sh, b_ret, b_vol = best["sharpe"], best["ret"], best["vol"]
    summary_cards = f"""
    <div class="cards">
        <div class="card"><div class="card-label">Melhor por Sharpe (OOS)</div>
            <div class="card-value" style="color:#FF6F00">{b_sh['Estratégia']}</div>
            <div class="card-sub">Sharpe {b_sh['Sharpe']:.2f}</div></div>
        <div class="card"><div class="card-label">Maior Retorno Anualiz.</div>
            <div class="card-value" style="color:#26A69A">{b_ret['Estratégia']}</div>
            <div class="card-sub">{b_ret['Retorno Anualiz.']:+.1%} a.a.</div></div>
        <div class="card"><div class="card-label">Menor Volatilidade</div>
            <div class="card-value" style="color:#2196F3">{b_vol['Estratégia']}</div>
            <div class="card-sub">{b_vol['Volatilidade Anual']:.1%} a.a.</div></div>
        <div class="card"><div class="card-label">Setores Analisados</div>
            <div class="card-value">{n_sectors}</div>
            <div class="card-sub">B3 · Yahoo Finance</div></div>
    </div>"""
    sections = "".join(
        f'<section class="chart-section" id="fig-{k}">'
        f'<div class="section-intro"><h2>{CHART_INTROS.get(k, ("", ""))[0]}</h2>'
        f'<p>{CHART_INTROS.get(k, ("", ""))[1]}</p></div>'
        f'<div class="chart-wrap">{v}</div></section>'
        for k, v in figures_html.items()
    )
    nav_toc       = build_nav(list(figures_html.keys()), has_sim=bool(simulator_html),
                              has_quotes=bool(quotes_html))
    glossary_html = build_glossary()
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Diversificação Setorial B3 — Backtest Unificado</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0D1117;color:#E6EDF3;font-family:Inter,Arial,sans-serif;padding:0 0 60px}}
  header{{background:linear-gradient(135deg,#161B22 0%,#0D1117 100%);
          border-bottom:1px solid #2D3748;padding:32px 48px 24px}}
  header h1{{font-size:2rem;font-weight:700;letter-spacing:-0.5px}}
  header h1 span{{color:#FF6F00}}
  header p{{color:#8B949E;margin-top:6px;font-size:.9rem}}
  .container{{max-width:1400px;margin:0 auto;padding:0 32px}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin:32px 0}}
  .card{{background:#161B22;border:1px solid #2D3748;border-radius:10px;padding:20px;transition:border-color .2s}}
  .card:hover{{border-color:#FF6F00}}
  .card-label{{font-size:.75rem;color:#8B949E;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px}}
  .card-value{{font-size:1.6rem;font-weight:700}}
  .card-sub{{font-size:.8rem;color:#8B949E;margin-top:4px}}
  .chart-section{{margin:24px 0;scroll-margin-top:64px}}
  .chart-wrap{{background:#161B22;border:1px solid #2D3748;border-radius:10px;padding:8px;overflow:hidden}}
  .alloc-section{{margin:24px 0;scroll-margin-top:64px}}
  .alloc-section h2{{font-size:1rem;color:#8B949E;margin-bottom:12px;text-transform:uppercase;letter-spacing:.8px}}
  table{{width:100%;border-collapse:collapse;background:#161B22;border-radius:10px;overflow:hidden;border:1px solid #2D3748}}
  th{{background:#21262D;padding:12px 16px;text-align:center;font-size:.8rem;color:#8B949E;text-transform:uppercase;letter-spacing:.6px}}
  td{{padding:10px 16px;text-align:center;border-top:1px solid #2D3748;font-size:.9rem}}
  tr:hover td{{background:#1C2128}}
  footer{{text-align:center;color:#3D444D;font-size:.75rem;margin-top:48px}}
  html{{scroll-behavior:smooth}}
  .toc{{position:sticky;top:0;z-index:50;background:#161B22;border-bottom:1px solid #2D3748;
       padding:10px 32px;display:flex;gap:6px;overflow-x:auto;white-space:nowrap}}
  .toc a{{color:#8B949E;font-size:.78rem;text-decoration:none;padding:6px 12px;border-radius:16px;
         border:1px solid transparent;transition:.15s}}
  .toc a:hover{{color:#E6EDF3;border-color:#2D3748}}
  .toc a:focus-visible{{outline:2px solid #FF6F00}}
  .summary-section{{background:#161B22;border:1px solid #2D3748;border-radius:10px;
                    padding:24px 28px;margin:32px 0;scroll-margin-top:64px}}
  .summary-section h2{{font-size:1.1rem;margin-bottom:12px}}
  .summary-section p{{color:#C9D1D9;font-size:.9rem;line-height:1.6;margin-bottom:14px}}
  .summary-table{{width:100%;border-collapse:collapse;margin-top:8px}}
  .summary-table th{{background:#21262D;padding:8px 12px;font-size:.75rem;color:#8B949E;
                     text-transform:uppercase;letter-spacing:.5px;text-align:center}}
  .summary-table td{{padding:8px 12px;text-align:center;font-size:.85rem;border-top:1px solid #2D3748}}
  .summary-note{{color:#8B949E;font-size:.75rem;margin-top:10px}}
  .summary-section code{{background:#0D1117;border:1px solid #2D3748;border-radius:4px;
                         padding:1px 5px;font-size:.85em;color:#FFA726}}
  .sector-head td{{background:#1C2333;color:#7FB2FF;font-weight:600;font-size:.8rem;
                   text-transform:uppercase;letter-spacing:.5px}}
  .range-cell{{min-width:180px;text-align:left}}
  .range-bar{{position:relative;height:8px;border-radius:4px;margin:2px 0 4px;
              background:linear-gradient(90deg,#EF5350 0%,#FFA726 50%,#26A69A 100%)}}
  .range-dot{{position:absolute;top:-3px;width:4px;height:14px;background:#E6EDF3;border-radius:2px;
              transform:translateX(-50%);box-shadow:0 0 3px rgba(0,0,0,.6)}}
  .range-txt{{font-size:.72rem;color:#8B949E}}
  .sim-periods{{display:flex;align-items:center;gap:6px;color:#8B949E;font-size:.8rem;flex-wrap:wrap}}
  .sim-periods button{{background:#21262D;color:#8B949E;border:1px solid #2D3748;border-radius:14px;
                       padding:4px 14px;cursor:pointer;font-size:.8rem;transition:.15s}}
  .sim-periods button:hover{{color:#E6EDF3;border-color:#FF6F00}}
  .sim-periods button.active{{background:#FF6F00;color:#0D1117;border-color:#FF6F00;font-weight:600}}
  .sim-layout{{display:grid;grid-template-columns:360px 1fr;gap:20px;align-items:start}}
  .sim-controls{{background:#0D1117;border:1px solid #2D3748;border-radius:10px;padding:14px 16px;
                 max-height:540px;overflow-y:auto}}
  .sim-buttons{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}}
  .sim-buttons button{{flex:1 1 auto;background:#21262D;color:#E6EDF3;border:1px solid #2D3748;
                       border-radius:6px;padding:6px 8px;font-size:.72rem;cursor:pointer;transition:.15s}}
  .sim-buttons button:hover{{border-color:#FF6F00;color:#FF6F00}}
  .sim-row{{display:grid;grid-template-columns:1fr 110px 52px;align-items:center;gap:10px;
            padding:6px 0;border-top:1px solid #21262D}}
  .sim-name{{font-size:.8rem;color:#E6EDF3}}
  .sim-weight{{font-size:.8rem;color:#FF6F00;text-align:right;font-variant-numeric:tabular-nums}}
  .sim-slider{{-webkit-appearance:none;appearance:none;height:5px;border-radius:3px;
               background:#2D3748;outline:none;cursor:pointer}}
  .sim-slider::-webkit-slider-thumb{{-webkit-appearance:none;appearance:none;width:15px;height:15px;
               border-radius:50%;background:#FF6F00;cursor:pointer;border:2px solid #0D1117}}
  .sim-slider::-moz-range-thumb{{width:15px;height:15px;border-radius:50%;background:#FF6F00;
               cursor:pointer;border:2px solid #0D1117}}
  .sim-chartwrap{{min-width:0}}
  .section-intro{{max-width:900px;margin:0 0 12px;color:#8B949E;font-size:.85rem;line-height:1.55}}
  .section-intro h2{{color:#E6EDF3;font-size:1rem;text-transform:none;letter-spacing:0;
                     margin-bottom:6px;font-weight:600}}
  .glossary{{margin:40px 0 8px;scroll-margin-top:64px}}
  .glossary h2{{font-size:1rem;color:#8B949E;margin-bottom:14px;text-transform:uppercase;letter-spacing:.8px}}
  .glossary dl{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px 24px}}
  .glossary dt{{color:#FF6F00;font-weight:600;font-size:.85rem;margin-bottom:4px}}
  .glossary dd{{color:#8B949E;font-size:.8rem;line-height:1.5}}
  #back-to-top{{position:fixed;right:24px;bottom:24px;width:44px;height:44px;border-radius:50%;
                background:#FF6F00;color:#0D1117;border:none;font-size:1.1rem;cursor:pointer;
                display:none;align-items:center;justify-content:center;box-shadow:0 4px 12px rgba(0,0,0,.4);z-index:60}}
  #back-to-top.show{{display:flex}}
  @media (max-width:900px){{ .sim-layout{{grid-template-columns:1fr}} }}
  @media (max-width:640px){{
    header{{padding:24px 20px 20px}}
    .container{{padding:0 16px}}
    .toc{{padding:8px 16px}}
    table{{display:block;overflow-x:auto}}
  }}
  @media print{{ .toc,#back-to-top{{display:none}} }}
</style>
</head>
<body>
<header>
  <div class="container">
    <h1>Diversificação Setorial <span>B3</span> — Backtest Unificado</h1>
    <p>Gerado em {datetime.today().strftime('%d/%m/%Y %H:%M')} &nbsp;|&nbsp;
       Out-of-sample: {oos_start:%d/%m/%Y} → {oos_end:%d/%m/%Y} &nbsp;|&nbsp;
       Livre de risco: {RISK_FREE_RATE:.2%} a.a. &nbsp;|&nbsp; Benchmark: {BENCHMARK_NAME}</p>
  </div>
</header>
<div class="container">
{nav_toc}
{summary_html}
  {summary_cards}
  {alloc_html}
  {simulator_html}
  {sections}
  {quotes_html}
</div>
{glossary_html}
<footer><div class="container">Análise de Diversificação Setorial · Dados via Yahoo Finance · {datetime.today().year}</div></footer>
{BACK_TO_TOP}
</body>
</html>"""


# ═════════════════════════════════════════════════════════════════════════════
#  10. EXIBIÇÃO — janela nativa (pywebview) com fallback para navegador
# ═════════════════════════════════════════════════════════════════════════════
def show_dashboard(html, out_path):
    """Salva o HTML em disco (sempre) e o exibe: janela nativa via pywebview quando
    disponível e USE_NATIVE_WINDOW=True; caso contrário, abre no navegador.

    A janela nativa aponta para o ARQUIVO em disco (url=file://...) em vez de injetar
    a string HTML: o NavigateToString do WebView2 tem teto de ~2 MB e o dashboard
    (com os dados dos gráficos) passa disso."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    abs_path = os.path.abspath(out_path)
    file_url = f"file:///{abs_path.replace(os.sep, '/')}"
    print(f"\n  Dashboard salvo em: {abs_path}")

    if USE_NATIVE_WINDOW:
        try:
            import webview
            print("  Abrindo janela nativa (pywebview)...")
            webview.create_window("Diversificação Setorial B3 — Backtest Unificado",
                                  url=file_url, width=1480, height=920)
            webview.start()
            return
        except ImportError:
            print("  ⚠ pywebview não instalado — abrindo no navegador. "
                  "(pip install pywebview para janela nativa.)")
        except Exception as e:
            print(f"  ⚠ não consegui abrir a janela nativa ({e}) — abrindo no navegador.")

    import webbrowser
    webbrowser.open(file_url)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Diversificação Setorial da B3 — Backtest Unificado")
    print("=" * 60)

    print("\n[1/7] Baixando preços (Yahoo Finance)...")
    all_stocks = sorted({t for lst in SECTORS.values() for t in lst})
    prices = download_prices(SECTORS, BENCHMARK, YEARS_HISTORY)
    got = [c[:-3] for c in prices.columns if c.endswith(".SA")]
    missing = sorted(set(all_stocks) - set(got))
    print(f"  {len(prices)} pregões | {len(got)}/{len(all_stocks)} ações carregadas.")
    if missing:
        print(f"  ! sem dados: {', '.join(missing)}")

    print("[2/7] Buscando market caps...")
    caps = fetch_market_caps(got)
    n_ok = sum(1 for v in caps.values() if v and not np.isnan(v))
    print(f"  market cap obtido para {n_ok}/{len(got)} ações.")

    print("[3/7] Construindo índices setoriais sintéticos...")
    sector_prices, sector_rets, composition = build_sector_indices(prices, SECTORS, caps)
    bench_rets = prices[BENCHMARK].pct_change().dropna()
    print(f"  {len(sector_rets.columns)} setores construídos.")

    print("[4/7] Rodando fronteira (in-sample) e walk-forward (out-of-sample)...")
    front = run_frontier(sector_rets)
    oos_series, last_weights, rebal_dates = run_walk_forward(sector_rets, bench_rets)
    if not oos_series:
        print("  ! Dados insuficientes para o backtest. Reduza LOOKBACK_YEARS.")
        return

    print("[5/7] Calculando métricas...")
    metrics_df = build_metrics(oos_series)
    metrics_df = metrics_df.sort_values("Sharpe", ascending=False).reset_index(drop=True)
    strat_only = metrics_df[metrics_df["Estratégia"] != BENCHMARK_NAME]
    best = {
        "sharpe": strat_only.loc[strat_only["Sharpe"].idxmax()],
        "ret":    strat_only.loc[strat_only["Retorno Anualiz."].idxmax()],
        "vol":    strat_only.loc[strat_only["Volatilidade Anual"].idxmin()],
    }
    summary_rows = build_period_summary(oos_series)

    print("[6/7] Gerando gráficos e seções...")
    figs = {
        "equity":         fig_equity(oos_series),
        "bar_returns":    fig_bar_returns(metrics_df),
        "metrics_table":  fig_metrics_table(metrics_df),
        "frontier":       fig_frontier(front),
        "allocation":     fig_allocation(last_weights),
        "drawdown":       fig_drawdown(oos_series),
        "rolling_sharpe": fig_rolling_sharpe(oos_series),
        "correlation":    fig_correlation(sector_rets),
        "composition":    fig_composition(composition),
    }
    first_key = next(iter(figs))
    figs_html = {k: v.to_html(full_html=False,
                              include_plotlyjs="cdn" if k == first_key else False,
                              config={"responsive": True, "displayModeBar": True})
                 for k, v in figs.items()}

    summary_html   = build_summary_section(summary_rows, best, last_weights)
    alloc_html     = build_allocation_table(last_weights)
    simulator_html = build_simulator(sector_rets, bench_rets, last_weights)

    print("  buscando cotações (preço atual + 52 semanas)...")
    try:
        got_stocks = [c[:-3] for c in prices.columns if c.endswith(".SA")]
        native = fetch_native_prices(got_stocks)
        quotes_html = build_quotes_section(native, composition)
    except Exception as e:
        print(f"  ⚠ não consegui montar a tabela de cotações: {e}")
        quotes_html = ""

    print("[7/7] Montando dashboard...")
    oos_start = min(v.index[0] for v in oos_series.values())
    oos_end = max(v.index[-1] for v in oos_series.values())
    html = build_html(figs_html, metrics_df, best, len(sector_rets.columns),
                      oos_start, oos_end, summary_html, alloc_html, simulator_html,
                      quotes_html=quotes_html)
    out_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.html")

    print("\n  Ranking por Sharpe (out-of-sample):")
    for _, r in metrics_df.iterrows():
        print(f"    {r['Estratégia']:<16} Sharpe {r['Sharpe']:5.2f} | "
              f"Ret {r['Retorno Anualiz.']:+6.1%} | Vol {r['Volatilidade Anual']:5.1%}")

    show_dashboard(html, out_path)


if __name__ == "__main__":
    main()
