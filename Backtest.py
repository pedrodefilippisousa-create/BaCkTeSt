"""
Portfolio Backtesting — Dashboard HTML Interativo (Plotly)
Suporta ativos BR (.SA) + US na mesma carteira, convertendo tudo pra BRL.
Gera um arquivo .html com todos os gráficos, métricas e dois benchmarks.
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
#  CONFIGURAÇÃO — edite tudo aqui
#
#  PORTFOLIO: dict {ticker: peso}. Adicione/remova linhas livremente.
#    • Tickers BR usam sufixo .SA           (ex: PETR4.SA, ITUB4.SA)
#    • Tickers US são o ticker puro         (ex: AAPL, MSFT)
#    • BRK.B no Yahoo é "BRK-B"
#    • Pesos NÃO precisam somar 1 — o script normaliza automaticamente
# ─────────────────────────────────────────────────────────────────────────────
PORTFOLIO = {
    # ── Brasil (preços já em BRL) ───────────────────────────────────
    "ITUB4.SA": 0.10,
    "PETR4.SA": 0.10,
    "CXSE3.SA": 0.05,
    "VALE3.SA": 0.10,
    # ── EUA (convertidos pra BRL via USD/BRL) ──────────────────────
    "AAPL":  0.10,
    "MSFT":  0.10,
    "AMZN":  0.08,
    "CAT":   0.07,
    "KMB":   0.05,
    "ASML":  0.10,
    "BRK-B": 0.10,
    "JPM":   0.05,
}

PORTFOLIO_NAME  = "Carteira Mista BR-US"
BENCHMARKS      = ["^BVSP", "SPY"]      # IBOV + SPY (ambos convertidos pra BRL)
BASE_CURRENCY   = "BRL"                 # moeda base do relatório
FX_TICKER       = "USDBRL=X"            # taxa USD → BRL (usada pra converter ativos US)
RISK_FREE_RATE  = 0.1075                # Selic anual
N_PORTFOLIOS    = 6_000                 # simulações Monte Carlo pra fronteira
OUTPUT_DIR      = "."                   # pasta onde salvar o HTML
# ─────────────────────────────────────────────────────────────────────────────

PERIODS = {
    "6M":  timedelta(days=183),
    "1A":  timedelta(days=365),
    "3A":  timedelta(days=1095),
    "5A":  timedelta(days=1825),
    "10A": timedelta(days=3650),
}

COLORS = {
    "inicial":    "#2196F3",
    "markowitz":  "#FF6F00",
    "bench_ibov": "#AB47BC",
    "bench_spy":  "#FFA726",
    "positive":   "#26A69A",
    "negative":   "#EF5350",
    "grid":       "#2D3748",
}

BENCH_STYLE = {
    "^BVSP": {"name": "IBOV", "color": COLORS["bench_ibov"]},
    "SPY":   {"name": "SPY",  "color": COLORS["bench_spy"]},
}

THEME = dict(
    paper_bgcolor="#0D1117",
    plot_bgcolor="#161B22",
    font_color="#E6EDF3",
    font_family="Inter, Arial, sans-serif",
    font_size=13,
    title_font_size=18,
)


def is_brl_native(ticker):
    """True se o ticker já é cotado em BRL (não precisa conversão FX)."""
    return ticker.endswith(".SA") or ticker == "^BVSP"


def download_prices(tickers, benchmarks, days=3660):
    """Baixa preços e converte ativos em USD pra BRL usando USDBRL=X."""
    all_tickers = list(dict.fromkeys(tickers + benchmarks))
    needs_fx = any(not is_brl_native(t) for t in all_tickers)
    fetch_list = all_tickers + ([FX_TICKER] if needs_fx else [])

    start = datetime.today() - timedelta(days=days)
    data = yf.download(fetch_list, start=start, progress=False, auto_adjust=True)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data.to_frame()
    close = close.dropna(how="all")

    if needs_fx:
        if FX_TICKER not in close.columns:
            raise RuntimeError(f"Não consegui baixar {FX_TICKER} pra conversão cambial.")
        fx = close[FX_TICKER].ffill().bfill()
        for t in all_tickers:
            if not is_brl_native(t) and t in close.columns:
                close[t] = close[t] * fx
        close = close.drop(columns=[FX_TICKER])

    return close.dropna(how="all")


def portfolio_returns(prices, weights, tickers):
    rets = prices[tickers].pct_change().dropna()
    return (rets * np.array(weights)).sum(axis=1)


def cumulative(returns):
    return (1 + returns).cumprod()


def annualized_stats(returns):
    n = len(returns)
    total = cumulative(returns).iloc[-1] - 1
    ann_r = (1 + total) ** (252 / n) - 1
    ann_v = returns.std() * np.sqrt(252)
    sharpe = (ann_r - RISK_FREE_RATE) / ann_v if ann_v else np.nan
    return ann_r, ann_v, sharpe


def max_drawdown(cum):
    return ((cum - cum.cummax()) / cum.cummax()).min()


def sortino(returns):
    mu = returns.mean() * 252
    downside_std = returns[returns < 0].std() * np.sqrt(252)
    return (mu - RISK_FREE_RATE) / downside_std if downside_std else np.nan


def var_cvar(returns, confidence=0.95):
    s = np.sort(returns.values)
    idx = int((1 - confidence) * len(s))
    return -s[idx], -s[:idx].mean()


def port_stats(w, mu, cov):
    r = np.dot(w, mu) * 252
    v = np.sqrt(w @ cov @ w) * np.sqrt(252)
    s = (r - RISK_FREE_RATE) / v
    return r, v, s


def run_markowitz(prices, tickers):
    rets = prices[tickers].pct_change().dropna()
    mu   = rets.mean()
    cov  = rets.cov()
    n    = len(tickers)
    bounds      = [(0, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    mc_r, mc_v, mc_s, mc_w = [], [], [], []
    for _ in range(N_PORTFOLIOS):
        w = np.random.dirichlet(np.ones(n))
        r, v, s = port_stats(w, mu, cov)
        mc_r.append(r); mc_v.append(v); mc_s.append(s); mc_w.append(w)

    res_sh = minimize(lambda w: -port_stats(w, mu, cov)[2],
                      np.ones(n)/n, bounds=bounds,
                      constraints=constraints, method="SLSQP")
    res_mv = minimize(lambda w: port_stats(w, mu, cov)[1],
                      np.ones(n)/n, bounds=bounds,
                      constraints=constraints, method="SLSQP")

    return {
        "mu": mu, "cov": cov,
        "mc":        {"ret": np.array(mc_r), "vol": np.array(mc_v),
                      "sharpe": np.array(mc_s), "weights": mc_w},
        "max_sharpe": {"weights": res_sh.x, "stats": port_stats(res_sh.x, mu, cov)},
        "min_vol":    {"weights": res_mv.x, "stats": port_stats(res_mv.x, mu, cov)},
    }


def build_metrics(prices, w_ini, w_opt, tickers):
    rows = []
    for label, delta in PERIODS.items():
        p = prices.loc[datetime.today() - delta:]
        if len(p) < 20:
            continue
        for name, weights in [("Inicial", w_ini), ("Markowitz", w_opt)]:
            pr   = portfolio_returns(p, weights, tickers)
            cum  = cumulative(pr)
            tot  = cum.iloc[-1] - 1
            a_r, a_v, sh = annualized_stats(pr)
            mdd  = max_drawdown(cum)
            so   = sortino(pr)
            var, cvar = var_cvar(pr)
            calmar = a_r / abs(mdd) if mdd else np.nan
            rows.append({
                "Período": label, "Carteira": name,
                "Retorno Total":      tot,
                "Retorno Anualiz.":   a_r,
                "Volatilidade Anual": a_v,
                "Sharpe":     sh,
                "Sortino":    so,
                "Max Drawdown": mdd,
                "VaR 95%":    var,
                "CVaR 95%":   cvar,
                "Calmar":     calmar,
            })
    return pd.DataFrame(rows)


def fig_frontier(mkt, prices, w_ini, w_opt, tickers):
    mc = mkt["mc"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mc["vol"] * 100, y=mc["ret"] * 100,
        mode="markers",
        marker=dict(size=4, color=mc["sharpe"], colorscale="Plasma",
                    showscale=True, opacity=0.6,
                    colorbar=dict(
                        title=dict(text="Sharpe", font=dict(color="#E6EDF3")),
                        tickfont=dict(color="#E6EDF3"))),
        text=[f"Sharpe: {s:.2f}<br>" + "<br>".join(
              f"{t}: {w*100:.1f}%" for t, w in zip(tickers, ws))
              for s, ws in zip(mc["sharpe"], mc["weights"])],
        hoverinfo="text", name="Portfólios Simulados",
    ))
    ms_r, ms_v = mkt["max_sharpe"]["stats"][0], mkt["max_sharpe"]["stats"][1]
    mv_r, mv_v = mkt["min_vol"]["stats"][0],    mkt["min_vol"]["stats"][1]
    rets = prices[tickers].pct_change().dropna()
    ini_r, ini_v, ini_s = port_stats(w_ini, rets.mean(), rets.cov())
    for name, r, v, color, symbol, extra in [
        ("Max Sharpe",   ms_r, ms_v, COLORS["markowitz"], "star",       mkt["max_sharpe"]),
        ("Min Vol",      mv_r, mv_v, COLORS["positive"],  "triangle-up", mkt["min_vol"]),
        ("Ini",          ini_r, ini_v, COLORS["inicial"], "diamond",    None),
    ]:
        w_list = extra["weights"] if extra else w_ini
        tip = f"{name}<br>Retorno: {r*100:.1f}%<br>Vol: {v*100:.1f}%<br>Sharpe: {(r-RISK_FREE_RATE)/v:.2f}<br>"
        tip += "<br>".join(f"{t}: {w*100:.1f}%" for t, w in zip(tickers, w_list))
        fig.add_trace(go.Scatter(
            x=[v*100], y=[r*100], mode="markers",
            marker=dict(size=16, color=color, symbol=symbol,
                        line=dict(color="white", width=1.5)),
            name=name, hovertext=tip, hoverinfo="text",
        ))
    fig.update_layout(title="Fronteira Eficiente de Markowitz",
        xaxis_title="Volatilidade Anual (%)", yaxis_title="Retorno Anual Esperado (%)",
        **THEME, height=500, legend=dict(bgcolor="#161B22", bordercolor="#2D3748"),
        xaxis=dict(gridcolor=COLORS["grid"]), yaxis=dict(gridcolor=COLORS["grid"]))
    return fig


def fig_equity_periods(prices, w_ini, w_opt, tickers):
    fig = make_subplots(rows=2, cols=3, subplot_titles=list(PERIODS.keys()),
                        vertical_spacing=0.14, horizontal_spacing=0.08)
    positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    for (r, c), (label, delta) in zip(positions, PERIODS.items()):
        p = prices.loc[datetime.today() - delta:]
        if len(p) < 20:
            continue
        for name, weights, color in [
            ("Inicial",   w_ini, COLORS["inicial"]),
            ("Markowitz", w_opt, COLORS["markowitz"]),
        ]:
            pr  = portfolio_returns(p, weights, tickers)
            cum = (cumulative(pr) - 1) * 100
            fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines",
                name=name, line=dict(color=color, width=2),
                showlegend=(r == 1 and c == 1),
                hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1f}%<extra>" + name + "</extra>",
            ), row=r, col=c)
        for bench in BENCHMARKS:
            if bench not in p.columns:
                continue
            style = BENCH_STYLE.get(bench, {"name": bench, "color": COLORS["bench_ibov"]})
            bench_cum = (cumulative(p[bench].pct_change().dropna()) - 1) * 100
            fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, mode="lines",
                name=style["name"], line=dict(color=style["color"], width=1.5, dash="dash"),
                showlegend=(r == 1 and c == 1),
                hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1f}%<extra>" + style["name"] + "</extra>",
            ), row=r, col=c)
    fig.update_layout(title=f"Curvas de Equity por Período (base {BASE_CURRENCY})",
                      **THEME, height=600, hovermode="x unified",
                      legend=dict(bgcolor="#161B22", bordercolor="#2D3748"))
    fig.update_xaxes(gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"], ticksuffix="%")
    return fig


def _hex_to_rgba(hex_color, alpha=0.2):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def fig_drawdown(prices, w_ini, w_opt, tickers):
    p = prices.loc[datetime.today() - PERIODS["1A"]:]
    fig = go.Figure()
    for name, weights, color in [
        ("Inicial",   w_ini, COLORS["inicial"]),
        ("Markowitz", w_opt, COLORS["markowitz"]),
    ]:
        pr  = portfolio_returns(p, weights, tickers)
        cum = cumulative(pr)
        dd  = ((cum - cum.cummax()) / cum.cummax()) * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", mode="lines",
            name=name, line=dict(color=color, width=2), fillcolor=_hex_to_rgba(color, 0.2),
            hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2f}%<extra>" + name + "</extra>",
        ))
    fig.update_layout(title="Drawdown — Últimos 12 Meses", yaxis_title="Drawdown (%)",
        **THEME, height=350, hovermode="x unified",
        legend=dict(bgcolor="#161B22", bordercolor="#2D3748"),
        xaxis=dict(gridcolor=COLORS["grid"]), yaxis=dict(gridcolor=COLORS["grid"], ticksuffix="%"))
    return fig


def fig_rolling_sharpe(prices, w_ini, w_opt, tickers):
    p = prices.loc[datetime.today() - PERIODS["3A"]:]
    fig = go.Figure()
    for name, weights, color in [
        ("Inicial",   w_ini, COLORS["inicial"]),
        ("Markowitz", w_opt, COLORS["markowitz"]),
    ]:
        pr       = portfolio_returns(p, weights, tickers)
        roll_mu  = pr.rolling(252).mean() * 252
        roll_vol = pr.rolling(252).std() * np.sqrt(252)
        roll_sh  = (roll_mu - RISK_FREE_RATE) / roll_vol
        fig.add_trace(go.Scatter(x=roll_sh.index, y=roll_sh.values, mode="lines",
            name=name, line=dict(color=color, width=2),
            hovertemplate="%{x|%d/%m/%Y}<br>Sharpe: %{y:.2f}<extra>" + name + "</extra>",
        ))
    fig.add_hline(y=1, line_dash="dot", line_color="#E6EDF3",
                  annotation_text="Sharpe = 1", annotation_font_color="#E6EDF3")
    fig.add_hline(y=0, line_color=COLORS["grid"])
    fig.update_layout(title="Rolling Sharpe (252 dias) — Últimos 3 Anos",
        yaxis_title="Sharpe", **THEME, height=350,
        legend=dict(bgcolor="#161B22", bordercolor="#2D3748"),
        xaxis=dict(gridcolor=COLORS["grid"]), yaxis=dict(gridcolor=COLORS["grid"]))
    return fig


def fig_correlation(prices, tickers):
    # agrupa BR (.SA) primeiro, depois US, para os blocos aparecerem juntos
    ordered = [t for t in tickers if t.endswith(".SA")] + [t for t in tickers if not t.endswith(".SA")]
    corr = prices[ordered].pct_change().dropna().corr()
    labels = [f"{t} 🇧🇷" if t.endswith(".SA") else f"{t} 🇺🇸" for t in ordered]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale=[[0,"#EF5350"],[0.5,"#161B22"],[1,"#26A69A"]],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}", textfont=dict(size=11),
        colorbar=dict(title=dict(text="Correlação", font=dict(color="#E6EDF3")),
                      tickfont=dict(color="#E6EDF3")),
        hovertemplate="%{y} × %{x}<br>Correlação: %{z:.2f}<extra></extra>",
    ))
    # linha separando o bloco BR do bloco US
    n_br = sum(t.endswith(".SA") for t in ordered)
    if 0 < n_br < len(ordered):
        pos = n_br - 0.5
        line = dict(color="#FF6F00", width=2)
        fig.add_vline(x=pos, line=line)
        fig.add_hline(y=pos, line=line)
    fig.update_layout(title="Matriz de Correlação (🇧🇷 agrupadas, depois 🇺🇸)", **THEME, height=480,
        xaxis=dict(tickfont=dict(color="#E6EDF3"), tickangle=-45),
        yaxis=dict(tickfont=dict(color="#E6EDF3"), autorange="reversed"))
    return fig


def fig_bar_returns(metrics_df):
    periods = metrics_df["Período"].unique().tolist()
    ini_vals, opt_vals = [], []
    for p in periods:
        row_i = metrics_df.query("Período==@p and Carteira=='Inicial'")
        row_o = metrics_df.query("Período==@p and Carteira=='Markowitz'")
        ini_vals.append(row_i["Retorno Total"].values[0] * 100 if not row_i.empty else 0)
        opt_vals.append(row_o["Retorno Total"].values[0] * 100 if not row_o.empty else 0)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Inicial", x=periods, y=ini_vals,
        marker_color=COLORS["inicial"], opacity=0.85,
        text=[f"{v:+.1f}%" for v in ini_vals], textposition="outside",
        textfont=dict(color="#E6EDF3"),
        hovertemplate="%{x}<br>Inicial: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Bar(name="Markowitz", x=periods, y=opt_vals,
        marker_color=COLORS["markowitz"], opacity=0.85,
        text=[f"{v:+.1f}%" for v in opt_vals], textposition="outside",
        textfont=dict(color="#E6EDF3"),
        hovertemplate="%{x}<br>Markowitz: %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=0, line_color=COLORS["grid"])
    fig.update_layout(title="Retorno Total por Período", barmode="group",
        yaxis_title="Retorno (%)", **THEME, height=400,
        legend=dict(bgcolor="#161B22", bordercolor="#2D3748"),
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], ticksuffix="%"))
    return fig


def fig_allocation_pie(tickers, w_ini, w_opt):
    fig = make_subplots(rows=1, cols=2, specs=[[{"type":"pie"},{"type":"pie"}]],
                        subplot_titles=["Carteira Inicial", "Markowitz (Max Sharpe)"])
    palette = (px.colors.qualitative.Set3 + px.colors.qualitative.Pastel)[:len(tickers)]
    for col, weights in [(1, w_ini), (2, w_opt)]:
        fig.add_trace(go.Pie(labels=tickers, values=[round(w*100, 2) for w in weights],
            marker=dict(colors=palette, line=dict(color="#0D1117", width=2)),
            textinfo="label+percent", hole=0.35,
            hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
            showlegend=False), row=1, col=col)
    fig.update_layout(title="Alocação da Carteira", **THEME, height=400)
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
    cell_colors = []
    for c in cols:
        col_colors = []
        for i in range(len(display)):
            col_colors.append("#1C2128" if metrics_df.iloc[i]["Carteira"] == "Markowitz" else "#161B22")
        cell_colors.append(col_colors)
    fig = go.Figure(go.Table(
        header=dict(values=[f"<b>{c}</b>" for c in cols], fill_color="#21262D",
                    font=dict(color="#E6EDF3", size=11), align="center", height=32,
                    line=dict(color="#2D3748")),
        cells=dict(values=cell_vals, fill_color=cell_colors,
                   font=dict(color="#E6EDF3", size=10), align="center", height=28,
                   line=dict(color="#2D3748")),
    ))
    fig.update_layout(title="Métricas por Período", **THEME, height=420)
    return fig


def fig_individual_assets(prices, tickers, titulo=None):
    p = prices.loc[datetime.today() - PERIODS["1A"]:]
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Bold
    # ordena por retorno no período (melhor -> pior) p/ leitura mais didática
    finais = []
    for t in tickers:
        if t not in p.columns:
            continue
        cum = (cumulative(p[t].pct_change().dropna()) - 1) * 100
        finais.append((t, cum))
    finais.sort(key=lambda x: x[1].iloc[-1] if len(x[1]) else 0, reverse=True)
    for i, (t, cum) in enumerate(finais):
        fim = cum.iloc[-1] if len(cum) else 0
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines",
            name=f"{t}  ({fim:+.0f}%)",
            line=dict(color=palette[i % len(palette)], width=1.8),
            hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1f}%<extra>" + t + "</extra>"))
    fig.add_hline(y=0, line_color=COLORS["grid"], line_width=1.5)
    fig.update_layout(
        title=titulo or f"Performance Individual dos Ativos — Últimos 12 Meses (base {BASE_CURRENCY})",
        yaxis_title="Retorno acumulado (%)", **THEME, height=440,
        legend=dict(bgcolor="#161B22", bordercolor="#2D3748", font=dict(size=12)),
        hovermode="x unified",
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], ticksuffix="%", zeroline=False))
    return fig


def _group_indices(tickers):
    """Índices dos ativos BR (.SA) e US dentro da lista de tickers."""
    br = [i for i, t in enumerate(tickers) if t.endswith(".SA")]
    us = [i for i, t in enumerate(tickers) if not t.endswith(".SA")]
    return br, us


def subportfolio_returns(prices, tickers, weights, idxs):
    """Retornos diários de um sub-conjunto de ativos, com pesos renormalizados."""
    if not idxs:
        return None
    sub_t = [tickers[i] for i in idxs]
    sub_w = np.array([weights[i] for i in idxs], dtype=float)
    if sub_w.sum() <= 0:
        sub_w = np.ones(len(idxs))
    sub_w = sub_w / sub_w.sum()
    return portfolio_returns(prices, sub_w, sub_t)


def build_group_stats(prices, tickers, w_ini, period="1A"):
    """Rentabilidade/risco das ações BR, US e da carteira completa (mesmo período)."""
    delta = PERIODS.get(period, PERIODS["1A"])
    p = prices.loc[datetime.today() - delta:]
    br_idx, us_idx = _group_indices(tickers)
    groups = [
        ("🇧🇷 Ações brasileiras", br_idx),
        ("🇺🇸 Ações americanas", us_idx),
        ("🧺 Carteira completa", list(range(len(tickers)))),
    ]
    rows = []
    for name, idx in groups:
        if not idx:
            continue
        pr = subportfolio_returns(p, tickers, w_ini, idx)
        if pr is None or len(pr) < 5:
            continue
        cum = cumulative(pr)
        a_r, a_v, sh = annualized_stats(pr)
        rows.append({
            "grupo": name,
            "n": len(idx),
            "retorno_total": cum.iloc[-1] - 1,
            "retorno_anual": a_r,
            "volatilidade": a_v,
            "sharpe": sh,
            "max_drawdown": max_drawdown(cum),
        })
    return rows


PERIOD_LABELS = {"6M": "6 meses", "1A": "1 ano", "3A": "3 anos",
                 "5A": "5 anos", "10A": "10 anos"}

# Título + texto explicativo exibido acima de cada gráfico (chave = id do fig)
CHART_INTROS = {
    "frontier": ("Risco x Retorno por Ativo &amp; Fronteira Eficiente",
        "Cada ponto é uma carteira simulada (Monte Carlo); o eixo X é a volatilidade anual (risco) e o "
        "eixo Y o retorno esperado. Os marcadores destacados são as carteiras ótimas (Máximo Sharpe e "
        "Mínima Volatilidade) e a inicial. A borda superior da nuvem é a fronteira eficiente — a melhor "
        "combinação de ativos para cada nível de risco."),
    "allocation": ("Alocação da Carteira",
        "Distribuição dos pesos entre os ativos, comparando a carteira inicial com a otimizada de "
        "Máximo Sharpe — a mesma informação da tabela acima em formato visual."),
    "equity": ("Curvas de Equity por Período",
        "Retorno acumulado (%) ao longo do tempo da carteira inicial e da otimizada, comparado aos "
        "benchmarks, em cinco janelas históricas (6M, 1A, 3A, 5A, 10A)."),
    "bar_returns": ("Retorno Total por Período",
        "Mesma comparação da seção anterior, condensada em retorno total por janela — facilita ver em "
        "quais períodos a otimização levou vantagem."),
    "assets": ("Performance Individual dos Ativos — Últimos 12 Meses",
        "Retorno de cada ativo isoladamente no último ano, em BRL. Útil para identificar quais posições "
        "puxaram o resultado da carteira para cima ou para baixo."),
    "assets_br": ("🇧🇷 Ações Brasileiras — Últimos 12 Meses",
        "Retorno acumulado de cada ação brasileira (.SA) no último ano, em BRL, ordenado da melhor "
        "para a pior. Passe o mouse para ver o valor em cada data."),
    "assets_us": ("🇺🇸 Ações Americanas — Últimos 12 Meses",
        "Retorno acumulado de cada ação americana no último ano, já convertido para BRL, ordenado da "
        "melhor para a pior. A conversão cambial (USD→BRL) afeta esses números."),
    "drawdown": ("Drawdown — Últimos 12 Meses",
        "Queda percentual em relação ao topo histórico mais recente (pico-a-vale). Quanto mais negativo "
        "e prolongado, maior o risco de perda e o tempo de recuperação."),
    "rolling_sharpe": ("Rolling Sharpe (252 dias) — Últimos 3 Anos",
        "Sharpe Ratio calculado em janelas móveis de 1 ano, mostrando se a relação risco-retorno foi "
        "consistente ao longo do tempo ou concentrada em poucos períodos bons."),
    "correlation": ("Matriz de Correlação",
        "Correlação histórica entre os retornos dos ativos. Valores próximos de 1 indicam ativos que se "
        "movem juntos (pouca diversificação); valores baixos ou negativos ajudam a reduzir o risco total."),
    "metrics_table": ("Métricas por Período",
        "Consolidado numérico (retorno, volatilidade, Sharpe, Sortino, drawdown, VaR/CVaR, Calmar) por "
        "janela de tempo, para leitura rápida sem precisar interpretar os gráficos."),
}

# Rótulo curto de cada gráfico no menu de navegação (chave = id do fig)
NAV_LABELS = {
    "frontier": "Fronteira Eficiente", "allocation": "Alocação", "equity": "Equity",
    "bar_returns": "Retorno/Período", "assets": "Ativos",
    "assets_br": "Ativos BR", "assets_us": "Ativos US", "drawdown": "Drawdown",
    "rolling_sharpe": "Rolling Sharpe", "correlation": "Correlação", "metrics_table": "Métricas",
}

GLOSSARY = [
    ("Sharpe Ratio", "Retorno em excesso ao ativo livre de risco, dividido pela volatilidade. "
                     "Quanto maior, melhor a relação risco-retorno."),
    ("Volatilidade", "Desvio-padrão anualizado dos retornos; mede a intensidade das oscilações de "
                     "preço, usada como proxy de risco."),
    ("Drawdown", "Maior queda percentual entre um pico e o vale seguinte antes de uma nova máxima "
                 "ser atingida."),
    ("Fronteira Eficiente", "Conjunto de carteiras que oferecem o maior retorno esperado para cada "
                            "nível de risco, segundo a teoria de Markowitz."),
    ("Correlação", "Mede o quanto dois ativos se movem juntos, de -1 (opostos) a +1 (idênticos); "
                   "correlações baixas melhoram a diversificação."),
    ("Rolling Sharpe", "Sharpe Ratio recalculado em janelas móveis, para revelar se o desempenho "
                       "ajustado ao risco foi estável ao longo do tempo."),
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


def build_period_summary(prices, w_ini, w_opt, tickers):
    """Retorno total acumulado por período p/ carteira inicial, otimizada e benchmarks."""
    rows = []
    for label, delta in PERIODS.items():
        p = prices.loc[datetime.today() - delta:]
        if len(p) < 20:
            continue
        entry = {"Período": label}
        for name, weights in [("Inicial", w_ini), ("Markowitz", w_opt)]:
            pr = portfolio_returns(p, weights, tickers)
            entry[name] = cumulative(pr).iloc[-1] - 1
        for bench in BENCHMARKS:
            if bench in p.columns:
                entry[bench] = cumulative(p[bench].pct_change().dropna()).iloc[-1] - 1
            else:
                entry[bench] = None
        rows.append(entry)
    return rows


def build_nav(fig_keys, has_groups=False, has_sim=False, has_quotes=False):
    links = ['<a href="#resumo">Resumo</a>']
    if has_quotes:
        links.append('<a href="#cotacoes">💰 Cotações</a>')
    links.append('<a href="#tabela-alocacao">Alocação Ótima</a>')
    if has_groups:
        links.append('<a href="#grupos">Brasil × EUA</a>')
    if has_sim:
        links.append('<a href="#simulador">🎚️ Simulador</a>')
    links += [f'<a href="#fig-{k}">{NAV_LABELS.get(k, k)}</a>' for k in fig_keys]
    links.append('<a href="#glossario">Glossário</a>')
    return ('<nav class="toc" aria-label="Navegação do relatório">\n  '
            + "\n  ".join(links) + "\n</nav>")


def build_glossary():
    items = "".join(f"<div><dt>{term}</dt><dd>{desc}</dd></div>" for term, desc in GLOSSARY)
    return (f'<div class="glossary" id="glossario"><div class="container">'
            f'<h2>Glossário</h2><dl>{items}</dl></div></div>')


def build_summary_section(summary_rows, tickers, w_opt):
    """Resumo executivo: texto dinâmico + tabela carteira vs. benchmarks por período."""
    bench_names = [BENCH_STYLE.get(b, {"name": b})["name"] for b in BENCHMARKS]

    header = ("<tr><th>Período</th><th>Carteira Inicial</th><th>Carteira Otimizada</th>"
              + "".join(f"<th>{n}</th>" for n in bench_names) + "</tr>")
    body = ""
    for row in summary_rows:
        cols = [row.get("Inicial"), row.get("Markowitz")] + [row.get(b) for b in BENCHMARKS]
        vals = [v for v in cols if v is not None]
        best = max(vals) if vals else None
        tds = ""
        for v in cols:
            if v is not None and best is not None and abs(v - best) < 1e-12:
                tds += f'<td style="color:#26A69A"><b>{_pct(v, decimals=2)}</b></td>'
            else:
                tds += f"<td>{_pct(v, decimals=2)}</td>"
        label = PERIOD_LABELS.get(row["Período"], row["Período"])
        body += f"<tr><td>{label}</td>{tds}</tr>"

    n_br = sum(t.endswith(".SA") for t in tickers)
    n_us = len(tickers) - n_br
    opt_pairs = sorted(zip(tickers, w_opt), key=lambda x: -x[1])
    top = [(t, w) for t, w in opt_pairs if w >= 0.05][:5] or opt_pairs[:3]
    top_str = ", ".join(f"{t} ({_pct(w, signed=False)})" for t, w in top)
    zeroed = [t for t, w in zip(tickers, w_opt) if w < 0.005]
    zeroed_str = ", ".join(zeroed) if zeroed else "nenhum ativo"
    period_names = ", ".join(PERIOD_LABELS.get(r["Período"], r["Período"]) for r in summary_rows)

    caveat = ""
    if summary_rows:
        last = summary_rows[-1]
        mk = last.get("Markowitz")
        winners = [BENCH_STYLE.get(b, {"name": b})["name"] for b in BENCHMARKS
                   if last.get(b) is not None and mk is not None and last[b] > mk]
        if winners:
            verbo = "supera" if len(winners) == 1 else "superam"
            caveat = (f" No período mais longo ({PERIOD_LABELS.get(last['Período'], last['Período'])}), "
                      f"{' e '.join(winners)} {verbo} a carteira otimizada — a otimização de Markowitz "
                      f"é ajustada com dados históricos e não garante repetição futura desse desempenho "
                      f"(risco de <i>overfitting</i> nos pesos).")

    return f"""<section class="summary-section" id="resumo">
  <h2>Resumo Executivo</h2>
  <p>Este relatório compara uma alocação inicial (pesos definidos manualmente) com uma carteira
  <b>otimizada via Markowitz</b> para o mesmo conjunto de {len(tickers)} ativos ({n_br} brasileiros e
  {n_us} americanos), medindo o retorno acumulado contra os benchmarks
  <b>{' e '.join(bench_names)}</b> em janelas de {period_names}.</p>
  <p>A carteira de <b>Máximo Sharpe</b> concentra posição em {top_str}, reduzindo a zero a exposição a
  {zeroed_str}. Essa realocação eleva o retorno esperado e o Sharpe Ratio, ao custo de maior
  concentração (menor diversificação) — vale conferir a <a href="#fig-correlation">matriz de
  correlação</a> antes de aplicar os pesos na prática.</p>
  <table class="summary-table">
    <thead>{header}</thead>
    <tbody>{body}</tbody>
  </table>
  <p class="summary-note">Retornos acumulados extraídos das curvas de equity da seção "Curvas de Equity
  por Período" abaixo.{caveat}</p>
</section>"""


def _money(v, cur):
    """Formata valor monetário em pt-BR: US$ 1.234,56 / R$ 32,45."""
    s = f"{v:,.2f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return f"{cur} {s}"


def fetch_native_prices(tickers, days=400):
    """Baixa os preços de fechamento em moeda NATIVA (sem conversão FX) p/ a tabela de cotações."""
    start = datetime.today() - timedelta(days=days)
    data = yf.download(list(dict.fromkeys(tickers)), start=start,
                       progress=False, auto_adjust=False)
    if "Close" in getattr(data, "columns", []):
        close = data["Close"]
    elif isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data.to_frame()
    return close.dropna(how="all")


def compute_quotes(native_prices, tickers):
    """Preço atual e máx/mín de 52 semanas (moeda nativa) de cada ticker."""
    rows = []
    corte = datetime.today() - timedelta(days=365)
    for t in tickers:
        if t not in native_prices.columns:
            continue
        s = native_prices[t].dropna()
        if s.empty:
            continue
        janela = s.loc[corte:]
        if janela.empty:
            janela = s
        last = float(s.iloc[-1])
        hi, lo = float(janela.max()), float(janela.min())
        pos = (last - lo) / (hi - lo) if hi > lo else 0.5
        rows.append({
            "ticker": t,
            "origem": "BR" if t.endswith(".SA") else "US",
            "moeda": "R$" if t.endswith(".SA") else "US$",
            "atual": last, "min52": lo, "max52": hi,
            "pos": max(0.0, min(1.0, pos)),
            "vs_max": last / hi - 1 if hi else 0.0,
        })
    return rows


def build_quotes_section(rows):
    """Seção HTML com preço atual + faixa de 52 semanas (com barra de posição)."""
    if not rows:
        return ""
    body = ""
    for r in rows:
        posp = r["pos"] * 100
        body += f"""
      <tr>
        <td style="text-align:left"><b>{r['ticker']}</b> <span class="mini">{r['origem']}</span></td>
        <td><b>{_money(r['atual'], r['moeda'])}</b></td>
        <td>{_money(r['min52'], r['moeda'])}</td>
        <td>{_money(r['max52'], r['moeda'])}</td>
        <td class="range-cell">
          <div class="range-bar"><div class="range-dot" style="left:{posp:.0f}%"></div></div>
          <span class="range-txt">{posp:.0f}% da faixa · {_pct(r['vs_max'])} vs. máx</span>
        </td>
      </tr>"""
    return f"""<section class="summary-section" id="cotacoes">
  <h2>💰 Cotações — Preço Atual e Faixa de 52 Semanas</h2>
  <p>Preço de fechamento mais recente de cada ativo, na moeda nativa (🇧🇷 R$, 🇺🇸 US$), com a
  <b>máxima e a mínima dos últimos 12 meses</b>. A barra mostra onde o preço atual está dentro dessa
  faixa: à <span style="color:#EF5350">esquerda (vermelho)</span> = perto da mínima;
  à <span style="color:#26A69A">direita (verde)</span> = perto da máxima.</p>
  <table class="summary-table">
    <thead><tr>
      <th style="text-align:left">Ativo</th><th>Preço Atual</th><th>Mín. 52s</th><th>Máx. 52s</th>
      <th style="text-align:left">Posição na faixa (52 semanas)</th>
    </tr></thead>
    <tbody>{body}
    </tbody>
  </table>
  <p class="summary-note">Atualizado a cada execução do <code>Backtest.py</code>. "vs. máx" = variação em
  relação à máxima de 52 semanas (quanto o preço está abaixo do topo do período).</p>
</section>"""


def build_group_section(prices, tickers, w_ini, period="1A"):
    """Seção HTML comparando ações BR, US e a carteira completa (rentab./risco)."""
    rows = build_group_stats(prices, tickers, w_ini, period)
    if not rows:
        return ""
    periodo_txt = PERIOD_LABELS.get(period, period)
    best_ret = max(r["retorno_total"] for r in rows)
    body = ""
    for r in rows:
        destaque = ' style="color:#26A69A"' if abs(r["retorno_total"] - best_ret) < 1e-12 else ""
        sh_color = "#26A69A" if r["sharpe"] >= 1 else ("#FFA726" if r["sharpe"] >= 0 else "#EF5350")
        body += f"""
      <tr>
        <td style="text-align:left"><b>{r['grupo']}</b> <span class="mini">{r['n']} ativos</span></td>
        <td{destaque}><b>{_pct(r['retorno_total'])}</b></td>
        <td>{_pct(r['retorno_anual'])}</td>
        <td>{_pct(r['volatilidade'], signed=False)}</td>
        <td style="color:{sh_color}"><b>{r['sharpe']:.2f}</b></td>
        <td style="color:#EF5350">{_pct(r['max_drawdown'])}</td>
      </tr>"""
    return f"""<section class="summary-section" id="grupos">
  <h2>Brasil × Estados Unidos × Carteira Completa</h2>
  <p>Comparação da fatia <b>brasileira</b>, da fatia <b>americana</b> e da <b>carteira completa</b> nos
  últimos {periodo_txt} (pesos iniciais renormalizados dentro de cada grupo). Ajuda a ver de onde vem o
  retorno e o risco — e o efeito da diversificação entre países ao juntar as duas.</p>
  <table class="summary-table">
    <thead><tr>
      <th style="text-align:left">Grupo</th><th>Retorno Total</th><th>Retorno Anualiz.</th>
      <th>Volatilidade</th><th>Sharpe</th><th>Máx. Drawdown</th>
    </tr></thead>
    <tbody>{body}
    </tbody>
  </table>
  <p class="summary-note">Sharpe em verde = acima de 1 (boa relação risco-retorno); laranja = entre 0 e 1;
  vermelho = negativo. A "carteira completa" costuma ter volatilidade menor que a média das partes graças
  à diversificação.</p>
</section>"""


def _css_id(t):
    """ID seguro para HTML/CSS a partir de um ticker (ex.: BRK-B -> BRK_B)."""
    return "".join(c if c.isalnum() else "_" for c in t)


SIMULATOR_JS = r"""
<script>
(function(){
  const SIM = __SIM_JSON__;
  const DATEOBJ = SIM.dates.map(d => new Date(d));
  const LAST = DATEOBJ[DATEOBJ.length - 1];
  const sliders = {};
  let curPeriod = "1A";

  const cssId = t => t.replace(/[^a-zA-Z0-9]/g, "_");
  const fmtPct = (v, dec=1, signed=true) => {
    if (!isFinite(v)) return "—";
    let s = (signed && v > 0 ? "+" : "") + (v*100).toFixed(dec);
    return s.replace(".", ",") + "%";
  };
  function currentWeights(){
    const raw = SIM.tickers.map(t => sliders[t] ? +sliders[t].value : 0);
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
    SIM.tickers.forEach((t, ti) => {
      const wt = w[ti]; if (!wt) return;
      const arr = SIM.returns[t];
      for (let i=0; i<N; i++) daily[i] += wt*arr[start+i];
    });
    SIM.tickers.forEach((t, ti) => {
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
    const bcolors = {"IBOV":"#AB47BC", "SPY":"#FFA726"};
    let bi = 0;
    Object.keys(SIM.benchmarks).forEach(bn => {
      const arr = SIM.benchmarks[bn].slice(start);
      traces.push({x:dates, y:cumSeries(arr), mode:"lines", name:bn,
        line:{color: bcolors[bn] || ["#42A5F5","#66BB6A"][bi++ % 2], width:1.5, dash:"dash"},
        hovertemplate:"%{x|%d/%m/%Y}<br>%{y:.1f}%<extra>"+bn+"</extra>"});
    });
    Plotly.react("sim-chart", traces, {
      paper_bgcolor:"#0D1117", plot_bgcolor:"#161B22",
      font:{color:"#E6EDF3", family:"Inter, Arial, sans-serif"},
      margin:{t:12, r:12, b:40, l:52}, height:420, hovermode:"x unified",
      legend:{orientation:"h", y:1.12, bgcolor:"rgba(0,0,0,0)"},
      xaxis:{gridcolor:"#2D3748"}, yaxis:{gridcolor:"#2D3748", ticksuffix:"%", title:"Retorno acumulado (%)"}
    }, {responsive:true, displayModeBar:false});
  }
  function setWeights(arr){
    SIM.tickers.forEach((t,i) => { if (sliders[t]) sliders[t].value = Math.round(arr[i]*100); });
    compute();
  }
  function waitPlotly(){ if (window.Plotly) compute(); else setTimeout(waitPlotly, 120); }
  function init(){
    SIM.tickers.forEach(t => {
      sliders[t] = document.getElementById("slider-"+cssId(t));
      if (sliders[t]) sliders[t].addEventListener("input", compute);
    });
    document.querySelectorAll(".sim-periods button").forEach(btn =>
      btn.addEventListener("click", () => {
        curPeriod = btn.dataset.period;
        document.querySelectorAll(".sim-periods button").forEach(b => b.classList.remove("active"));
        btn.classList.add("active"); compute();
      }));
    const bind = (id, fn) => { const b = document.getElementById(id); if (b) b.addEventListener("click", fn); };
    bind("sim-btn-ini", () => setWeights(SIM.w_ini));
    bind("sim-btn-opt", () => setWeights(SIM.w_opt));
    bind("sim-btn-eq",  () => setWeights(SIM.tickers.map(() => 1/SIM.tickers.length)));
    bind("sim-btn-zero",() => setWeights(SIM.tickers.map(() => 0)));
    waitPlotly();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
</script>
"""


def build_simulator(prices, tickers, w_ini, w_opt):
    """Painel interativo: sliders de peso recalculando retorno/risco/Sharpe/curva no navegador."""
    avail_bench = [b for b in BENCHMARKS if b in prices.columns]
    rets = prices[tickers + avail_bench].pct_change().dropna()
    if len(rets) < 20:
        return ""
    data = {
        "dates":    [d.strftime("%Y-%m-%d") for d in rets.index],
        "returns":  {t: [round(float(x), 5) for x in rets[t].values] for t in tickers},
        "benchmarks": {BENCH_STYLE.get(b, {"name": b})["name"]:
                       [round(float(x), 5) for x in rets[b].values] for b in avail_bench},
        "tickers":  tickers,
        "w_ini":    [round(float(x), 4) for x in w_ini],
        "w_opt":    [round(float(x), 4) for x in w_opt],
        "rf":       RISK_FREE_RATE,
        "periods":  {"6M": 183, "1A": 365, "3A": 1095, "5A": 1825, "10A": 3650},
    }
    sim_json = json.dumps(data, ensure_ascii=False)

    rows = ""
    for t, wi in zip(tickers, w_ini):
        origem = "BR" if t.endswith(".SA") else "US"
        cid = _css_id(t)
        rows += f"""
        <div class="sim-row">
          <span class="sim-name">{t} <span class="mini">{origem}</span></span>
          <input type="range" min="0" max="100" step="1" value="{round(wi*100)}"
                 id="slider-{cid}" class="sim-slider" aria-label="Peso de {t}">
          <span class="sim-weight" id="w-{cid}">{wi*100:.1f}%</span>
        </div>"""

    def _pbtn(k):
        cls = ' class="active"' if k == "1A" else ''
        return f'<button data-period="{k}"{cls}>{k}</button>'
    periods_html = "".join(_pbtn(k) for k in ["6M", "1A", "3A", "5A", "10A"])

    section = f"""<section class="summary-section" id="simulador">
  <h2>🎚️ Simulador de Carteira</h2>
  <p>Arraste os controles para mudar o peso de cada ativo — retorno, risco, Sharpe e a curva abaixo
  recalculam <b>na hora</b>. Os pesos são normalizados automaticamente para somar 100%. Coloque um ativo
  em 0 para removê-lo da simulação. (Para incluir um ativo <i>novo</i>, edite o <code>PORTFOLIO</code> no
  <code>Backtest.py</code> e rode de novo.)</p>
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
        <button id="sim-btn-ini" type="button">Pesos iniciais</button>
        <button id="sim-btn-opt" type="button">Pesos ótimos</button>
        <button id="sim-btn-eq" type="button">Igualar (1/N)</button>
        <button id="sim-btn-zero" type="button">Zerar</button>
      </div>
      {rows}
    </div>
    <div class="chart-wrap sim-chartwrap"><div id="sim-chart" style="height:420px;width:100%"></div></div>
  </div>
</section>
{SIMULATOR_JS.replace("__SIM_JSON__", sim_json)}"""
    return section


def build_html(figures_html, portfolio_name, tickers, w_ini, w_opt, mkt, metrics_df, summary_rows,
               group_html="", simulator_html="", quotes_html=""):
    ms_r, ms_v, ms_s = mkt["max_sharpe"]["stats"]
    mv_r, mv_v, _    = mkt["min_vol"]["stats"]
    alloc_rows = ""
    for t, wi, wo in zip(tickers, w_ini, w_opt):
        diff = wo - wi
        arrow = "+" if diff > 0 else ""
        origem = "BR" if t.endswith(".SA") else "US"
        alloc_rows += f"""
        <tr>
            <td>{t} <span class="mini">{origem}</span></td>
            <td>{wi:.1%}</td><td><b>{wo:.1%}</b></td>
            <td style="color:{'#26A69A' if diff >= 0 else '#EF5350'}">{arrow}{diff:.1%}</td>
        </tr>"""
    summary_cards = f"""
    <div class="cards">
        <div class="card"><div class="card-label">Max Sharpe — Retorno Esp.</div>
            <div class="card-value" style="color:#FF6F00">{ms_r:+.1%}</div></div>
        <div class="card"><div class="card-label">Max Sharpe — Volatilidade</div>
            <div class="card-value">{ms_v:.1%}</div></div>
        <div class="card"><div class="card-label">Max Sharpe — Sharpe Ratio</div>
            <div class="card-value" style="color:#26A69A">{ms_s:.2f}</div></div>
        <div class="card"><div class="card-label">Min Vol — Retorno Esp.</div>
            <div class="card-value" style="color:#2196F3">{mv_r:+.1%}</div></div>
        <div class="card"><div class="card-label">Min Vol — Volatilidade</div>
            <div class="card-value">{mv_v:.1%}</div></div>
        <div class="card"><div class="card-label">Ativos na Carteira</div>
            <div class="card-value">{len(tickers)}</div></div>
    </div>"""
    sections = "".join(
        f'<section class="chart-section">'
        f'<div class="section-intro"><h2>{CHART_INTROS.get(k, ("", ""))[0]}</h2>'
        f'<p>{CHART_INTROS.get(k, ("", ""))[1]}</p></div>'
        f'<div id="fig-{k}" class="chart-wrap">{v}</div></section>'
        for k, v in figures_html.items()
    )
    nav_toc       = build_nav(list(figures_html.keys()), has_groups=bool(group_html),
                              has_sim=bool(simulator_html), has_quotes=bool(quotes_html))
    summary_html  = build_summary_section(summary_rows, tickers, w_opt)
    glossary_html = build_glossary()
    back_to_top   = BACK_TO_TOP
    bench_labels = " + ".join(BENCH_STYLE.get(b, {"name": b})["name"] for b in BENCHMARKS)
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Portfolio: {portfolio_name}</title>
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
  .chart-section{{margin:24px 0}}
  .chart-wrap{{background:#161B22;border:1px solid #2D3748;border-radius:10px;padding:8px;overflow:hidden}}
  .alloc-section{{margin:24px 0}}
  .alloc-section h2{{font-size:1rem;color:#8B949E;margin-bottom:12px;text-transform:uppercase;letter-spacing:.8px}}
  table{{width:100%;border-collapse:collapse;background:#161B22;border-radius:10px;overflow:hidden;border:1px solid #2D3748}}
  th{{background:#21262D;padding:12px 16px;text-align:center;font-size:.8rem;color:#8B949E;text-transform:uppercase;letter-spacing:.6px}}
  td{{padding:10px 16px;text-align:center;border-top:1px solid #2D3748;font-size:.9rem}}
  tr:hover td{{background:#1C2128}}
  .mini{{display:inline-block;font-size:.65rem;padding:2px 6px;border-radius:4px;background:#21262D;color:#8B949E;margin-left:6px;vertical-align:middle}}
  footer{{text-align:center;color:#3D444D;font-size:.75rem;margin-top:48px}}
  .tag{{display:inline-block;background:#21262D;border:1px solid #2D3748;border-radius:20px;padding:4px 12px;font-size:.75rem;color:#8B949E;margin:4px 2px}}
  html{{scroll-behavior:smooth}}
  .toc{{position:sticky;top:0;z-index:50;background:#161B22;border-bottom:1px solid #2D3748;
       padding:10px 32px;display:flex;gap:6px;overflow-x:auto;white-space:nowrap}}
  .toc a{{color:#8B949E;font-size:.78rem;text-decoration:none;padding:6px 12px;border-radius:16px;
         border:1px solid transparent;transition:.15s}}
  .toc a:hover{{color:#E6EDF3;border-color:#2D3748}}
  .toc a:focus-visible{{outline:2px solid #FF6F00}}
  .summary-section{{background:#161B22;border:1px solid #2D3748;border-radius:10px;
                    padding:24px 28px;margin:32px 0}}
  .summary-section h2{{font-size:1.1rem;margin-bottom:12px}}
  .summary-section p{{color:#C9D1D9;font-size:.9rem;line-height:1.6;margin-bottom:14px}}
  .summary-table{{width:100%;border-collapse:collapse;margin-top:8px}}
  .summary-table th{{background:#21262D;padding:8px 12px;font-size:.75rem;color:#8B949E;
                     text-transform:uppercase;letter-spacing:.5px;text-align:center}}
  .summary-table td{{padding:8px 12px;text-align:center;font-size:.85rem;border-top:1px solid #2D3748}}
  .summary-note{{color:#8B949E;font-size:.75rem;margin-top:10px}}
  .summary-section code{{background:#0D1117;border:1px solid #2D3748;border-radius:4px;
                         padding:1px 5px;font-size:.85em;color:#FFA726}}
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
  .sim-layout{{display:grid;grid-template-columns:340px 1fr;gap:20px;align-items:start}}
  .sim-controls{{background:#0D1117;border:1px solid #2D3748;border-radius:10px;padding:14px 16px;
                 max-height:520px;overflow-y:auto}}
  .sim-buttons{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}}
  .sim-buttons button{{flex:1 1 auto;background:#21262D;color:#E6EDF3;border:1px solid #2D3748;
                       border-radius:6px;padding:6px 8px;font-size:.72rem;cursor:pointer;transition:.15s}}
  .sim-buttons button:hover{{border-color:#FF6F00;color:#FF6F00}}
  .sim-row{{display:grid;grid-template-columns:1fr 120px 52px;align-items:center;gap:10px;
            padding:6px 0;border-top:1px solid #21262D}}
  .sim-name{{font-size:.82rem;color:#E6EDF3}}
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
  .chart-section{{scroll-margin-top:64px}}
  .glossary{{margin:40px 0 8px}}
  .glossary h2{{font-size:1rem;color:#8B949E;margin-bottom:14px;text-transform:uppercase;letter-spacing:.8px}}
  .glossary dl{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px 24px}}
  .glossary dt{{color:#FF6F00;font-weight:600;font-size:.85rem;margin-bottom:4px}}
  .glossary dd{{color:#8B949E;font-size:.8rem;line-height:1.5}}
  #back-to-top{{position:fixed;right:24px;bottom:24px;width:44px;height:44px;border-radius:50%;
                background:#FF6F00;color:#0D1117;border:none;font-size:1.1rem;cursor:pointer;
                display:none;align-items:center;justify-content:center;box-shadow:0 4px 12px rgba(0,0,0,.4);z-index:60}}
  #back-to-top.show{{display:flex}}
  @media (max-width:900px){{
    .sim-layout{{grid-template-columns:1fr}}
  }}
  @media (max-width:640px){{
    header{{padding:24px 20px 20px}}
    .container{{padding:0 16px}}
    .toc{{padding:8px 16px}}
    table{{display:block;overflow-x:auto}}
  }}
  @media print{{
    .toc,#back-to-top{{display:none}}
  }}
</style>
</head>
<body>
<header>
  <div class="container">
    <h1>Portfolio <span>{portfolio_name}</span></h1>
    <p>Gerado em {datetime.today().strftime('%d/%m/%Y %H:%M')} &nbsp;|&nbsp;
       Moeda base: {BASE_CURRENCY} &nbsp;|&nbsp;
       Risk-free: {RISK_FREE_RATE:.2%} a.a. &nbsp;|&nbsp;
       Benchmarks: {bench_labels}</p>
    <div style="margin-top:12px">{"".join(f'<span class="tag">{t}</span>' for t in tickers)}</div>
  </div>
</header>
<div class="container">
{nav_toc}
{summary_html}
  {summary_cards}
  {quotes_html}
  <div class="alloc-section" id="tabela-alocacao">
    <h2>Alocação Ótima — Markowitz (Max Sharpe)</h2>
    <table>
      <thead><tr><th>Ativo</th><th>Peso Inicial</th><th>Peso Ótimo</th><th>Variação</th></tr></thead>
      <tbody>{alloc_rows}</tbody>
    </table>
  </div>
  {group_html}
  {simulator_html}
  {sections}
</div>
{glossary_html}
<footer><div class="container">Portfolio Analysis Dashboard · Dados via Yahoo Finance · {datetime.today().year}</div></footer>
{back_to_top}
</body>
</html>"""


def main():
    if not PORTFOLIO:
        raise ValueError("PORTFOLIO está vazio — adicione ao menos um ticker.")

    raw_tickers = list(PORTFOLIO.keys())
    raw_weights = np.array(list(PORTFOLIO.values()), dtype=float)
    if (raw_weights < 0).any():
        raise ValueError("Pesos negativos não são suportados.")
    if raw_weights.sum() <= 0:
        raise ValueError("Soma dos pesos deve ser > 0.")

    total = raw_weights.sum()
    if abs(total - 1.0) > 1e-6:
        print(f"  ⚠ pesos somavam {total:.4f} — normalizando pra 1.0")
    weights_ini = raw_weights / total

    print("=" * 60)
    print("  Portfolio HTML Dashboard")
    print("=" * 60)
    print(f"  Carteira   : {PORTFOLIO_NAME}")
    print(f"  Ativos     : {len(raw_tickers)}  ({sum(t.endswith('.SA') for t in raw_tickers)} BR / {sum(not t.endswith('.SA') for t in raw_tickers)} US)")
    print(f"  Moeda base : {BASE_CURRENCY}")
    print(f"  Benchmarks : {', '.join(BENCHMARKS)}")
    print(f"  Risk-free  : {RISK_FREE_RATE:.2%} a.a.")
    print("=" * 60)

    print("\n[1/5] Baixando dados + FX...")
    prices = download_prices(raw_tickers, BENCHMARKS)

    tickers = [t for t in raw_tickers if t in prices.columns]
    dropped = [t for t in raw_tickers if t not in prices.columns]
    if dropped:
        print(f"  ⚠ ignorando (sem dados): {', '.join(dropped)}")
    keep_mask = [t in prices.columns for t in raw_tickers]
    w_ini = weights_ini[keep_mask]
    w_ini = w_ini / w_ini.sum()
    print(f"  {len(prices)} pregões carregados. {len(tickers)} ativos válidos.")

    print("[2/5] Rodando Markowitz...")
    mkt = run_markowitz(prices, tickers)
    w_opt = mkt["max_sharpe"]["weights"]

    print("[3/5] Calculando métricas...")
    metrics_df   = build_metrics(prices, w_ini, w_opt, tickers)
    summary_rows = build_period_summary(prices, w_ini, w_opt, tickers)

    print("[4/5] Gerando gráficos...")
    br_tickers = [t for t in tickers if t.endswith(".SA")]
    us_tickers = [t for t in tickers if not t.endswith(".SA")]

    figs = {
        "frontier":       fig_frontier(mkt, prices, w_ini, w_opt, tickers),
        "allocation":     fig_allocation_pie(tickers, w_ini, w_opt),
        "equity":         fig_equity_periods(prices, w_ini, w_opt, tickers),
        "bar_returns":    fig_bar_returns(metrics_df),
    }
    # Separa a performance individual por país (BR / US); se só houver um grupo,
    # mantém um único gráfico combinado.
    if br_tickers and us_tickers:
        figs["assets_br"] = fig_individual_assets(
            prices, br_tickers, "🇧🇷 Ações Brasileiras — Últimos 12 Meses (base BRL)")
        figs["assets_us"] = fig_individual_assets(
            prices, us_tickers, "🇺🇸 Ações Americanas — Últimos 12 Meses (base BRL)")
    else:
        figs["assets"] = fig_individual_assets(prices, tickers)
    figs.update({
        "drawdown":       fig_drawdown(prices, w_ini, w_opt, tickers),
        "rolling_sharpe": fig_rolling_sharpe(prices, w_ini, w_opt, tickers),
        "correlation":    fig_correlation(prices, tickers),
        "metrics_table":  fig_metrics_table(metrics_df),
    })
    group_html     = build_group_section(prices, tickers, w_ini, period="1A")
    simulator_html = build_simulator(prices, tickers, w_ini, w_opt)
    print("  buscando cotações (preço atual + 52 semanas)...")
    try:
        native = fetch_native_prices(tickers)
        quotes_html = build_quotes_section(compute_quotes(native, tickers))
    except Exception as e:
        print(f"  ⚠ não consegui montar a tabela de cotações: {e}")
        quotes_html = ""
    figs_html = {k: v.to_html(full_html=False,
                               include_plotlyjs="cdn" if k == list(figs.keys())[0] else False,
                               config={"responsive": True, "displayModeBar": True})
                 for k, v in figs.items()}

    print("[5/5] Montando HTML...")
    html = build_html(figs_html, PORTFOLIO_NAME, tickers, w_ini, w_opt, mkt, metrics_df,
                      summary_rows, group_html=group_html, simulator_html=simulator_html,
                      quotes_html=quotes_html)
    safe_name = PORTFOLIO_NAME.replace(" ", "_").replace("/", "-")
    out_path  = os.path.join(OUTPUT_DIR, f"portfolio_{safe_name}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  Dashboard salvo em: {os.path.abspath(out_path)}")
    import webbrowser
    webbrowser.open(f"file:///{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
