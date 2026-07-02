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
                      **THEME, height=600,
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
        **THEME, height=350, legend=dict(bgcolor="#161B22", bordercolor="#2D3748"),
        xaxis=dict(gridcolor=COLORS["grid"]), yaxis=dict(gridcolor=COLORS["grid"]))
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
    corr = prices[tickers].pct_change().dropna().corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=tickers, y=tickers,
        colorscale=[[0,"#EF5350"],[0.5,"#161B22"],[1,"#26A69A"]],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate="%{y} × %{x}<br>Correlação: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(title="Matriz de Correlação", **THEME, height=460,
        xaxis=dict(tickfont=dict(color="#E6EDF3")),
        yaxis=dict(tickfont=dict(color="#E6EDF3")))
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


def fig_individual_assets(prices, tickers):
    p = prices.loc[datetime.today() - PERIODS["1A"]:]
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Bold
    for i, t in enumerate(tickers):
        if t not in p.columns:
            continue
        cum = (cumulative(p[t].pct_change().dropna()) - 1) * 100
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines", name=t,
            line=dict(color=palette[i % len(palette)], width=1.8),
            hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1f}%<extra>" + t + "</extra>"))
    fig.add_hline(y=0, line_color=COLORS["grid"])
    fig.update_layout(title=f"Performance Individual dos Ativos — Últimos 12 Meses (base {BASE_CURRENCY})",
        yaxis_title="Retorno (%)", **THEME, height=420,
        legend=dict(bgcolor="#161B22", bordercolor="#2D3748"),
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], ticksuffix="%"))
    return fig


def build_html(figures_html, portfolio_name, tickers, w_ini, w_opt, mkt, metrics_df):
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
        f'<section class="chart-section"><div id="fig-{k}" class="chart-wrap">{v}</div></section>'
        for k, v in figures_html.items()
    )
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
  {summary_cards}
  <div class="alloc-section">
    <h2>Alocação Ótima — Markowitz (Max Sharpe)</h2>
    <table>
      <thead><tr><th>Ativo</th><th>Peso Inicial</th><th>Peso Ótimo</th><th>Variação</th></tr></thead>
      <tbody>{alloc_rows}</tbody>
    </table>
  </div>
  {sections}
</div>
<footer><div class="container">Portfolio Analysis Dashboard · Dados via Yahoo Finance · {datetime.today().year}</div></footer>
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
    metrics_df = build_metrics(prices, w_ini, w_opt, tickers)

    print("[4/5] Gerando gráficos...")
    figs = {
        "frontier":       fig_frontier(mkt, prices, w_ini, w_opt, tickers),
        "allocation":     fig_allocation_pie(tickers, w_ini, w_opt),
        "equity":         fig_equity_periods(prices, w_ini, w_opt, tickers),
        "bar_returns":    fig_bar_returns(metrics_df),
        "assets":         fig_individual_assets(prices, tickers),
        "drawdown":       fig_drawdown(prices, w_ini, w_opt, tickers),
        "rolling_sharpe": fig_rolling_sharpe(prices, w_ini, w_opt, tickers),
        "correlation":    fig_correlation(prices, tickers),
        "metrics_table":  fig_metrics_table(metrics_df),
    }
    figs_html = {k: v.to_html(full_html=False,
                               include_plotlyjs="cdn" if k == list(figs.keys())[0] else False,
                               config={"responsive": True, "displayModeBar": True})
                 for k, v in figs.items()}

    print("[5/5] Montando HTML...")
    html = build_html(figs_html, PORTFOLIO_NAME, tickers, w_ini, w_opt, mkt, metrics_df)
    safe_name = PORTFOLIO_NAME.replace(" ", "_").replace("/", "-")
    out_path  = os.path.join(OUTPUT_DIR, f"portfolio_{safe_name}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  Dashboard salvo em: {os.path.abspath(out_path)}")
    import webbrowser
    webbrowser.open(f"file:///{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
