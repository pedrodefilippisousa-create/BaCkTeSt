# Diversificação Setorial B3 — Backtest Unificado

Backtest de **diversificação entre setores da B3** que gera um **relatório HTML
autocontido** (gráficos interativos via [Plotly](https://plotly.com/python/),
tema escuro). Compara quatro estratégias de alocação num backtest
**walk-forward out-of-sample**, medindo o desempenho contra o **Ibovespa**.
Dados 100% via Yahoo Finance — apenas ativos brasileiros.

Cada setor é um índice sintético = cesta das ações líderes do setor, ponderada
pelo valor de mercado (market cap) atual.

## Estratégias comparadas (out-of-sample)

- **Markowitz — Máximo Sharpe**
- **Markowitz — Mínima Variância**
- **Equal-Weight (1/N)**
- **Risk Parity** (paridade de risco)
- **Ibovespa** (benchmark)

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `Backtest.py` | Script gerador: baixa os dados, roda o backtest e produz o HTML |
| `Atualizar e abrir relatorio.command` | Atalho para Mac: atualiza os dados e abre o relatório |
| `COMO_RODAR.md` | Passo a passo para rodar no Mac |

## O que o relatório mostra

- **Resumo executivo** com tabela de retorno por período e por estratégia
- **Alocação recomendada** por estratégia (última janela)
- **🎚️ Simulador de carteira setorial** interativo (sliders por setor, recálculo ao vivo)
- **Curvas de equity** out-of-sample e **retorno por estratégia**
- **Tabela de métricas** (retorno, volatilidade, Sharpe, Sortino, drawdown...)
- **Fronteira eficiente**, **alocação**, **drawdown**, **rolling Sharpe**
- **Matriz de correlação** entre setores e **composição** de cada setor
- **💰 Cotações**: preço atual e máxima/mínima de 52 semanas de cada ação (em R$)
- Menu de navegação, glossário e layout responsivo

## Requisitos

- Python 3.9+
- Conexão com a internet (os dados são baixados do Yahoo Finance)

## Instalação

```bash
pip install -r requirements.txt
```

## Como rodar

```bash
python Backtest.py
```

O script baixa os dados, roda o backtest e salva `setores_b3_dashboard.html`
na pasta atual, abrindo-o automaticamente no navegador. No Mac, você também pode
dar dois cliques em `Atualizar e abrir relatorio.command`.

## Configuração

Toda a configuração fica no topo do `Backtest.py` (bloco `CONFIGURAÇÃO`) — não é
preciso mexer no resto do código:

```python
SECTORS = {
    "Financeiro":       ["ITUB4", "BBDC4", "BBAS3", "B3SA3", ...],
    "Petróleo e Gás":   ["PETR4", "PETR3", "PRIO3", ...],
    # ... adicione/remova setores e ações livremente (tickers B3, sem .SA)
}

BENCHMARK        = "^BVSP"     # Ibovespa
RISK_FREE_RATE   = 0.1425      # taxa livre de risco a.a. (Selic/CDI aprox.)
YEARS_HISTORY    = 10          # histórico total baixado
LOOKBACK_YEARS   = 3           # janela de estimação no walk-forward
REBALANCE_MONTHS = 1           # rebalanceia a cada N meses
MAX_WEIGHT       = 0.30        # teto por setor na otimização
N_PORTFOLIOS     = 8_000       # nuvem Monte Carlo p/ fronteira eficiente
```

Notas:

- Os tickers são da **B3 sem o sufixo `.SA`** (ex.: `PETR4`, `VALE3`) — o script
  adiciona o `.SA` automaticamente.
- Ações sem dados no Yahoo são ignoradas com um aviso.
- `MAX_WEIGHT` evita que a otimização concentre tudo em um único setor.

## Aviso

Este projeto é apenas para fins educacionais e de estudo. As estratégias são
ajustadas com dados históricos e **não garantem desempenho futuro** (risco de
*overfitting*). Nada aqui constitui recomendação de investimento.
