# Portfolio Backtest — Dashboard HTML Interativo

Backtest e otimização de carteira de investimentos que gera um **relatório HTML
autocontido** (com gráficos interativos via [Plotly](https://plotly.com/python/)).
Suporta ativos brasileiros (`.SA`) e americanos na mesma carteira, convertendo
tudo para BRL, e compara a alocação inicial com uma carteira otimizada por
**Markowitz** contra dois benchmarks de mercado (IBOV e S&P 500).

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `Backtest.py` | Script gerador: baixa os dados, roda a otimização e produz o HTML |
| `portfolio_Carteira_Mista_BR-US.html` | Exemplo de relatório gerado |

## O que o relatório mostra

- **Resumo executivo** com tabela de retorno por período (carteira inicial vs. otimizada vs. IBOV vs. SPY)
- **Alocação ótima** (Markowitz — Máximo Sharpe): pesos iniciais vs. ótimos
- **Fronteira eficiente** de Markowitz (simulação Monte Carlo)
- **Curvas de equity** por período (6M, 1A, 3A, 5A, 10A)
- **Retorno total** por período e **performance individual** dos ativos
- **Drawdown**, **Rolling Sharpe** (252 dias) e **matriz de correlação**
- **Tabela de métricas**: retorno, volatilidade, Sharpe, Sortino, VaR/CVaR, Calmar, Max Drawdown
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

O script baixa os dados, roda a otimização e salva um arquivo
`portfolio_<NOME_DA_CARTEIRA>.html` na pasta atual, abrindo-o automaticamente
no navegador.

## Configuração

Toda a configuração fica no topo do `Backtest.py` (bloco `CONFIGURAÇÃO`) — não é
preciso mexer no resto do código:

```python
PORTFOLIO = {
    # Brasil (preços já em BRL) — sufixo .SA
    "ITUB4.SA": 0.10,
    "PETR4.SA": 0.10,
    # EUA (convertidos para BRL via USDBRL=X) — ticker puro
    "AAPL": 0.10,
    "MSFT": 0.10,
    # ...
}

PORTFOLIO_NAME = "Carteira Mista BR-US"
BENCHMARKS     = ["^BVSP", "SPY"]   # IBOV + SPY
BASE_CURRENCY  = "BRL"              # moeda base do relatório
FX_TICKER      = "USDBRL=X"         # taxa USD -> BRL
RISK_FREE_RATE = 0.1075             # taxa livre de risco a.a. (ex.: Selic)
N_PORTFOLIOS   = 6_000             # nº de simulações Monte Carlo
OUTPUT_DIR     = "."               # pasta de saída do HTML
```

Notas:

- Os **pesos não precisam somar 1** — o script normaliza automaticamente.
- Tickers **BR** usam o sufixo `.SA` (ex.: `PETR4.SA`); tickers **US** são o
  ticker puro (ex.: `AAPL`). A classe B da Berkshire no Yahoo é `BRK-B`.
- Ativos sem dados disponíveis são ignorados com um aviso.

## Aviso

Este projeto é apenas para fins educacionais e de estudo. A otimização de
Markowitz é ajustada com dados históricos e **não garante desempenho futuro**
(risco de *overfitting*). Nada aqui constitui recomendação de investimento.
