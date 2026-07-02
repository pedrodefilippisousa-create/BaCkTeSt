# Como rodar o Backtest no seu computador (Mac)

Guia passo a passo para gerar o relatório com dados atualizados no seu próprio
Mac. A instalação (passos 1 e 5) é feita **uma vez só**; depois, rodar leva
poucos segundos.

---

## Passo 1 — Instalar o Python (só uma vez)

1. Abra: https://www.python.org/downloads/
2. Clique no botão amarelo **"Download Python 3.x.x"**.
3. Abra o arquivo `.pkg` baixado e clique em **Continuar → Concordar → Instalar**
   (vai pedir a senha do Mac).
4. Ao terminar, pode fechar.

---

## Passo 2 — Baixar o projeto

1. Vá em https://github.com/pedrodefilippisousa-create/BaCkTeSt
2. Botão verde **"< > Code" → "Download ZIP"**.
3. Na pasta **Downloads**, dê dois cliques no ZIP para extrair. Vai criar a pasta
   `BaCkTeSt-main`.

> Dica: quando eu fizer novas alterações, basta baixar o ZIP de novo (ou, se
> você aprender o `git pull` depois, atualizar sem baixar tudo).

---

## Passo 3 — Abrir o Terminal

- Aperte **⌘ (Command) + barra de espaço**, digite **Terminal** e aperte Enter.

---

## Passo 4 — Entrar na pasta do projeto

Digite `cd ` (com um espaço depois), **sem apertar Enter**:

```
cd 
```

Agora **arraste a pasta `BaCkTeSt-main`** do Finder para dentro da janela do
Terminal e solte — o caminho é preenchido sozinho. **Aí sim aperte Enter.**

---

## Passo 5 — Instalar as bibliotecas (só uma vez)

```
python3 -m pip install -r requirements.txt
```

Espere terminar (baixa alguns pacotes, ~1–2 min).

> Se aparecer o erro **"externally-managed-environment"**, use esta versão com
> ambiente virtual:
>
> ```
> python3 -m venv .venv
> source .venv/bin/activate
> pip install -r requirements.txt
> ```
>
> Nesse caso, nas próximas vezes rode `source .venv/bin/activate` antes do
> `python3 Backtest.py`.

---

## Passo 6 — Rodar

```
python3 Backtest.py
```

O script baixa os dados do Yahoo Finance, roda a otimização de Markowitz e
**abre o relatório automaticamente no navegador**. Ele também salva o arquivo
`portfolio_Carteira_Mista_BR-US.html` na mesma pasta.

---

## Nas próximas vezes

Só precisa dos passos **3, 4 e 6** (abrir Terminal, entrar na pasta, rodar). A
instalação (1 e 5) já estará feita.

## Para mudar a carteira

Abra o `Backtest.py` num editor de texto, edite o bloco `PORTFOLIO` no início do
arquivo (tickers e pesos), salve e rode de novo. Lembrando:

- Ações BR usam o sufixo `.SA` (ex.: `PETR4.SA`); ações US são o ticker puro
  (ex.: `AAPL`).
- Os pesos não precisam somar 1 — o script normaliza sozinho.

---

## Se algo der errado

Copie a mensagem que apareceu no Terminal e me mande — a gente resolve.
