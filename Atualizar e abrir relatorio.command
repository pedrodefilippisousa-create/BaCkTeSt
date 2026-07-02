#!/bin/bash
# ============================================================================
#  Atalho para Mac: baixa os dados atualizados do Yahoo Finance e abre o
#  relatório. Dê DOIS CLIQUES neste arquivo.
#
#  Na primeira vez, se o Mac disser que "não pode ser aberto", clique com o
#  botão direito neste arquivo -> Abrir -> Abrir. (Ou rode uma vez no Terminal:
#  chmod +x "Atualizar e abrir relatorio.command")
# ============================================================================

# entra na pasta onde este arquivo está (funciona em qualquer lugar)
cd "$(dirname "$0")" || exit 1

echo "============================================================"
echo "  Atualizando o relatório com dados do Yahoo Finance..."
echo "============================================================"

# escolhe o Python 3 disponível
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo
  echo "  ⚠ Python 3 não encontrado. Instale em https://www.python.org/downloads/"
  echo "    e rode este atalho de novo."
  echo
  read -n 1 -s -r -p "Pressione qualquer tecla para fechar..."
  exit 1
fi

# garante as bibliotecas (rápido se já estiverem instaladas)
if ! "$PY" -c "import yfinance, plotly, scipy, pandas, numpy" >/dev/null 2>&1; then
  echo "  Instalando bibliotecas necessárias (só na primeira vez)..."
  "$PY" -m pip install -r requirements.txt
fi

# roda o backtest (ele baixa os dados, gera o HTML e abre no navegador)
"$PY" Backtest.py
STATUS=$?

echo
if [ $STATUS -ne 0 ]; then
  echo "  ⚠ Algo deu errado ao gerar o relatório (veja as mensagens acima)."
  echo "    Copie o erro e me mande que eu ajudo."
  read -n 1 -s -r -p "Pressione qualquer tecla para fechar..."
else
  echo "  ✅ Relatório atualizado e aberto no navegador!"
  echo "     Pode fechar esta janela."
fi
