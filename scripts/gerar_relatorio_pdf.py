"""Gera docs/relatorio.pdf a partir de docs/relatorio.md (o PDF que vai para o Teams).

    pip install -r requirements/dev.txt && python -m playwright install chromium
    python scripts/gerar_relatorio_pdf.py

As imagens são lidas dos caminhos relativos do próprio Markdown (docs/prints/, resultados/).
"""

import sys

import markdown
from playwright.sync_api import sync_playwright

from _comum import RAIZ

DOCS = RAIZ / "docs"

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1d1f24; }
h1 { font-size: 20pt; margin: 0 0 4pt; color: #8a1240; }
h2 { font-size: 14pt; margin: 16pt 0 6pt; border-bottom: 1.5px solid #e5e6ea; padding-bottom: 3pt; color: #8a1240; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; }
p { margin: 4pt 0 6pt; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #d4d6dc; padding: 4pt 6pt; text-align: left; vertical-align: top; }
th { background: #f3f4f7; }
img { max-width: 100%; max-height: 105mm; display: block; margin: 6pt auto; page-break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f3f4f7; padding: 0 2pt; }
pre { background: #f3f4f7; padding: 6pt; font-size: 8.5pt; white-space: pre-wrap; page-break-inside: avoid; }
li { margin-bottom: 2pt; }
"""


def main() -> int:
    md = (DOCS / "relatorio.md").read_text(encoding="utf-8")
    corpo = markdown.markdown(md, extensions=["tables", "fenced_code"])
    html = (f"<!doctype html><html lang=pt-BR><head><meta charset=utf-8>"
            f"<style>{CSS}</style></head><body>{corpo}</body></html>")
    saida = DOCS / "relatorio.pdf"
    # aberto como arquivo em docs/ para os caminhos relativos das imagens funcionarem
    temporario = DOCS / "_relatorio.html"
    temporario.write_text(html, encoding="utf-8")
    with sync_playwright() as p:
        navegador = p.chromium.launch()
        pagina = navegador.new_page()
        pagina.goto(temporario.as_uri(), wait_until="load")
        pagina.pdf(path=str(saida), format="A4", print_background=True,
                   margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
                   display_header_footer=True, header_template="<span></span>",
                   footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#888'>"
                                   "Passo Seguro · Applied Computer Vision · <span class=pageNumber></span>/<span class=totalPages></span></div>")
        navegador.close()
    temporario.unlink()
    print(f"{saida} ({saida.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
