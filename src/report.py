"""Geração de relatório HTML autocontido a partir dos artefatos reais."""
from __future__ import annotations

import base64
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _format_value(value: Any) -> str:
    """Formata números para leitura humana sem alterar o valor armazenado."""
    if pd.isna(value):
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))


def _image_data(path: Path) -> str:
    """Converte uma imagem em data URI, tornando o HTML portátil."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _table(frame: pd.DataFrame) -> str:
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = "".join(f"<td>{_format_value(value)}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def generate_html_report(
    metrics_dir: Path,
    figures_dir: Path,
    models_dir: Path,
    output_path: Path,
) -> Path:
    """Cria dashboard HTML usando somente métricas e gráficos já calculados."""
    required = [
        metrics_dir / "model_comparison.csv",
        metrics_dir / "threshold_analysis.csv",
        metrics_dir / "final_test_metrics.json",
        metrics_dir / "classification_report_test.json",
        models_dir / "model_metadata.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Artefatos necessários ao relatório ausentes: {missing}")

    comparison = pd.read_csv(required[0])
    thresholds = pd.read_csv(required[1])
    final_metrics = json.loads(required[2].read_text(encoding="utf-8"))
    classification = json.loads(required[3].read_text(encoding="utf-8"))
    metadata = json.loads(required[4].read_text(encoding="utf-8"))
    ranked = comparison.loc[~comparison["Balanceamento"].str.startswith("dummy")].sort_values(
        ["Custo", "PR-AUC", "Recall da fraude"], ascending=[True, False, False]
    )
    winner = ranked.iloc[0]
    fraud_support = int(classification.get("1", {}).get("support", 0))
    normal_support = int(classification.get("0", {}).get("support", 0))
    detected_frauds = fraud_support - int(final_metrics["false_negatives"])
    reviewed_transactions = int(final_metrics["false_positives"]) + detected_frauds

    metric_cards = [
        ("Modelo selecionado", winner["Balanceamento"]),
        ("Threshold", final_metrics["threshold"]),
        ("Precisão da fraude", final_metrics["fraud_precision"]),
        ("Recall da fraude", final_metrics["fraud_recall"]),
        ("F1 da fraude", final_metrics["fraud_f1"]),
        ("PR-AUC", final_metrics["average_precision"]),
        ("ROC-AUC", final_metrics["roc_auc"]),
        ("Custo estimado", final_metrics["cost"]),
        ("Falsos positivos", final_metrics["false_positives"]),
        ("Falsos negativos", final_metrics["false_negatives"]),
    ]
    cards = "".join(
        f'<article class="card"><span>{html.escape(label)}</span><strong>{_format_value(value)}</strong></article>'
        for label, value in metric_cards
    )
    class_frame = pd.DataFrame(classification).T.reset_index(names="Classe")
    threshold_view = thresholds[
        ["threshold", "fraud_precision", "fraud_recall", "fraud_f1",
         "false_positives", "false_negatives", "cost"]
    ]
    main_figures = [
        ("Matriz de confusão — teste", "teste_matriz_confusao.png"),
        ("Curva Precision-Recall — teste", "teste_pr.png"),
        ("Curva ROC — teste", "teste_roc.png"),
        ("Métricas por threshold", "teste_thresholds.png"),
    ]
    eda_figures = [
        ("Distribuição das classes", "distribuicao_classes.png"),
        ("Distribuição dos valores", "distribuicao_amount.png"),
        ("Distribuição logarítmica", "distribuicao_amount_log.png"),
        ("Distribuição temporal", "distribuicao_time.png"),
        ("Valores por classe", "amount_por_classe.png"),
        ("Correlações relevantes", "matriz_correlacao.png"),
        ("Fraudes ao longo do tempo", "fraudes_tempo.png"),
    ]

    def gallery(items: list[tuple[str, str]]) -> str:
        figures = []
        for title, filename in items:
            path = figures_dir / filename
            if path.is_file():
                figures.append(
                    f'<figure><img src="{_image_data(path)}" alt="{html.escape(title)}">'
                    f"<figcaption>{html.escape(title)}</figcaption></figure>"
                )
        return "".join(figures) or "<p>Nenhuma figura disponível.</p>"

    generated_at = datetime.now().astimezone().strftime("%d/%m/%Y %H:%M %Z")
    document = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Resultado de Detecção de Fraudes em Transações</title>
<style>
:root{{--ink:#132238;--muted:#617085;--paper:#f2f5f8;--card:#fff;--navy:#102a43;--blue:#176b87;--teal:#18a999;--gold:#d7a33d;--line:#dbe3ea}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 Inter,Segoe UI,Arial,sans-serif}}
header{{position:relative;overflow:hidden;background:linear-gradient(120deg,#0b2036 0%,#123f5a 65%,#176b87 100%);color:#fff;padding:26px max(5vw,24px) 62px}}
header:after{{content:"";position:absolute;width:420px;height:420px;border:90px solid #ffffff0d;border-radius:50%;right:-120px;top:-220px}}
.topbar{{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #ffffff2b;padding-bottom:18px;margin-bottom:48px}}
.brand{{font-weight:700;letter-spacing:.08em}} .status{{font-size:12px;background:#22c55e22;border:1px solid #86efac66;padding:5px 11px;border-radius:99px}}
header p{{max-width:820px;color:#dceaf1;font-size:17px}} main{{max-width:1320px;margin:auto;padding:34px 24px 70px}}
h1{{font-size:clamp(32px,5vw,58px);max-width:960px;line-height:1.08;margin:0 0 18px;letter-spacing:-.035em}} h2{{margin:54px 0 8px;font-size:26px;color:var(--navy)}}
.section-intro{{color:var(--muted);max-width:900px;margin:0 0 22px}}
.eyebrow{{text-transform:uppercase;letter-spacing:.17em;font-size:11px;font-weight:800;color:#70d6c9;margin-bottom:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}}
.card,figure,.panel{{background:var(--card);border:1px solid var(--line);border-radius:10px;box-shadow:0 8px 28px #17324d0d}}
.card{{padding:20px;border-top:3px solid var(--teal)}} .card span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.045em}} .card strong{{font-size:24px;color:var(--navy)}}
.executive{{display:grid;grid-template-columns:1.45fr 1fr;gap:18px;margin-top:24px}}
.executive .panel{{padding:25px}} .executive h3{{margin:0 0 10px;color:var(--navy)}} .executive ul{{padding-left:20px;margin-bottom:0}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}}
figure{{margin:0;padding:16px}} figure img{{display:block;width:100%;height:auto;border-radius:5px}} figcaption{{padding:12px 4px 2px;font-weight:700;color:var(--navy)}}
.table-wrap{{overflow:auto;background:#fff;border:1px solid var(--line);border-radius:10px}}
table{{border-collapse:collapse;width:100%;white-space:nowrap}} th,td{{padding:11px 13px;border-bottom:1px solid var(--line);text-align:right}}
th{{position:sticky;top:0;background:#eaf1f5;color:#29465b;text-transform:uppercase;font-size:11px;letter-spacing:.03em}} th:first-child,td:first-child{{text-align:left}}
tbody tr:hover{{background:#f5fafb}} .note{{padding:18px 20px;border-left:4px solid var(--gold);background:#fff9eb;border-radius:6px}}
.definitions{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}} .definition{{padding:20px;background:#fff;border:1px solid var(--line);border-radius:8px}}
.definition strong{{display:block;color:var(--blue);font-size:16px;margin-bottom:6px}} .definition p{{margin:0;color:var(--muted)}}
footer{{color:var(--muted);margin-top:50px}}
@media(max-width:760px){{main{{padding:24px 14px}}.gallery,.executive{{grid-template-columns:1fr}}.topbar{{align-items:flex-start;gap:10px;flex-direction:column}}}}
</style>
</head>
<body>
<header><div class="topbar"><div class="brand">ANÁLISE DE RISCO · MACHINE LEARNING</div><div class="status">● Processamento concluído</div></div>
<div class="eyebrow">Relatório executivo de desempenho</div>
<h1>Resultado de Detecção de Fraudes em Transações</h1>
<p>Visão consolidada do desempenho do modelo, do impacto operacional das decisões e dos
principais indicadores de identificação de transações potencialmente fraudulentas.</p></header>
<main>
<h2>Resumo executivo</h2><section class="grid">{cards}</section>
<section class="executive">
<div class="panel"><h3>Leitura dos resultados</h3>
<p>O modelo identificou <strong>{detected_frauds} de {fraud_support} fraudes</strong> presentes no conjunto
de teste, equivalente a um recall de <strong>{final_metrics["fraud_recall"]:.1%}</strong>.
Entre os casos classificados como fraude, <strong>{final_metrics["fraud_precision"]:.1%}</strong>
eram efetivamente fraudulentos.</p>
<p>Ao todo, <strong>{reviewed_transactions}</strong> transações foram sinalizadas para bloqueio ou revisão,
incluindo <strong>{final_metrics["false_positives"]}</strong> operações legítimas. Permaneceram
<strong>{final_metrics["false_negatives"]}</strong> fraudes não detectadas.</p></div>
<div class="panel"><h3>Contexto operacional</h3><ul>
<li><strong>{normal_support:,}</strong> transações legítimas avaliadas.</li>
<li><strong>{fraud_support}</strong> transações fraudulentas avaliadas.</li>
<li>Threshold operacional de <strong>{final_metrics["threshold"]:.2f}</strong>.</li>
<li>Modelo selecionado: <strong>{html.escape(str(winner["Balanceamento"]))}</strong>.</li>
</ul></div></section>
<p class="note"><strong>Nota de interpretação:</strong> o custo apresentado é uma simulação
configurável, não um valor bancário real. A decisão de negócio deve considerar conjuntamente
recall, precisão, PR-AUC e o volume operacional de falsos alertas.</p>
<h2>Como interpretar as métricas</h2>
<p class="section-intro">Cada indicador responde a uma pergunta diferente. Em fraude,
nenhuma métrica isolada oferece uma visão completa da qualidade do sistema.</p>
<section class="definitions">
<div class="definition"><strong>Recall da fraude</strong><p>De todas as fraudes reais, qual percentual foi detectado? Recall alto reduz fraudes que passam despercebidas.</p></div>
<div class="definition"><strong>Precisão da fraude</strong><p>De todos os alertas gerados, qual percentual era fraude? Precisão baixa aumenta bloqueios e revisões desnecessárias.</p></div>
<div class="definition"><strong>F1-score</strong><p>Síntese harmônica entre precisão e recall. É útil para comparar equilíbrio, mas não substitui a análise de custo.</p></div>
<div class="definition"><strong>PR-AUC</strong><p>Resume a relação entre precisão e recall em vários thresholds. É especialmente relevante quando fraudes são raras.</p></div>
<div class="definition"><strong>ROC-AUC</strong><p>Mede a capacidade geral de ordenar fraudes acima de transações normais em diferentes limiares.</p></div>
<div class="definition"><strong>Falsos positivos e negativos</strong><p>Falso positivo é uma transação legítima sinalizada; falso negativo é uma fraude não detectada.</p></div>
</section>
<h2>Desempenho final no teste</h2><section class="gallery">{gallery(main_figures)}</section>
<h2>Comparação na validação</h2><p class="section-intro">Visão comparativa das estratégias
avaliadas antes do teste final. O modelo foi escolhido considerando custo estimado, PR-AUC
e capacidade de detectar fraudes, não apenas acurácia.</p>{_table(comparison)}
<h2>Análise de threshold</h2><p class="section-intro">O threshold controla a sensibilidade:
valores menores tendem a detectar mais fraudes, mas também podem aumentar falsos positivos.</p>{_table(threshold_view)}
<h2>Relatório por classe</h2><p class="section-intro">Detalhamento da qualidade preditiva
para transações normais e fraudulentas, incluindo suporte — a quantidade real de registros
de cada classe no teste.</p>{_table(class_frame)}
<h2>Perfil dos dados analisados</h2><p class="section-intro">Os gráficos exploratórios
contextualizam o forte desbalanceamento, os valores financeiros e os padrões temporais
observados no conjunto de dados.</p><section class="gallery">{gallery(eda_figures)}</section>
<h2>Rastreabilidade</h2>
<div class="panel" style="padding:20px"><p><strong>Classe do estimador:</strong>
{html.escape(str(metadata.get("model_class", "—")))}</p>
<p><strong>Quantidade de features brutas:</strong> {len(metadata.get("raw_features", []))}</p>
<p><strong>Gerado em:</strong> {html.escape(generated_at)}</p></div>
<footer>Resultado de Detecção de Fraudes em Transações · Projeto educacional ·
Gerado automaticamente a partir dos artefatos do pipeline.</footer>
</main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path
