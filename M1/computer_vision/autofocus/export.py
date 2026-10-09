"""English reports with exact top-three image identities and per-method figures."""
from pathlib import Path
from datetime import datetime
import base64
import csv
import html
import io
import json
import re
import cv2
from .analysis import AnalysisResult, method_name, top_indices
from .metrics import BY_KEY, PAPER_URL
from .plotting import curves_figure, statistics_figure, batch_figure, method_figure


def _png(figure):
    stream = io.BytesIO()
    figure.savefig(stream,format="png",dpi=130,facecolor="white")
    figure.clear()
    return stream.getvalue()


def _image_tag(data,caption):
    return f'<figure><img src="data:image/png;base64,{base64.b64encode(data).decode()}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


HEADER = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FocusLab · Static autofocus report</title><style>body{font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#edf2f7;color:#24344a;margin:0}main{max-width:1180px;margin:40px auto;padding:0 26px}h1{font-size:36px;letter-spacing:-1px}h2{margin-top:0}section{padding:30px;background:white;border:1px solid #dce4ed;border-radius:16px;margin:24px 0}img{width:100%;height:auto}figure{margin:18px 0}figcaption{font-size:12px;color:#687b94}table{border-collapse:collapse;min-width:800px;width:100%;font-size:12px}td,th{padding:10px;border-bottom:1px solid #e1e7ef;text-align:left}th{background:#f1f6fa}.table{overflow:auto}.path,pre{word-break:break-all;white-space:pre-wrap;font-size:12px;background:#f4f7fb;padding:12px;border-radius:8px}.warning{color:#90672a}.topthree{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}code{color:#0b8c76}footer{color:#687b94;font-size:12px;padding:20px 0}</style><main><h1>FocusLab / Static autofocus</h1><p>CV1A §5.6 · Coarse image selection and local quadratic position estimation</p><p>Fine WD is an interpolated estimate, not an acquired image. R² is goodness of fit, not a confidence probability. Peak gap is a descriptive statistic, not statistical significance. Consensus averages ranks of correlated measures; agreement does not establish ground truth.</p>'


def export_results(results: list[AnalysisResult],destination: str | Path) -> Path:
    if not results:
        raise ValueError("No analysis results to export.")
    root=Path(destination).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime("autofocus_%Y%m%d_%H%M%S")
    folder=root/stamp
    suffix=1
    while folder.exists():
        folder=root/f"{stamp}_{suffix}"
        suffix+=1
    folder.mkdir()
    sections,overview,toprows=[],[],[]
    for number,result in enumerate(results,1):
        slug=re.sub(r"[^\w.-]","_",result.dataset.name)
        sub=folder/f"{number:02d}_{slug}"
        sub.mkdir()
        (sub/"summary.json").write_text(json.dumps(result.summary(),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        with (sub/"scores.csv").open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["sequence_1_based","filename","absolute_source_path","working_distance_mm",*result.scores])
            for i,e in enumerate(result.entries):
                writer.writerow([e.sequence+1,e.name,result.dataset.display_path(e),e.distance_mm,*[float(v[i]) for v in result.scores.values()]])
        rows,gallery=[],[]
        for key,out in result.methods.items():
            entry=result.entries[out.search.index]
            fine,r2=out.fine.distance_mm,out.fine.r_squared
            focus=float(result.scores[key][out.search.index])
            ms=out.statistics.get("median_scoring_ms")
            overview.append([result.dataset.name,key,entry.name,entry.sequence+1,entry.distance_mm,focus,fine,r2,out.search.corrected,ms,out.statistics["peak_gap_percent"]])
            fields=[method_name(key),entry.name,str(entry.sequence+1),f"{entry.distance_mm:.6f}",f"{focus:.7g}",f"{fine:.6f}" if fine is not None else "Unavailable",f"{r2:.4f}" if r2 is not None else "—",f"{ms:.3f}" if ms is not None else "—"]
            rows.append('<tr>'+''.join(f'<td>{html.escape(value)}</td>' for value in fields)+'</tr>')
            images=[]
            for rank,index in enumerate(top_indices(result.scores[key]),1):
                e=result.entries[index]
                original=result.dataset.load(e)
                ok,encoded=cv2.imencode(".png",original)
                if not ok:
                    raise ValueError(f"Image export failed: {e.name}")
                filename=f"best_{key}.png" if rank==1 else f"rank{rank}_{key}.png"
                (sub/filename).write_bytes(encoded.tobytes())
                scale=min(1.,600/max(original.shape[:2]))
                preview=cv2.resize(original,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else original
                _,small=cv2.imencode(".png",preview)
                score=float(result.scores[key][index])
                toprows.append([result.dataset.name,key,rank,e.sequence+1,e.name,result.dataset.display_path(e),e.distance_mm,score])
                images.append(_image_tag(small.tobytes(),f"Rank {rank} · image #{e.sequence+1}/{len(result.dataset.images)} · {e.name} · {e.distance_mm:.6f} mm · focus {score:.7g}"))
            graph=_png(method_figure(result,key))
            (sub/f"focus_{key}.png").write_bytes(graph)
            gallery.append(f'<h4>{html.escape(method_name(key))}</h4><div class="topthree">{"".join(images)}</div>'+_image_tag(graph,"Recorded focus curve and local quadratic refinement"))
        plots=[]
        for filename,fig,caption in [("focus_curves.png",curves_figure(result),"All selected focus measures; scaled within each method"),("statistics.png",statistics_figure(result),"Score distributions and Spearman rank correlations")]:
            data=_png(fig)
            (sub/filename).write_bytes(data)
            plots.append(_image_tag(data,caption))
        warn=''.join(f'<li>{html.escape(w)}</li>' for w in result.warnings)
        status=''.join(f'<li><b>{html.escape(method_name(k))}</b>: {html.escape(o.fine.status)} {html.escape(o.search.message)}</li>' for k,o in result.methods.items())
        definitions=''.join(f'<li><b>{html.escape(BY_KEY[k].name)}</b> · <code>{html.escape(BY_KEY[k].formula)}</code><p>{html.escape(BY_KEY[k].explanation)} {html.escape(BY_KEY[k].limitation)} {html.escape(BY_KEY[k].parameters)}</p></li>' for k in result.options.methods)
        heads=['Algorithm','Rank-1 file','Image #','Coarse mm','Focus','Fine mm','Fit R²','Median scoring ms']
        head=''.join(f'<th>{h}</th>' for h in heads)
        settings=html.escape(json.dumps(result.summary()['settings'],indent=2))
        sections.append(f'<section><h2>{html.escape(result.dataset.name)}</h2><p class="path">{html.escape(str(result.dataset.source))}</p><p>{len(result.entries)} / {len(result.dataset.images)} images · {result.elapsed_seconds:.2f} s · {html.escape(result.dataset.position_source)}</p><pre>{settings}</pre><ul class="warning">{warn}</ul><div class="table"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>{"".join(plots)}<h3>Fit and search status</h3><ul>{status}</ul><h3>Top 3 and graph for each algorithm</h3>{"".join(gallery)}<h3>Method definitions</h3><ul>{definitions}</ul></section>')
    for filename,heads,data in [
        ('comparison.csv',['dataset','method','coarse_image','sequence_1_based','coarse_mm','focus','fine_mm','fit_r_squared','global_correction','median_scoring_ms','peak_gap_percent'],overview),
        ('top_three.csv',['dataset','method','score_rank','image_sequence_1_based','filename','original_full_path','working_distance_mm','focus_score'],toprows)]:
        with (folder/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(heads)
            writer.writerows(data)
    batch=''
    if len(results)>1:
        data=_png(batch_figure(results))
        (folder/'scanning_speed_comparison.png').write_bytes(data)
        batch=_image_tag(data,'Coarse positions across scanning speeds')
    document=HEADER+'<p>Peak gap, fit R² and measured scoring time describe separate trade-offs. No universal best algorithm is asserted for stacks without independent focus labels.</p>'+batch+''.join(sections)
    document+=f'<footer>Course: 2026–2027_CV1A-lab, §5.6, pp. 29–30; 2026–2027_IMG_lectures, Chapter 6.<br>Research operators: <a href="{PAPER_URL}">Pertuz, Puig &amp; Garcia (2013), Analysis of focus measure operators for shape-from-focus</a>. Global discrete adaptations; no new synthetic or reference-label experiment is included.</footer></main></html>'
    report=folder/'report.html'
    report.write_text(document,encoding='utf-8')
    return report
