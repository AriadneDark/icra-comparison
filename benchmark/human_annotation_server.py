#!/usr/bin/env python3
"""Serve a dependency-free blind annotation UI for the hybrid graph benchmark."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Any

try:
    from .hybrid_common import (
        ROLE_ORDER, load_json, load_ontology, resolve_study_path, scoped_artifact, write_json,
    )
except ImportError:
    from hybrid_common import (
        ROLE_ORDER, load_json, load_ontology, resolve_study_path, scoped_artifact, write_json,
    )


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Task-role graph annotation</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#15171a;color:#eee} header{position:sticky;top:0;background:#222;padding:10px 18px;z-index:2}
main{max-width:1200px;margin:auto;padding:16px} button,select,input,textarea{font:inherit} button{padding:7px 13px;margin:3px}
video{width:100%;max-height:520px;background:#000}.card{background:#22262b;border-radius:8px;padding:14px;margin:12px 0}.evidence img{width:150px;margin:4px;border:1px solid #666}
table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #444;padding:7px;text-align:left;vertical-align:top}
input[type=text]{width:95%;padding:5px;background:#111;color:#eee;border:1px solid #555}textarea{width:98%;min-height:90px;background:#111;color:#eee}
.goal{font-size:1.1rem;color:#ffe08a}.muted{color:#aaa}.yes{color:#8ee28e}.no{color:#ff9696}.uncertain{color:#ffd27a}
.fact{font-family:ui-monospace,monospace}.nav{display:flex;align-items:center;gap:8px}.nav input{width:70px}.status{margin-left:auto}
</style></head><body>
<header><div class="nav"><button onclick="move(-1)">← Previous</button><input id="idx" type="number" min="1" onchange="jump()"><span id="total"></span><button onclick="move(1)">Next →</button><button onclick="save()">Save</button><span class="status" id="status"></span></div></header>
<main><h2 id="title"></h2><div class="goal" id="goal"></div><p class="muted">Judge only visible evidence. Method identities are hidden. Mark Complete only after adding facts missed by every candidate.</p>
<video id="video" controls preload="metadata"></video>
<section class="card"><h3>Ground-truth roles</h3><table><thead><tr><th>Role</th><th>Canonical visible object</th><th>Status</th><th>Visible intervals</th></tr></thead><tbody id="roles"></tbody></table></section>
<section class="card"><h3>Candidate role tracks (boxes)</h3><p class="muted">Judge the highlighted box. Keep only intervals where it follows the correct role object; exclude frames with a wrong box.</p><table><thead><tr><th>Blind role claim and box samples</th><th>Verdict</th><th>Frames with correct box</th></tr></thead><tbody id="roleFacts"></tbody></table></section>
<section class="card"><h3>Candidate relations</h3><p class="muted">Judge the relation from the video itself. Intervals here mean frames where the relation is true, not where a detector box is correct.</p><table><thead><tr><th>Blind relation claim</th><th>Verdict</th><th>Frames where relation is true</th></tr></thead><tbody id="relationFacts"></tbody></table></section>
<section class="card"><h3>Facts missed by the candidate pool</h3><p class="muted">One relation per line: subject | predicate | object | 0-8,20-29</p><textarea id="missing"></textarea></section>
<section class="card"><label>Notes<br><textarea id="notes"></textarea></label><br><label><input id="complete" type="checkbox"> Annotation complete</label></section>
</main><script>
let tasks=[], current=0, task=null;
const roles=['robot','manipulated_object','initial_support','target'];
const idx=document.getElementById('idx'), title=document.getElementById('title'), goal=document.getElementById('goal');
const video=document.getElementById('video'), missing=document.getElementById('missing'), notes=document.getElementById('notes');
const complete=document.getElementById('complete'), status=document.getElementById('status');
async function init(){tasks=await (await fetch('/api/tasks')).json();document.getElementById('total').textContent='/ '+tasks.length;await load(0)}
function esc(x){return String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function load(i){if(i<0||i>=tasks.length)return;current=i;task=await (await fetch('/api/task?index='+i)).json();idx.value=i+1;title.textContent=task.relative_path+' ['+task.evaluation_split+']';goal.textContent='Goal: '+task.planning_goal;video.src='/media/video?index='+i;
 let a=task.annotation||{}, gr=a.roles||{};rolesBody='';for(const r of roles){const v=gr[r]||{};rolesBody+=`<tr><td>${r}</td><td><input id="role_${r}_label" type="text" value="${esc(v.label||'')}"></td><td><select id="role_${r}_status">${['present','not_visible','not_applicable','ambiguous'].map(x=>`<option ${v.status===x?'selected':''}>${x}</option>`)}</select></td><td><input id="role_${r}_intervals" type="text" placeholder="0-29" value="${esc(formatIntervals(v.visible_intervals||[]))}"></td></tr>`}document.getElementById('roles').innerHTML=rolesBody;
 let labels=a.claim_labels||{}, roleRows='', relationRows='';for(const f of task.facts){let d=f.kind==='role'?`${f.role} is “${f.label}”`:`${f.subject} --${f.predicate}--> ${f.object}`;let v=labels[f.id]||{};let pics=f.kind==='role'?(f.evidence_boxes||[]).map(e=>`<img loading="lazy" src="/media/evidence?index=${i}&claim_id=${encodeURIComponent(f.id)}&frame=${e.frame_index}">`).join(''):'';let row=`<tr><td class="fact">${esc(d)}<div class="muted">${f.id}</div><div class="evidence">${pics}</div></td><td>${['yes','no','uncertain'].map(x=>`<label class="${x}"><input type="radio" name="claim_${f.id}" value="${x}" ${v.label===x?'checked':''}>${x}</label><br>`).join('')}</td><td><input id="interval_${f.id}" type="text" value="${esc(formatIntervals(v.intervals||f.suggested_intervals||[]))}"></td></tr>`;if(f.kind==='role')roleRows+=row;else relationRows+=row}document.getElementById('roleFacts').innerHTML=roleRows||'<tr><td colspan="3" class="muted">No candidate role tracks.</td></tr>';document.getElementById('relationFacts').innerHTML=relationRows||'<tr><td colspan="3" class="muted">No candidate relations.</td></tr>';
 missing.value=(a.missing_relations||[]).map(x=>`${x.subject} | ${x.predicate} | ${x.object} | ${formatIntervals(x.intervals||[])}`).join('\n');notes.value=a.notes||'';complete.checked=!!a.complete;status.textContent=a.complete?'✓ complete':'not saved/unfinished'}
function formatIntervals(xs){return xs.map(x=>x[0]+'-'+x[1]).join(',')}
function parseIntervals(text){if(!text.trim())return[];return text.split(',').map(x=>{let p=x.trim().split('-').map(Number);if(p.length!==2||p.some(Number.isNaN))throw Error('Bad interval: '+x);return [Math.min(...p),Math.max(...p)]})}
function parseMissing(){let out=[];for(const line of missing.value.split('\n')){if(!line.trim())continue;let p=line.split('|').map(x=>x.trim());if(p.length!==4||!roles.includes(p[0])||!roles.includes(p[2]))throw Error('Bad missing relation: '+line);out.push({subject:p[0],predicate:p[1],object:p[2],intervals:parseIntervals(p[3])})}return out}
async function save(){try{let annotation={video_id:task.video_id,annotator:task.annotator,complete:complete.checked,roles:{},claim_labels:{},missing_relations:parseMissing(),notes:notes.value};for(const r of roles){annotation.roles[r]={label:document.getElementById(`role_${r}_label`).value.trim(),status:document.getElementById(`role_${r}_status`).value,visible_intervals:parseIntervals(document.getElementById(`role_${r}_intervals`).value)}}for(const f of task.facts){let radio=document.querySelector(`input[name="claim_${f.id}"]:checked`);if(radio)annotation.claim_labels[f.id]={label:radio.value,intervals:parseIntervals(document.getElementById(`interval_${f.id}`).value)}}let response=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(annotation)});if(!response.ok)throw Error(await response.text());task.annotation=annotation;tasks[current].complete=annotation.complete;status.textContent=annotation.complete?'✓ complete':'saved (unfinished)'}catch(e){alert(e.message)}}
async function move(delta){await save();await load(current+delta)}async function jump(){await save();await load(Number(idx.value)-1)}init();
</script></body></html>"""


class AnnotationApp:
    def __init__(
        self, study_root: Path, video_root: Path, source_root: Path,
        annotator: str, only_double: bool = False,
    ) -> None:
        self.study_root = study_root.resolve()
        self.video_root = video_root.resolve()
        self.source_root = source_root.resolve()
        self.annotator = annotator
        study = load_json(self.study_root / "study_manifest.json")
        self.ontology = load_ontology(resolve_study_path(self.study_root, study["ontology"]))
        self.records = [
            record for record in study["episodes"]
            if record["evaluation_split"].startswith("human_")
            and (not only_double or record.get("double_annotation", False))
        ]
        self.annotation_root = self.study_root / "human_annotations" / annotator
        self.annotation_root.mkdir(parents=True, exist_ok=True)

    def annotation_path(self, record: dict[str, Any]) -> Path:
        return self.annotation_root / f"{record['video_id']}.json"

    def facts(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        vlm_path = self.study_root / "vlm" / f"{record['video_id']}.json"
        source = scoped_artifact(
            load_json(vlm_path) if vlm_path.exists()
            else load_json(self.study_root / record["candidate_path"]),
            self.ontology,
        )
        blind = []
        for fact in source["facts"]:
            item = {key: value for key, value in fact.items() if key not in {"sources", "source_intervals"}}
            intervals = sorted({
                tuple(interval) for values in fact.get("source_intervals", {}).values()
                for interval in values if len(interval) == 2
            })
            item["suggested_intervals"] = [list(value) for value in intervals]
            blind.append(item)
        return blind

    def source_facts(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        vlm_path = self.study_root / "vlm" / f"{record['video_id']}.json"
        source = scoped_artifact(
            load_json(vlm_path) if vlm_path.exists()
            else load_json(self.study_root / record["candidate_path"]),
            self.ontology,
        )
        return source["facts"]

    def evidence_image(self, index: int, claim_id: str, frame_index: int) -> bytes:
        from io import BytesIO
        from PIL import Image, ImageDraw

        record = self.records[index]
        fact = next((item for item in self.source_facts(record) if item["id"] == claim_id), None)
        if fact is None or fact.get("kind") != "role":
            raise ValueError("Unknown role claim")
        evidence = next(
            (item for item in fact.get("evidence_boxes", []) if int(item["frame_index"]) == frame_index), None
        )
        if evidence is None:
            raise ValueError("No evidence box for that frame")
        images = sorted((self.source_root / record["relative_path"] / "images").glob("frame_*.png"))
        if not 0 <= frame_index < len(images):
            raise ValueError("Invalid frame index")
        with Image.open(images[frame_index]) as source:
            image = source.convert("RGB")
        box = [float(value) for value in evidence["bbox"]]
        if evidence.get("normalized"):
            box = [box[0] * image.width, box[1] * image.height, box[2] * image.width, box[3] * image.height]
        draw = ImageDraw.Draw(image)
        draw.rectangle(tuple(box), outline=(255, 220, 40), width=max(2, image.width // 200))
        draw.rectangle((0, 0, min(image.width, 370), 24), fill=(15, 15, 15))
        draw.text((5, 5), f"blind track {claim_id} | frame {frame_index}", fill=(255, 255, 255))
        image.thumbnail((640, 640))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        return buffer.getvalue()

    def task(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = self.annotation_path(record)
        return {
            "video_id": record["video_id"], "relative_path": record["relative_path"],
            "planning_goal": record["planning_goal"], "frame_count": record["frame_count"],
            "evaluation_split": record["evaluation_split"], "annotator": self.annotator,
            "facts": self.facts(record), "annotation": load_json(path) if path.exists() else None,
        }

    def save(self, payload: dict[str, Any]) -> None:
        record = next((item for item in self.records if item["video_id"] == payload.get("video_id")), None)
        if record is None or payload.get("annotator") != self.annotator:
            raise ValueError("Unknown video or annotator")
        if not isinstance(payload.get("claim_labels"), dict) or not isinstance(payload.get("roles"), dict):
            raise ValueError("roles and claim_labels must be objects")
        if payload.get("complete"):
            facts = self.facts(record)
            expected = {fact["id"] for fact in facts}
            labeled = {
                identifier for identifier, value in payload["claim_labels"].items()
                if value.get("label") in {"yes", "no", "uncertain"}
            }
            if expected - labeled:
                raise ValueError(f"Complete annotation has {len(expected - labeled)} unlabeled candidate facts")
            if set(payload["roles"]) != set(ROLE_ORDER):
                raise ValueError("Complete annotation must describe all four roles")
            for role in ROLE_ORDER:
                role_value = payload["roles"][role]
                if role_value.get("status") not in {
                    "present", "not_visible", "not_applicable", "ambiguous"
                }:
                    raise ValueError(f"Invalid status for {role}")
                if role_value.get("status") == "present":
                    if not str(role_value.get("label", "")).strip():
                        raise ValueError(f"Present role {role} needs a canonical object label")
                    if not role_value.get("visible_intervals"):
                        raise ValueError(f"Present role {role} needs visible intervals")
            fact_by_id = {fact["id"]: fact for fact in facts}
            for identifier, value in payload["claim_labels"].items():
                if identifier not in fact_by_id:
                    raise ValueError(f"Unknown claim: {identifier}")
                if (
                    value.get("label") == "yes"
                    and not value.get("intervals")
                ):
                    raise ValueError(f"Accepted claim {identifier} needs corrected intervals")
        write_json(self.annotation_path(record), {"schema_version": "human_task_graph_v1", **payload})

    def video_path(self, index: int) -> Path:
        record = self.records[index]
        path = (self.video_root / f"{record['video_id']}.mp4").resolve()
        if path.parent != self.video_root:
            raise ValueError("Invalid video path")
        return path


def make_handler(app: AnnotationApp):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    body = HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif parsed.path == "/api/tasks":
                    tasks = []
                    for record in app.records:
                        path = app.annotation_path(record)
                        complete = bool(load_json(path).get("complete")) if path.exists() else False
                        tasks.append({"video_id": record["video_id"], "complete": complete})
                    self.send_json(tasks)
                elif parsed.path == "/api/task":
                    index = int(parse_qs(parsed.query).get("index", ["0"])[0])
                    self.send_json(app.task(index))
                elif parsed.path == "/media/video":
                    index = int(parse_qs(parsed.query).get("index", ["0"])[0])
                    self.send_file(app.video_path(index))
                elif parsed.path == "/media/evidence":
                    query = parse_qs(parsed.query)
                    data = app.evidence_image(
                        int(query.get("index", ["0"])[0]),
                        query.get("claim_id", [""])[0],
                        int(query.get("frame", ["0"])[0]),
                    )
                    self.send_bytes(data, "image/jpeg")
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, IndexError, FileNotFoundError) as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def send_file(self, path: Path) -> None:
            if not path.is_file():
                raise FileNotFoundError(path)
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_bytes(self, data: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/save":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                app.save(payload)
                self.send_json({"ok": True})
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"annotation-ui: {fmt % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--only-double", action="store_true",
                        help="Show only the preselected double-annotation subset.")
    args = parser.parse_args()
    app = AnnotationApp(
        Path(args.study_root), Path(args.video_root), Path(args.source_root),
        args.annotator, args.only_double,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Annotator {args.annotator}: {len(app.records)} tasks at http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
