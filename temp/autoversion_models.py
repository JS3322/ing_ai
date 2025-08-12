#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autoversion_models.py
- Scan models/ for ONNX models
- Detect code/model changes
- Maintain per-model registries and auto bump versions

Policy:
  MAJOR  -> data_hash changed (from .meta_info.pipeline.prediction.input_dir_path)
  MINOR  -> any of schema_hash | arch_hash | weights_hash | behavior_hash changed
  PATCH  -> only code_hash changed or no effective change

Behavior hash execution:
  --behavior=auto (default): run only when first-time OR arch changed
  --behavior=always: always run once per model
  --behavior=never: never run (behavior_hash kept as "")
"""

import os
import re
import json
import argparse
import hashlib
import ast
from typing import Any, Dict, List, Optional, Tuple, Set

import onnx
import numpy as np
from onnx import numpy_helper

# ------------------ small utils ------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def json_hash(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode())

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def parse_version(ver: str) -> Tuple[int,int,int]:
    try:
        a,b,c = ver.split(".")
        return int(a), int(b), int(c)
    except Exception:
        return (0,0,0)

def bump_version(prev: str, kind: str) -> str:
    a,b,c = parse_version(prev)
    if kind == "MAJOR": return f"{a+1}.0.0"
    if kind == "MINOR": return f"{a}.{b+1}.0"
    return f"{a}.{b}.{c+1}"

# ------------------ ONNX fingerprints ------------------

def onnx_schema_hash(path: str, include_names: bool=False) -> str:
    m = onnx.load(path)
    def iodef(vs):
        items = []
        for v in vs:
            t = v.type.tensor_type
            elem = t.elem_type
            shape = [d.dim_value if d.HasField("dim_value") else -1 for d in t.shape.dim]
            items.append((v.name if include_names else None, elem, tuple(shape)))
        return sorted(items, key=lambda x: (x[0] or "", x[1], x[2]))
    return json_hash({"in": iodef(m.graph.input), "out": iodef(m.graph.output)})

def onnx_arch_hash(path: str) -> str:
    m = onnx.load(path)
    sig = []
    for n in m.graph.node:
        attrs = []
        for a in n.attribute:
            if   a.type == onnx.AttributeProto.FLOAT:    v = ("f", float(a.f))
            elif a.type == onnx.AttributeProto.INT:      v = ("i", int(a.i))
            elif a.type == onnx.AttributeProto.STRING:   v = ("s", a.s.decode("utf-8", "ignore"))
            elif a.type == onnx.AttributeProto.FLOATS:   v = ("fs", tuple(float(x) for x in a.floats))
            elif a.type == onnx.AttributeProto.INTS:     v = ("is", tuple(int(x) for x in a.ints))
            else:                                        v = ("other", a.name)  # tensor attrs → name only
            attrs.append((a.name, v))
        sig.append((
            n.op_type,
            tuple(sorted(n.input)),
            tuple(sorted(n.output)),
            tuple(sorted(attrs)),
        ))
    return json_hash(sig)

def onnx_weights_hash(path: str) -> str:
    m = onnx.load(path)
    parts = []
    for init in sorted(m.graph.initializer, key=lambda x: x.name):
        arr = numpy_helper.to_array(init)  # handles external data too
        parts.append((init.name, arr.shape, str(arr.dtype), sha256_bytes(arr.tobytes())))
    return json_hash(parts)

def make_synthetic_feeds(onnx_path: str, seed: int = 20250812) -> Optional[Dict[str, np.ndarray]]:
    from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
    m = onnx.load(onnx_path)
    rng = np.random.default_rng(seed)
    feeds: Dict[str, np.ndarray] = {}

    for v in m.graph.input:
        t = v.type
        if not t.HasField("tensor_type"): 
            return None  # sequence/map not supported here
        tt = t.tensor_type
        elem = tt.elem_type
        if elem not in TENSOR_TYPE_TO_NP_TYPE: 
            return None
        np_dtype = TENSOR_TYPE_TO_NP_TYPE[elem]
        # dynamic dims -> 1
        shape = [d.dim_value if d.HasField("dim_value") else 1 for d in tt.shape.dim]
        if np.issubdtype(np_dtype, np.floating):
            arr = rng.normal(0.0, 1.0, size=tuple(shape)).astype(np_dtype)
        elif np.issubdtype(np_dtype, np.integer):
            arr = rng.integers(0, 10, size=tuple(shape), dtype=np_dtype)
        elif np_dtype == np.bool_:
            arr = rng.integers(0, 2, size=tuple(shape)).astype(np.bool_)
        else:
            return None
        feeds[v.name] = arr
    return feeds

def behavior_hash(onnx_path: str, round_ndigits: int = 8) -> str:
    feeds = make_synthetic_feeds(onnx_path)
    if feeds is None:
        return ""  # unsupported → skip
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outs = sess.run(None, feeds)
    norm = [np.round(o.astype(np.float64), round_ndigits) for o in outs]
    return sha256_bytes(b"".join(o.tobytes() for o in norm))

# ------------------ data & code fingerprints ------------------

def dir_data_hash(root: str) -> str:
    if not root or not os.path.isdir(root):
        return ""
    chunks: List[str] = []
    for base, _, files in os.walk(root):
        for f in sorted(files):
            p = os.path.join(base, f)
            try:
                with open(p, "rb") as fh:
                    chunks.append(sha256_bytes(fh.read()))
            except Exception:
                pass
    return json_hash(chunks) if chunks else ""

def extract_py_targets_from_command(command: str) -> List[str]:
    """Extract .py path or -m module from a shell-like command"""
    toks = re.split(r"\s+", command.strip())
    pys: List[str] = []
    for i, t in enumerate(toks):
        if t.endswith(".py"):
            pys.append(t)
        if t == "-m" and i+1 < len(toks):
            pys.append(("MODULE:", toks[i+1]))
    return pys

def resolve_module_to_path(module: str, base_dir: str) -> Optional[str]:
    """
    Resolve 'a.b.c' to a local path like base_dir/a/b/c.py or package __init__.py
    Only searches under base_dir (project root near .meta_info).
    """
    parts = module.split(".")
    cand1 = os.path.join(base_dir, *parts) + ".py"
    cand2 = os.path.join(base_dir, *parts, "__init__.py")
    if os.path.isfile(cand1): return cand1
    if os.path.isfile(cand2): return cand2
    return None

class StripDocstrings(ast.NodeTransformer):
    """Replace string constants likely to be docstrings with a placeholder to reduce noise."""
    def visit_Expr(self, node):
        # Top-level/module/class/function docstrings are usually Expr(Constant(str))
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return ast.Pass()
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value="__STR__"), node)
        return node

def ast_hash_of_file(py_path: str) -> str:
    try:
        with open(py_path, "r", encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src)
        tree = StripDocstrings().visit(tree)
        ast.fix_missing_locations(tree)
        dump = ast.dump(tree, annotate_fields=True, include_attributes=False)
        return sha256_bytes(dump.encode())
    except Exception:
        # fallback to raw file hash
        try:
            with open(py_path, "rb") as f:
                return sha256_bytes(f.read())
        except Exception:
            return ""

def collect_code_files(meta: Dict[str, Any], project_root: str) -> List[str]:
    """
    From .meta_info.pipeline.*.command, collect main .py and local imported modules (best-effort).
    """
    code_files: Set[str] = set()
    pipeline = meta.get("pipeline", {})
    for stage in pipeline.values():
        cmd = stage.get("command", "") or ""
        targets = extract_py_targets_from_command(cmd)
        for t in targets:
            if isinstance(t, tuple) and t[0] == "MODULE:":
                mod = t[1]
                p = resolve_module_to_path(mod, project_root)
                if p: code_files.add(os.path.abspath(p))
            else:
                p = t
                if not os.path.isabs(p):
                    p = os.path.abspath(os.path.join(project_root, p))
                if os.path.isfile(p): code_files.add(p)

    # (best-effort) parse direct 'import' from these files to add local modules
    def add_local_imports(py_path: str):
        try:
            with open(py_path, "r", encoding="utf-8") as f:
                src = f.read()
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        p = resolve_module_to_path(n.name, project_root)
                        if p: code_files.add(os.path.abspath(p))
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        p = resolve_module_to_path(node.module, project_root)
                        if p: code_files.add(os.path.abspath(p))
        except Exception:
            pass

    for p in list(code_files):
        add_local_imports(p)

    return sorted(code_files)

def code_hash(meta: Dict[str, Any], project_root: str) -> Tuple[str, List[str]]:
    files = collect_code_files(meta, project_root)
    if not files:
        # fall back to hashing all 'py' under project_root? too heavy; hash commands instead
        commands = []
        for stage in meta.get("pipeline", {}).values():
            commands.append(stage.get("command", "") or "")
        return json_hash(commands), []
    hashes = [ast_hash_of_file(p) for p in files]
    combined = json_hash(list(zip(files, hashes)))
    return combined, files

# ------------------ discovery & decision ------------------

def discover_models(model_dir: str) -> List[str]:
    out: List[str] = []
    for base, _, files in os.walk(model_dir):
        for f in files:
            if f.lower().endswith(".onnx"):
                out.append(os.path.join(base, f))
    return sorted(out)

def suggest_bump(prev_fp: Dict[str,str], cur_fp: Dict[str,str]) -> str:
    # MAJOR by data policy
    if prev_fp.get("data_hash") != cur_fp.get("data_hash"):
        return "MAJOR"
    # MINOR if any of model-facing hashes changed
    for k in ("schema_hash","arch_hash","weights_hash","behavior_hash"):
        if prev_fp.get(k) != cur_fp.get(k):
            return "MINOR"
    # PATCH if only code changed (or nothing)
    if prev_fp.get("code_hash") != cur_fp.get("code_hash"):
        return "PATCH"
    return "PATCH"

# ------------------ main flow ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=".meta_info", help="Path to .meta_info JSON (default: ./.meta_info)")
    ap.add_argument("--models-dir", default=None, help="Override models directory. Defaults to .meta_info.env.model_dir_path or ./models")
    ap.add_argument("--behavior", choices=["auto","always","never"], default="auto", help="When to run behavior hash (default: auto)")
    ap.add_argument("--include-names-in-schema", action="store_true", help="Include tensor names in schema hash (default: off)")
    args = ap.parse_args()

    meta = read_json(args.meta)
    project_root = os.path.dirname(os.path.abspath(args.meta))

    # models directory resolve
    model_dir = args.models_dir or meta.get("env", {}).get("model_dir_path") or os.path.join(project_root, "models")
    model_dir = os.path.abspath(model_dir)

    # data hash from prediction.input_dir_path (policy per request)
    input_dir = meta.get("pipeline", {}).get("prediction", {}).get("input_dir_path", "")
    if input_dir and not os.path.isabs(input_dir):
        input_dir = os.path.abspath(os.path.join(project_root, input_dir))
    d_hash = dir_data_hash(input_dir)

    # code hash from pipeline commands
    c_hash, code_files = code_hash(meta, project_root)

    # discover models
    model_paths = discover_models(model_dir)
    if not model_paths:
        print(json.dumps({"error": f"No .onnx models found under {model_dir}"}, ensure_ascii=False))
        return

    # registry folder
    registry_dir = os.path.join(model_dir, ".registry")
    os.makedirs(registry_dir, exist_ok=True)

    # pass 1: compute static hashes for all
    static_cache: Dict[str, Dict[str,str]] = {}
    for mpath in model_paths:
        try:
            schema = onnx_schema_hash(mpath, include_names=args.include_names_in_schema)
            arch   = onnx_arch_hash(mpath)
            wght   = onnx_weights_hash(mpath)
        except Exception as e:
            print(json.dumps({"model": mpath, "error": f"ONNX parse failed: {e}"}, ensure_ascii=False))
            continue
        static_cache[mpath] = {
            "schema_hash": schema,
            "arch_hash": arch,
            "weights_hash": wght,
        }

    summary: List[Dict[str,Any]] = []

    # pass 2: per model, load registry, decide if behavior needed, bump, write
    for mpath in model_paths:
        model_id = os.path.splitext(os.path.basename(mpath))[0]
        reg_path = os.path.join(registry_dir, f"{model_id}.json")

        prev_latest: Dict[str,Any] = {}
        prev_ver = "0.0.0"
        need_behavior = False

        if os.path.exists(reg_path):
            reg = read_json(reg_path)
            prev_latest = reg.get("latest", {})
            prev_ver = prev_latest.get("version", "0.0.0")
            prev_fp = prev_latest.get("fingerprints", {})
            # behavior run policy
            if args.behavior == "always":
                need_behavior = True
            elif args.behavior == "never":
                need_behavior = False
            else:  # auto
                # run if first time (no prev) or architecture changed
                need_behavior = (not prev_fp) or (prev_fp.get("arch_hash") != static_cache[mpath]["arch_hash"])
            reg_obj = reg
        else:
            reg_obj = {"history": []}
            if args.behavior == "always":
                need_behavior = True
            elif args.behavior == "never":
                need_behavior = False
            else:
                need_behavior = True  # first time

        # compute behavior (maybe)
        if need_behavior:
            try:
                b_hash = behavior_hash(mpath)
            except Exception:
                b_hash = ""
        else:
            # inherit previous behavior hash to avoid noise
            b_hash = prev_latest.get("fingerprints", {}).get("behavior_hash", "")

        # current fingerprints
        cur_fp = {
            **static_cache[mpath],
            "behavior_hash": b_hash,
            "data_hash": d_hash,
            "code_hash": c_hash,
        }

        # decide bump
        bump = "MAJOR" if not prev_latest else suggest_bump(prev_latest.get("fingerprints", {}), cur_fp)
        new_ver = "1.0.0" if not prev_latest else bump_version(prev_ver, bump)

        entry = {
            "version": new_ver,
            "file": os.path.relpath(mpath, model_dir),
            "fingerprints": cur_fp,
            "changed": []  # filled below
        }

        if prev_latest:
            prev_fp = prev_latest.get("fingerprints", {})
            diffs = []
            for k in ("data_hash","schema_hash","arch_hash","weights_hash","behavior_hash","code_hash"):
                if prev_fp.get(k) != cur_fp.get(k):
                    diffs.append(k)
            entry["changed"] = diffs

        reg_obj["history"] = reg_obj.get("history", []) + [entry]
        reg_obj["latest"]  = entry
        write_json(reg_path, reg_obj)

        summary.append({
            "model_id": model_id,
            "from": prev_ver if prev_latest else "—",
            "to": new_ver,
            "bump": bump,
            "registry": reg_path
        })

    print(json.dumps({"summary": summary, "models_dir": model_dir, "code_files": code_files}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()