#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autoversion_bundle.py
- Treat models/ (env.model_dir_path) as ONE bundle
- Compute bundle fingerprints and bump version into .registory/autoversion_bundle.json

Versioning policy (bundle):
  MAJOR -> data_hash changed
  MINOR -> root_code_hash changed
  PATCH -> otherwise (inventory/schema/arch/weights/behavior changes etc.)

Behavior hash execution:
  --behavior=auto (default): run when first-time OR any model's arch changed
  --behavior=always: compute for every model
  --behavior=never: skip behavior (kept as "" if not present)

Writes:
  <model_dir>/.registory/autoversion_bundle.json
"""

import os, re, json, argparse, hashlib, ast, subprocess
from typing import Any, Dict, List, Tuple, Optional
import onnx
import numpy as np
from onnx import numpy_helper

# ---------- utils ----------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def json_hash(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode())

def read_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(p: str, obj: Any):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
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

# ---------- ONNX fingerprints ----------
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
            elif a.type == onnx.AttributeProto.STRING:   v = ("s", a.s.decode("utf-8","ignore"))
            elif a.type == onnx.AttributeProto.FLOATS:   v = ("fs", tuple(float(x) for x in a.floats))
            elif a.type == onnx.AttributeProto.INTS:     v = ("is", tuple(int(x) for x in a.ints))
            else:                                        v = ("other", a.name)
            attrs.append((a.name, v))
        sig.append((n.op_type, tuple(sorted(n.input)), tuple(sorted(n.output)), tuple(sorted(attrs))))
    return json_hash(sig)

def onnx_weights_hash(path: str) -> str:
    m = onnx.load(path)
    parts = []
    for init in sorted(m.graph.initializer, key=lambda x: x.name):
        arr = numpy_helper.to_array(init)
        parts.append((init.name, arr.shape, str(arr.dtype), sha256_bytes(arr.tobytes())))
    return json_hash(parts)

def make_synth_feeds(path: str, seed: int = 20250812) -> Optional[Dict[str, np.ndarray]]:
    from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
    m = onnx.load(path)
    rng = np.random.default_rng(seed)
    feeds = {}
    for v in m.graph.input:
        t = v.type
        if not t.HasField("tensor_type"): return None
        tt = t.tensor_type
        elem = tt.elem_type
        if elem not in TENSOR_TYPE_TO_NP_TYPE: return None
        np_dtype = TENSOR_TYPE_TO_NP_TYPE[elem]
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

def behavior_hash(path: str, ndigits: int = 8) -> str:
    feeds = make_synth_feeds(path)
    if feeds is None: return ""
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    outs = sess.run(None, feeds)
    norm = [np.round(o.astype(np.float64), ndigits) for o in outs]
    return sha256_bytes(b"".join(o.tobytes() for o in norm))

# ---------- data / code / git ----------
def dir_data_hash(root: str) -> str:
    if not root or not os.path.isdir(root):
        return ""
    chunks = []
    for base, _, files in os.walk(root):
        for f in sorted(files):
            p = os.path.join(base, f)
            try:
                with open(p, "rb") as fh:
                    chunks.append(sha256_bytes(fh.read()))
            except Exception:
                pass
    return json_hash(chunks) if chunks else ""

def extract_py_targets(command: str) -> List[str]:
    toks = re.split(r"\s+", command.strip())
    pys = []
    for i, t in enumerate(toks):
        if t.endswith(".py"): pys.append(t)
        if t == "-m" and i+1 < len(toks): pys.append(("MODULE:", toks[i+1]))
    return pys

def resolve_module(module: str, base_dir: str) -> Optional[str]:
    parts = module.split(".")
    cand1 = os.path.join(base_dir, *parts) + ".py"
    cand2 = os.path.join(base_dir, *parts, "__init__.py")
    if os.path.isfile(cand1): return cand1
    if os.path.isfile(cand2): return cand2
    return None

class StripDocstrings(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return ast.Pass()
        return self.generic_visit(node)
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value="__STR__"), node)
        return node

def ast_hash(py_path: str) -> str:
    try:
        with open(py_path, "r", encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src)
        tree = StripDocstrings().visit(tree)
        ast.fix_missing_locations(tree)
        dump = ast.dump(tree, annotate_fields=True, include_attributes=False)
        return sha256_bytes(dump.encode())
    except Exception:
        try:
            with open(py_path, "rb") as f:
                return sha256_bytes(f.read())
        except Exception:
            return ""

def collect_code_files(meta: Dict[str,Any], project_root: str) -> List[str]:
    code_files = set()
    pipeline = meta.get("pipeline", {})
    for stage in pipeline.values():
        cmd = stage.get("command", "") or ""
        for t in extract_py_targets(cmd):
            if isinstance(t, tuple):
                p = resolve_module(t[1], project_root)
            else:
                p = t if os.path.isabs(t) else os.path.join(project_root, t)
            if p and os.path.isfile(p):
                code_files.add(os.path.abspath(p))
    # 1-hop local imports
    def add_local(py):
        try:
            with open(py, "r", encoding="utf-8") as f:
                src = f.read()
            tree = ast.parse(src)
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    for name in n.names:
                        p = resolve_module(name.name, project_root)
                        if p: code_files.add(os.path.abspath(p))
                elif isinstance(n, ast.ImportFrom) and n.module:
                    p = resolve_module(n.module, project_root)
                    if p: code_files.add(os.path.abspath(p))
        except Exception:
            pass
    for p in list(code_files): add_local(p)
    return sorted(code_files)

def root_code_fingerprint(meta: Dict[str,Any], project_root: str) -> Dict[str,Any]:
    # Git info (best-effort)
    git = {"present": False}
    try:
        head = subprocess.check_output(["git","rev-parse","--short","HEAD"], cwd=project_root, text=True).strip()
        cnt  = subprocess.checkoutput(["git","rev-list","--count","HEAD"], cwd=project_root, text=True).strip()
        git = {"present": True, "short_sha": head, "revcount": int(cnt), "sha8_int": int(head, 16)}
    except Exception:
        pass

    files = collect_code_files(meta, project_root)
    if files:
        hashes = [ast_hash(p) for p in files]
        ast_combined = json_hash(list(zip(files, hashes)))
    else:
        # fall back to pipeline commands only
        cmds = [stage.get("command","") or "" for stage in meta.get("pipeline", {}).values()]
        ast_combined = json_hash(cmds)

    # root_code_hash 문자열(사람이 보기도 좋게)
    if git.get("present"):
        root_code_hash = f"git:revcount:{git['revcount']}|sha8:{git['short_sha']}|ast:{ast_combined[:12]}"
    else:
        root_code_hash = f"ast:{ast_combined}"

    return {
        "root_code_hash": root_code_hash,
        "git": git,
        "code_files": files
    }

# ---------- model discovery ----------
def discover_models(model_dir: str) -> List[str]:
    out = []
    for base, _, files in os.walk(model_dir):
        for f in files:
            if f.lower().endswith(".onnx"):
                out.append(os.path.join(base, f))
    return sorted(out)

def file_sha(p: str) -> str:
    try:
        with open(p, "rb") as f:
            return sha256_bytes(f.read())
    except Exception:
        return ""

# ---------- version decision ----------
def decide_bump(prev_fp: Dict[str,str], cur_fp: Dict[str,str]) -> str:
    if prev_fp.get("data_hash") != cur_fp.get("data_hash"):
        return "MAJOR"
    if prev_fp.get("root_code_hash") != cur_fp.get("root_code_hash"):
        return "MINOR"
    # anything else (inventory/schema/arch/weights/behavior/env)
    bundle_keys = ["inventory_hash","bundle_schema_hash","bundle_arch_hash","bundle_weights_hash","bundle_behavior_hash"]
    if any(prev_fp.get(k) != cur_fp.get(k) for k in bundle_keys):
        return "PATCH"
    return "PATCH"

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_info", required=True, help="Path to .meta_info JSON (required)")
    ap.add_argument("--behavior", choices=["auto","always","never"], default="auto", help="Behavior hash mode")
    ap.add_argument("--include-names-in-schema", action="store_true", help="Include tensor names into schema hash")
    args = ap.parse_args()

    meta_path = os.path.abspath(args.meta_info)
    if not os.path.isfile(meta_path):
        print(json.dumps({"error": f".meta_info not found: {meta_path}"}, ensure_ascii=False))
        return
    meta = read_json(meta_path)
    project_root = os.path.dirname(meta_path)

    # resolve model_dir
    model_dir_cfg = (meta.get("env", {}).get("model_dir_path") or "").strip()
    if not model_dir_cfg:
        print(json.dumps({"error":"env.model_dir_path is missing in .meta_info"}, ensure_ascii=False)); return
    model_dir = model_dir_cfg if os.path.isabs(model_dir_cfg) else os.path.abspath(os.path.join(project_root, model_dir_cfg))
    if not os.path.isdir(model_dir):
        print(json.dumps({"error": f"model_dir not found: {model_dir}"}, ensure_ascii=False)); return

    # data hash from prediction.input_dir_path
    input_dir_cfg = (meta.get("pipeline", {}).get("prediction", {}).get("input_dir_path") or "").strip()
    input_dir = input_dir_cfg if os.path.isabs(input_dir_cfg) else os.path.abspath(os.path.join(project_root, input_dir_cfg)) if input_dir_cfg else ""
    data_hash_val = dir_data_hash(input_dir)

    # root code fingerprint
    rcf = root_code_fingerprint(meta, project_root)
    root_code_hash = rcf["root_code_hash"]

    # models
    models = discover_models(model_dir)
    if not models:
        print(json.dumps({"error": f"No .onnx models found under {model_dir}"}, ensure_ascii=False)); return

    # bundle registry path (.registory)
    bundle_dir = os.path.join(model_dir, ".registory")
    os.makedirs(bundle_dir, exist_ok=True)
    bundle_path = os.path.join(bundle_dir, "autoversion_bundle.json")

    # load previous
    prev = {}
    if os.path.exists(bundle_path):
        try: prev = read_json(bundle_path)
        except Exception: prev = {}
    prev_latest = prev.get("latest", {})
    prev_fp = prev_latest.get("fingerprints", {})
    prev_per_model = prev.get("latest_per_model", {})  # {rel_path: {...}}

    # per-model fingerprints
    per_model: Dict[str, Dict[str,str]] = {}
    any_arch_changed = False

    for abspath in models:
        rel = os.path.relpath(abspath, model_dir)
        try:
            schema = onnx_schema_hash(abspath, include_names=args.include_names_in_schema)
            arch   = onnx_arch_hash(abspath)
            wght   = onnx_weights_hash(abspath)
        except Exception as e:
            # skip this model on error
            per_model[rel] = {"error": f"ONNX parse failed: {e}"}
            any_arch_changed = True
            continue

        # behavior policy
        need_behavior = False
        if args.behavior == "always":
            need_behavior = True
        elif args.behavior == "never":
            need_behavior = False
        else:  # auto
            prev_arch = prev_per_model.get(rel, {}).get("arch_hash", "")
            need_behavior = (prev_arch == "") or (prev_arch != arch)

        if need_behavior:
            try:
                bh = behavior_hash(abspath)
            except Exception:
                bh = ""
        else:
            bh = prev_per_model.get(rel, {}).get("behavior_hash", "")

        per_model[rel] = {
            "file_sha": file_sha(abspath),
            "schema_hash": schema,
            "arch_hash":   arch,
            "weights_hash": wght,
            "behavior_hash": bh
        }

        if prev_per_model.get(rel, {}).get("arch_hash", "") != arch:
            any_arch_changed = True

    # bundle-level hashes
    inventory = sorted([(rel, pm.get("file_sha",""), os.path.getsize(os.path.join(model_dir, rel)) if os.path.exists(os.path.join(model_dir, rel)) else 0) for rel, pm in per_model.items()])
    inventory_hash = json_hash(inventory)

    bundle_schema = sorted([(rel, pm.get("schema_hash","")) for rel, pm in per_model.items()])
    bundle_schema_hash = json_hash(bundle_schema)

    bundle_arch = sorted([(rel, pm.get("arch_hash","")) for rel, pm in per_model.items()])
    bundle_arch_hash = json_hash(bundle_arch)

    bundle_weights = sorted([(rel, pm.get("weights_hash","")) for rel, pm in per_model.items()])
    bundle_weights_hash = json_hash(bundle_weights)

    # behavior: compute if available; if never mode, may be many ""
    bundle_behavior = sorted([(rel, pm.get("behavior_hash","")) for rel, pm in per_model.items()])
    bundle_behavior_hash = json_hash(bundle_behavior)

    # current bundle fingerprints (flat)
    cur_fp = {
        "inventory_hash":     inventory_hash,
        "bundle_schema_hash": bundle_schema_hash,
        "bundle_arch_hash":   bundle_arch_hash,
        "bundle_weights_hash":bundle_weights_hash,
        "bundle_behavior_hash": bundle_behavior_hash,
        "root_code_hash":     root_code_hash,
        "data_hash":          data_hash_val
    }

    # decide bump
    bump = "MAJOR" if not prev_latest else decide_bump(prev_fp, cur_fp)
    new_ver = "1.0.0" if not prev_latest else bump_version(prev_latest.get("version","0.0.0"), bump)

    # entry
    entry = {
        "version": new_ver,
        "timestamp": None,  # CI에서 채우길 권장
        "fingerprints": cur_fp,
        "changed": [k for k in cur_fp.keys() if prev_fp.get(k) != cur_fp.get(k)] if prev_fp else list(cur_fp.keys()),
        "notes": ""
    }

    # write registry
    out = {
        "history": prev.get("history", []) + [entry],
        "latest": entry,
        # per-model fingerprints snapshot (helps 'auto' behavior decisions next run)
        "latest_per_model": per_model,
        # extras for debugging
        "extras": {
            "model_dir": model_dir,
            "input_dir": input_dir,
            "behavior_mode": args.behavior,
            "git": rcf.get("git", {}),
            "code_files": rcf.get("code_files", [])
        }
    }
    write_json(bundle_path, out)

    # stdout summary
    print(json.dumps({
        "bundle_registry": bundle_path,
        "version_bump": bump,
        "new_version": new_ver,
        "models_count": len(models),
        "changed_keys": entry["changed"]
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()