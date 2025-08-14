# production_ready_api.py
# MIT License

import os
import io
import re
import gc
import sys
import json
import uuid
import time
import base64
import shutil
import atexit
import asyncio
import logging
import tempfile
import importlib
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# --------------------------- Logging & Config ---------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data-analyst-agent")

# Hard cap: keep under ~3 minutes end-to-end
HARD_TIME_CAP_SECONDS = int(os.getenv("API_TIME_CAP_SECONDS", "170"))
EXEC_SLICE_SECONDS = int(os.getenv("EXEC_SLICE_SECONDS", "120"))
DATA_URI_MAX_BYTES = int(os.getenv("DATA_URI_MAX_BYTES", "100000"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "3"))

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# --------------------------- App Setup ---------------------------------

app = FastAPI(title="Data Analyst Agent API", version="1.0.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_active_requests = set()
_TEMP_DIRS: List[str] = []


def _cleanup_tempdirs():
    for d in list(_TEMP_DIRS):
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


atexit.register(_cleanup_tempdirs)

# --------------------------- Helpers -----------------------------------


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def unique_work_dir() -> str:
    d = os.path.join(tempfile.gettempdir(), f"daa_{uuid.uuid4().hex}")
    ensure_dir(d)
    _TEMP_DIRS.append(d)
    return d


def safe_join(base: str, *paths: str) -> str:
    final = os.path.realpath(os.path.join(base, *paths))
    base_real = os.path.realpath(base)
    if not final.startswith(base_real + os.sep) and final != base_real:
        raise PermissionError("Path traversal blocked")
    return final


def to_data_uri_capped(fig, max_bytes: int = DATA_URI_MAX_BYTES) -> str:
    try:
        raw = b""
        for dpi in (120, 100, 90, 80, 72, 60, 50, 40, 30):
            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="png",
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="white",
                pad_inches=0.05,
            )
            plt.close(fig)
            buf.seek(0)
            raw = buf.read()
            if len(raw) <= max_bytes:
                b64 = base64.b64encode(raw).decode("ascii")
                return f"data:image/png;base64,{b64}"
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        # tiny 1x1 PNG
        return (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/"
            "PchI7wAAAABJRU5ErkJggg=="
        )


def extract_code(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"```python\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    return text.strip()

# --------------------------- Sandbox / Safety ---------------------------

ALLOWED_IMPORTS = {
    "math", "statistics",
    "json", "re", "csv",
    "datetime", "time",
    "pandas", "numpy",
    "matplotlib", "matplotlib.pyplot",
    "io", "itertools", "functools", "collections",
    "requests", "bs4",  # web scraping
}

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "pow": pow,
    "print": print,
    "range": range,
    "round": round,
    "sorted": sorted,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

BANNED_MODULES = {"os", "sys", "subprocess", "shutil", "pathlib", "socket", "pdb", "pickle", "builtins"}

# allow open (we replace it with a safe read-only version)
BANNED_NAMES = {
    "exec", "eval", "compile",
    "__import__", "__builtins__",
    "input", "exit", "quit",
}


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root in BANNED_MODULES:
        raise ImportError(f"Import of '{root}' is blocked")
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"Import of '{root}' is not allowed")
    return importlib.import_module(name)


def make_safe_open(allowed_dir: str):
    import builtins as _py_builtins

    def _safe_open(file, mode="r", *args, **kwargs):
        if any(c in mode for c in ("w", "a", "+")):
            raise PermissionError("Write access is blocked")
        fpath = file if isinstance(file, str) else getattr(file, "name", "")
        if not fpath:
            raise PermissionError("Anonymous file handles are blocked")
        real_target = safe_join(
            allowed_dir, os.path.relpath(os.path.realpath(fpath), allowed_dir)
        )
        return _py_builtins.open(real_target, mode, *args, **kwargs)

    return _safe_open


def deny_dangerous_ast(code: str):
    import ast
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise RuntimeError(f"Use of '{node.id}' is blocked")
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in BANNED_MODULES:
                raise RuntimeError(f"Access to '{node.value.id}.*' is blocked")
        if isinstance(node, ast.Import):
            for n in node.names:
                root = n.name.split(".")[0]
                if root in BANNED_MODULES or root not in ALLOWED_IMPORTS:
                    raise RuntimeError(f"Import '{n.name}' is not allowed")
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in BANNED_MODULES or root not in ALLOWED_IMPORTS:
                raise RuntimeError(f"Import from '{node.module}' is not allowed")

# --------------------------- LLM Orchestration --------------------------


def build_llm_prompt(question_text: str, csv_paths: List[str], attachment_map: Dict[str, str]) -> str:
    sys_prompt = (
        "You are a senior data analyst and Python engineer. "
        "Write ONLY Python code (no backticks, no prose). Your code must:\n"
        "1) Optionally load any provided CSVs from CSV_PATHS.\n"
        "2) If the question requires web data, you MAY fetch/scrape pages using 'requests' and 'bs4' "
        "(if available). If bs4 is unavailable, prefer pandas.read_html.\n"
        "3) Perform the analysis and answer the question(s).\n"
        "4) Put the final output in a variable named 'result' (dict, list, str, int, or float).\n"
        "5) For any chart, use matplotlib and return a Data URI via to_data_uri_capped(fig).\n"
        f"6) Keep any chart under {DATA_URI_MAX_BYTES} bytes.\n"
        "7) Do not read or write files outside CSV_PATHS/ATTACHMENTS. Do not print; just assign 'result'."
    )
    user_prompt = (
        f"QUESTION:\n{question_text}\n\n"
        f"CSV_PATHS = {json.dumps(csv_paths)}\n"
        f"ATTACHMENTS = {json.dumps(attachment_map)}\n\n"
        "Use pandas as 'pd', numpy as 'np', matplotlib.pyplot as 'plt'. "
        "Helper available: to_data_uri_capped(fig) -> str."
    )
    return sys_prompt, user_prompt


async def get_generated_code(question_text: str, csv_paths: List[str], attachment_map: Dict[str, str]) -> str:
    if not (OpenAI and OPENAI_KEY):
        return ""
    sys_prompt, user_prompt = build_llm_prompt(question_text, csv_paths, attachment_map)
    try:
        client = OpenAI(api_key=OPENAI_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.1,
            max_tokens=1800,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        return extract_code(text)
    except Exception as e:
        logger.warning("OpenAI generation failed: %s", e)
        return ""

# --------------------------- Code Execution -----------------------------


def _exec_under_timeout(code: str, g: Dict[str, Any], l: Dict[str, Any], seconds: int) -> Tuple[bool, Optional[str]]:
    try:
        exec(compile(code, "<generated>", "exec"), g, l)
        return True, None
    except Exception as e:
        return False, f"{e}"


async def execute_generated_code(
    code: str,
    csv_paths: List[str],
    question: str,
    attachments: Dict[str, str],
    request_dir: str,
) -> Union[Dict[str, Any], List[Any], str, int, float]:
    deny_dangerous_ast(code)

    builtins_sandbox = dict(SAFE_BUILTINS)
    builtins_sandbox["__import__"] = safe_import  # <-- ensures import statements work
    builtins_sandbox["open"] = make_safe_open(request_dir)  # read-only within request dir

    # Optional imports (don't fail if not installed)
    try:
        _requests = importlib.import_module("requests")
    except Exception:
        _requests = None
    try:
        _bs4 = importlib.import_module("bs4")
        from bs4 import BeautifulSoup as _BeautifulSoup
    except Exception:
        _bs4 = None
        _BeautifulSoup = None

    safe_globals: Dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "io": io,
        "base64": base64,
        "json": json,
        "requests": _requests,
        "bs4": _bs4,
        "BeautifulSoup": _BeautifulSoup,
        "to_data_uri_capped": to_data_uri_capped,
        "CSV_PATHS": csv_paths or [],
        "CSV_PATH": csv_paths[0] if csv_paths else None,
        "ATTACHMENTS": attachments or {},
        "QUESTION": question,
        "__builtins__": builtins_sandbox,
        "__name__": "__main__",
    }
    safe_locals: Dict[str, Any] = {}

    ok, err = await asyncio.get_event_loop().run_in_executor(
        None, _exec_under_timeout, code, safe_globals, safe_locals, EXEC_SLICE_SECONDS
    )
    if not ok:
        return {"error": f"execution_error: {err}"}

    result = safe_locals.get("result", safe_locals.get("RESULT"))
    if result is None:
        for k, v in safe_locals.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (dict, list, str, int, float)):
                return v
        return {"error": "no_result_produced"}
    return result

# --------------------------- Multipart Parsing --------------------------


async def parse_multipart_any(request: Request, work_dir: str) -> Tuple[str, List[str], Dict[str, str]]:
    form = await request.form()
    question_text = ""
    csv_paths: List[str] = []
    attachment_map: Dict[str, str] = {}
    saved_txt_files: List[str] = []

    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            field_name = str(key or "").lower()
            filename = value.filename or "unnamed"
            lower_name = filename.lower()

            fpath = safe_join(work_dir, filename)
            content = await value.read()
            with open(fpath, "wb") as out:
                out.write(content)

            is_questions_by_field = "question" in field_name
            is_questions_by_filename = (
                lower_name == "questions.txt"
                or lower_name == "question.txt"
                or "question" in lower_name
            )

            if is_questions_by_field or is_questions_by_filename:
                try:
                    question_text = content.decode("utf-8", errors="ignore")
                except Exception:
                    question_text = ""
                continue

            if lower_name.endswith(".csv"):
                csv_paths.append(fpath)
                attachment_map[filename] = fpath
            else:
                attachment_map[filename] = fpath

            if lower_name.endswith(".txt"):
                saved_txt_files.append(fpath)

    if not question_text and saved_txt_files:
        try:
            with open(saved_txt_files[0], "rb") as f:
                question_text = f.read().decode("utf-8", errors="ignore")
        except Exception:
            question_text = ""

    return question_text, csv_paths, attachment_map

# --------------------------- Fallbacks ----------------------------------


def generic_fallback_answer() -> Dict[str, Any]:
    return {
        "status": "ok",
        "note": "LLM unavailable or code generation failed; returned generic response.",
    }


def try_simple_csv_heuristics(question: str, csv_paths: List[str]) -> Dict[str, Any]:
    try:
        if not csv_paths:
            return generic_fallback_answer()
        df = pd.read_csv(csv_paths[0])
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return generic_fallback_answer()

        y = num_cols[0]
        out: Dict[str, Any] = {}
        q = question.lower()

        series = pd.to_numeric(df[y], errors="coerce").dropna()

        if "median" in q:
            out["median"] = float(series.median())
        if "total" in q or "sum" in q:
            out["total"] = float(series.sum())
        if "average" in q or "mean" in q:
            out["average"] = float(series.mean())

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(series.values)
        ax.set_title(f"{y} over index")
        ax.set_xlabel("index")
        ax.set_ylabel(y)
        out["chart"] = to_data_uri_capped(fig)

        return out or generic_fallback_answer()
    except Exception as e:
        logger.warning("CSV heuristic failed: %s", e)
        return generic_fallback_answer()

# --------------------------- Endpoints ----------------------------------


@app.post("/api/")
async def api(request: Request):
    start = time.time()
    req_id = uuid.uuid4().hex
    _active_requests.add(req_id)
    work_dir = unique_work_dir()

    try:
        question_text, csv_paths, attachment_map = await parse_multipart_any(request, work_dir)
        if not question_text:
            return JSONResponse({"error": "questions.txt missing"}, status_code=200)

        remaining = HARD_TIME_CAP_SECONDS - (time.time() - start)
        if remaining <= 5:
            return JSONResponse(generic_fallback_answer())

        code = ""
        try:
            code = await asyncio.wait_for(
                get_generated_code(question_text, csv_paths, attachment_map),
                timeout=min(30, max(5, int(remaining * 0.3))),
            )
        except asyncio.TimeoutError:
            code = ""

        if code:
            remaining = HARD_TIME_CAP_SECONDS - (time.time() - start)
            if remaining <= 5:
                return JSONResponse(generic_fallback_answer())

            try:
                result = await asyncio.wait_for(
                    execute_generated_code(
                        code=code,
                        csv_paths=csv_paths,
                        question=question_text,
                        attachments=attachment_map,
                        request_dir=work_dir,
                    ),
                    timeout=min(EXEC_SLICE_SECONDS, max(5, int(remaining) - 1)),
                )
                if isinstance(result, (dict, list, str, int, float)):
                    return JSONResponse(result)
                return JSONResponse({"answer": str(result)})
            except asyncio.TimeoutError:
                return JSONResponse(try_simple_csv_heuristics(question_text, csv_paths))
            except Exception as e:
                logger.warning("Execution failed: %s", e)
                return JSONResponse({"error": f"execution_error: {e}"})
        else:
            return JSONResponse(try_simple_csv_heuristics(question_text, csv_paths))

    except Exception as e:
        logger.exception("Top-level error")
        return JSONResponse({"error": f"{e}"}, status_code=200)
    finally:
        _active_requests.discard(req_id)
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
        gc.collect()


@app.post("/")
async def root(request: Request):
    return await api(request)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "openai_available": bool(OpenAI and OPENAI_KEY),
        "active_requests": len(_active_requests),
        "max_concurrent_hint": MAX_CONCURRENT,
        "time_cap_seconds": HARD_TIME_CAP_SECONDS,
        "data_uri_cap_bytes": DATA_URI_MAX_BYTES,
    }

# --------------------------- Local Dev Server ---------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
