#!/usr/bin/env python3
"""CI gate: every mart_value_index.json enum value must map to an ontology entity.
Usage: check_value_index_coverage.py [--mvi PATH] [--db PATH] [--allowlist PATH] [--out DIR]
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, ".."))
ML_ENG = os.path.abspath(os.path.join(PKG, ".."))


def _env(n: str, d: str) -> str:
    return os.environ.get(n, d).strip() or d


ap = argparse.ArgumentParser()
ap.add_argument(
    "--mvi",
    default=_env(
        "ONTOLOGY_VALUE_INDEX",
        os.path.join(ML_ENG, "ml", "rag", "value_index", "mart_value_index.json"),
    ),
)
ap.add_argument(
    "--db",
    default=os.path.join(_env("ONTOLOGY_DATA_DIR", os.path.join(PKG, "data")), "ontology.sqlite"),
)
ap.add_argument(
    "--allowlist",
    default=os.path.join(HERE, "curated", "value_index_allowlist.csv"),
)
ap.add_argument(
    "--out",
    default=os.path.join(_env("ONTOLOGY_DATA_DIR", os.path.join(PKG, "data")), "reports"),
)
args = ap.parse_args()
MVI, DB, ALLOW, OUT = args.mvi, args.db, args.allowlist, args.out


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    db = sqlite3.connect(DB)
    errs: list[str] = []
    h = hashlib.sha256(open(MVI, "rb").read()).hexdigest()
    built = dict(db.execute("SELECT k,v FROM meta")).get("value_index_sha256")
    if built != h:
        errs.append(
            f"ontology.sqlite built from another value index (built={built}, given={h}); "
            "rerun build_value_map.py"
        )
    enums = json.load(open(MVI, encoding="utf-8")).get("enums", {})
    vm = {
        (f"{t}.{c}", v): (e, mt, cl)
        for e, t, c, v, mt, cl in db.execute(
            'SELECT entity_id,"table","column",stored_value,match_type,col_class FROM value_map'
        )
    }
    allow: set[tuple[str, str]] = set()
    if os.path.exists(ALLOW):
        for r in csv.DictReader(open(ALLOW, encoding="utf-8")):
            allow.add((r["column_key"], r["stored_value"]))
    stats: collections.OrderedDict = collections.OrderedDict()
    unmapped: list = []
    allowed: list = []
    for ck in sorted(enums):
        s = stats.setdefault(ck, collections.Counter())
        for v in map(str, enums[ck]):
            s["total"] += 1
            if (ck, v) not in vm:
                s["missing_row"] += 1
                unmapped.append((ck, v, "", "no value_map row"))
                continue
            e, mt, cl = vm[(ck, v)]
            s["class"] = cl
            if not mt.startswith("minted"):
                s["ontology"] += 1
            elif cl in ("dimension", "structural"):
                s["minted"] += 1
            elif (ck, v) in allow or ("*", v) in allow:
                s["allowlisted"] += 1
                allowed.append((ck, v, e))
            else:
                s["unmapped"] += 1
                unmapped.append((ck, v, e, mt))
    with open(f"{OUT}/value_index_coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "column_key",
                "col_class",
                "total",
                "ontology",
                "minted",
                "allowlisted",
                "unmapped",
                "ontology_pct",
            ]
        )
        for ck, s in stats.items():
            w.writerow(
                [
                    ck,
                    s["class"],
                    s["total"],
                    s["ontology"],
                    s["minted"],
                    s["allowlisted"],
                    s["unmapped"] + s["missing_row"],
                    round(100 * s["ontology"] / s["total"], 1) if s["total"] else 0,
                ]
            )
    T: collections.Counter = collections.Counter()
    for s in stats.values():
        T.update({k: v for k, v in s.items() if isinstance(v, int)})
    byc: collections.defaultdict = collections.defaultdict(collections.Counter)
    for s in stats.values():
        byc[s["class"]].update({k: v for k, v in s.items() if k != "class"})
    L = [
        "# mart_value_index coverage",
        "",
        f"value index sha256 `{h[:16]}`; {len(stats)} columns, {T['total']} values.",
        "",
        "| class | values | ontology | minted (OT:MV) | allowlisted | unmapped |",
        "|---|---|---|---|---|---|",
    ]
    for cl in ("country", "entity", "dimension", "structural"):
        c = byc[cl]
        L.append(
            f"| {cl} | {c['total']} | {c['ontology']} | {c['minted']} | "
            f"{c['allowlisted']} | {c['unmapped'] + c['missing_row']} |"
        )
    L.append(
        f"| **all** | {T['total']} | {T['ontology']} | {T['minted']} | "
        f"{T['allowlisted']} | {T['unmapped'] + T['missing_row']} |"
    )
    L += ["", "## Unmapped (CI failures)", ""] + (
        [f"- `{ck}` = `{v}`" for ck, v, *_ in unmapped] or ["none"]
    )
    L += [
        "",
        "## Allowlisted (known gaps, see ontology_build/curated/value_index_allowlist.csv)",
        "",
    ] + [f"- `{ck}` = `{v}` -> {e}" for ck, v, e in allowed]
    L += [
        "",
        "## Per column",
        "",
        "| column | class | total | ontology | minted | allowlisted | unmapped |",
        "|---|---|---|---|---|---|---|",
    ]
    L += [
        f"| {ck} | {s['class']} | {s['total']} | {s['ontology']} | {s['minted']} | "
        f"{s['allowlisted']} | {s['unmapped'] + s['missing_row']} |"
        for ck, s in stats.items()
    ]
    open(f"{OUT}/value_index_coverage.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(
        f"values={T['total']} ontology={T['ontology']} minted={T['minted']} "
        f"allowlisted={T['allowlisted']} unmapped={T['unmapped'] + T['missing_row']}"
    )
    for ck, v, *_ in unmapped:
        errs.append(f"UNMAPPED {ck} = {v!r}")
    if errs:
        print("\n".join(errs), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
