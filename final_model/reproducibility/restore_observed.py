#!/usr/bin/env python
"""restore_observed.py -- rebuild the full result files from the masked copies.

The published repository holds, under ``masked_results/``, copies of the result
files that contain values of the observed zinc record, with those values removed
(see ``mask_observed.py``). The observed record itself (``zinc_dataset.xlsx``,
derived from ILZSG data) is not redistributable.

With the dataset::

    python reproducibility/restore_observed.py --xlsx zinc_dataset.xlsx

reads the dataset with the model's own loader, puts every removed value back
(bit-exact, using the positions stored in ``masked_results/restore_map.npz``),
recomputes the few derived columns that have a recipe, checks each rebuilt array
or file against the SHA-256 digest of the original, and writes the files to their
original locations (``anchor_v4/``, ``anchor_gam/``, ``analysis/``), where every
script and the figure builder expect them.

Without the dataset::

    python reproducibility/restore_observed.py --without-data

copies the masked files into place unchanged, so that the scripts run; the
observed points are then simply missing from the figures.

Existing files are never overwritten unless ``--force`` is given.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

FM = Path(__file__).resolve().parents[1]
MASKED = FM / "masked_results"
FIRST = ["analysis/wp2a_A_of_t.npz"]          # recipes read the restored A(t)
DEST = FM                                      # set from --dest in main()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def array_sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    if a.dtype.kind == "O":
        return sha(repr(a.tolist()).encode())
    return sha(a.tobytes())


def write_atomic(dst: Path, writer) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as fh:
        writer(fh)
    os.replace(tmp, dst)


def load_reference(xlsx: Path, keys):
    sys.path.insert(0, str(FM))
    import zinc_colloc_v5 as z  # noqa: E402  (imports JAX)
    cfg = json.loads((FM / "anchor_v4" / "config_used.json").read_text())
    d = z.load_zinc_data(str(xlsx), extra_exog_cols=cfg.get("extra_exog_cols"))
    return [np.asarray(d[k], float).ravel() for k in keys]


def values_from(ref, maps, tag):
    key, pos, ulp = (maps[tag + "::key"], maps[tag + "::pos"], maps[tag + "::ulp"])
    base = np.array([ref[k][p] for k, p in zip(key, pos)], float)
    return maps[tag + "::idx"], (base.view(np.int64) + ulp).view(np.float64)


# --------------------------------------------------------------------------- #
# Recipes for derived columns (recomputed with the code that wrote them)
# --------------------------------------------------------------------------- #
def recipe_wp2g_saturation(rows, columns):
    sys.path.insert(0, str(FM))
    import run_wp2g  # noqa: E402
    d = np.load(DEST / "analysis" / "wp2a_A_of_t.npz", allow_pickle=True)
    sat = run_wp2g.part12(np.asarray(d["A"], float), np.asarray(d["b"], float),
                          np.asarray(d["S_obs"], float),
                          np.asarray(d["S_pred"], float),
                          np.asarray(d["years"], float))[0]
    header = rows[0]
    if len(sat) != len(rows) - 1:
        raise RuntimeError("wp2g_saturation: row count differs")
    for name, j in columns.items():
        for i, x in enumerate(sat[name].to_numpy(float)):
            rows[i + 1][j] = repr(float(x))
    return rows


def recipe_wp1f_imbalance(rows, columns):
    """Imbalance columns of wp1f_mass_imbalance.csv from the implied and the
    observed stock (implied - observed = cumulative imbalance). Equal to the
    originals to rounding error, not bit for bit."""
    h = rows[0]
    c = {n: h.index(n) for n in ("model", "stock", "year", "implied_stock_kt",
                                 "observed_stock_kt")}
    prev = {}
    for r in rows[1:]:
        g = (r[c["model"]], r[c["stock"]])
        cum = float(r[c["implied_stock_kt"]]) - float(r[c["observed_stock_kt"]])
        obs = float(r[c["observed_stock_kt"]])
        vals = {"cumulative_imbalance_kt": cum,
                "imbalance_kt": cum - prev.get(g, 0.0),
                "observed_stock_pct": 100.0 * cum / max(abs(obs), 1e-12)}
        prev[g] = cum
        for name, j in columns.items():
            r[j] = repr(float(vals[name]))
    return rows


def recipe_wp1c_log(rows):
    """wp1c_by_year.csv repeats alpha_obs on the log scale (space == 'log')."""
    h = rows[0]
    c = {n: h.index(n) for n in ("channel", "space", "stage", "year", "split",
                                 "alpha_obs")}
    key = lambda r: tuple(r[c[n]] for n in ("channel", "stage", "year", "split"))  # noqa: E731
    lin = {key(r): r[c["alpha_obs"]] for r in rows[1:]
           if r[c["space"]] != "log" and r[c["alpha_obs"]] != ""}
    for r in rows[1:]:
        if r[c["space"]] == "log" and r[c["alpha_obs"]] == "" and key(r) in lin:
            r[c["alpha_obs"]] = repr(float(np.log(float(lin[key(r)]))))
    return rows


RECIPES = {"wp2g_saturation": recipe_wp2g_saturation,
           "wp1f_imbalance": recipe_wp1f_imbalance}
FILE_RECIPES = {"analysis/wp1c_by_year.csv": recipe_wp1c_log}


def restore_file(rel, entry, ref, maps):
    """Return (writer, report lines) for one file."""
    report, ok = [], True
    src = MASKED / rel
    if entry["kind"] == "npz":
        with np.load(src, allow_pickle=True) as d:
            arrays = {k: d[k] for k in entry["keys"]}
        for name, f in entry["fields"].items():
            a = arrays[name].astype(f["dtype"]).ravel().copy()
            if f["mode"] != "derived":
                idx, vals = values_from(ref, maps, f"{rel}::{name}")
                a[idx] = vals
            a = a.reshape(f["shape"])
            arrays[name] = a
            good = array_sha(a) == f["sha256"]
            ok &= good
            if not good:
                missing = int(np.isnan(a).sum())
                report.append(f"    {name}: not restored exactly "
                              f"({missing} values left missing)")
        writer = lambda fh: np.savez_compressed(fh, **arrays)  # noqa: E731
        return writer, ok, report
    text = src.read_text()
    rows = list(csv.reader(io.StringIO(text, newline="")))
    recipes = {}
    for name, f in entry["fields"].items():
        j = f["column"]
        if f["mode"] == "derived":
            if f["recipe"]:
                recipes.setdefault(f["recipe"], {})[name] = j
            continue
        idx, vals = values_from(ref, maps, f"{rel}::{name}")
        for i, x in zip(idx, vals):
            rows[i + 1][j] = repr(float(x))
    for recipe, cols in recipes.items():
        rows = RECIPES[recipe](rows, cols)
    if rel in FILE_RECIPES:
        rows = FILE_RECIPES[rel](rows)
    buf = io.StringIO(newline="")
    csv.writer(buf, lineterminator="\n").writerows(rows)
    out = buf.getvalue().encode()
    ok = sha(out) == entry["sha256"]
    if not ok:
        unrestored = [n for n, f in entry["fields"].items()
                      if f["mode"] == "derived" and not f["recipe"]]
        report.append("    not byte-identical to the original"
                      + (f"; columns regenerated only by re-running the work "
                         f"package: {', '.join(unrestored)}" if unrestored else ""))
    return (lambda fh: fh.write(out)), ok, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--xlsx", type=Path, help="path to zinc_dataset.xlsx")
    g.add_argument("--without-data", action="store_true",
                   help="copy the masked files into place unchanged")
    ap.add_argument("--force", action="store_true",
                    help="overwrite files that already exist")
    ap.add_argument("--dest", type=Path, default=FM,
                    help="root to write into (default: final_model/)")
    args = ap.parse_args(argv)
    global DEST
    DEST = args.dest

    manifest = json.loads((MASKED / "manifest.json").read_text())
    files = manifest["files"]
    order = [f for f in FIRST if f in files] + [f for f in files if f not in FIRST]

    if args.without_data:
        for rel in order:
            dst = args.dest / rel
            if dst.exists() and not args.force:
                print(f"exists, kept  {rel}")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(MASKED / rel, dst)
            print(f"copied (masked) {rel}")
        return 0

    ref = load_reference(args.xlsx, manifest["reference_keys"])
    maps = np.load(MASKED / "restore_map.npz")
    n_exact = 0
    for rel in order:
        dst = args.dest / rel
        if dst.exists() and not args.force:
            print(f"exists, kept  {rel}")
            continue
        writer, ok, report = restore_file(rel, files[rel], ref, maps)
        write_atomic(dst, writer)
        n_exact += ok
        print(f"{'restored' if ok else 'PARTIAL '}  {rel}")
        for line in report:
            print(line)
    print(f"{n_exact} of {len(order)} files identical to the originals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
