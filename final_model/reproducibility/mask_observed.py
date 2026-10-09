#!/usr/bin/env python
"""mask_observed.py -- build publishable copies of result files that contain
values of the observed zinc record.

The observed record (``zinc_dataset.xlsx``, derived from ILZSG data) may not be
redistributed. Some result files store parts of it verbatim: the observed stocks
and flows saved next to each prediction, coefficients pinned to observed transfer
shares, the observed concentrate input, the exogenous driver paths, and model
trajectories that start from observed initial stocks. A few more hold quantities
computed directly from the observed series (ratios and differences).

For every file given on the command line this script

* finds the values that are copies of the loader output
  (``zinc_colloc_v5.load_zinc_data``: stocks_obs, flows_obs, cp_obs, alpha_obs,
  tau_sup_obs, exog_values_full);
* writes a copy to ``masked_results/<same relative path>`` in which those values
  are NaN (npz) or empty cells (csv);
* records, for every masked value, the array name and position in the loader
  output that it came from (never the value itself) in
  ``masked_results/restore_map.npz``, and a per-file summary with SHA-256 digests
  of the original arrays/files in ``masked_results/manifest.json``.

Files with nothing to mask are left alone (they are published as they are).
``restore_observed.py`` reverses the operation for anyone who holds the dataset.

Run from ``final_model/`` with the dataset in place::

    python reproducibility/mask_observed.py --xlsx zinc_dataset.xlsx FILE [FILE ...]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FM = Path(__file__).resolve().parents[1]
OUT = FM / "masked_results"
REF_KEYS = ("stocks_obs", "flows_obs", "cp_obs", "alpha_obs", "tau_sup_obs",
            "exog_values_full")

# Fields that are the observed record itself. Every element is masked when at
# least half of the field's values are copies of the loader output (this keeps
# the synthetic cycle's own 'alpha_obs', which is not real data, unmasked).
RAW_NAMES = {"stocks_obs", "flows_obs", "stocks_obs_testrun", "S_obs",
             "alpha_obs", "parent_stock_obs", "observed_stock_kt", "cp_obs"}

# Fields computed from the observed series but not verbatim copies of it.
# They are masked whole. A recipe name means restore_observed.py recomputes
# the field with the code that wrote it; None means the field is regenerated
# only by re-running the work package.
DERIVED = {
    "analysis/wp2g_saturation.csv": {
        c: "wp2g_saturation" for c in (
            "gap_obs_kt_median", "sat_ratio_obs_median", "sat_ratio_obs_hl",
            "sat_ratio_obs_lo", "sat_ratio_obs_hi")},
    # implied - cumulative imbalance = observed stock, so the imbalance columns
    # would give the observed stock back
    "analysis/wp1f_mass_imbalance.csv": {
        c: "wp1f_imbalance" for c in (
            "imbalance_kt", "cumulative_imbalance_kt", "observed_stock_pct")},
    "analysis/wp2a_A_of_t.npz": {"rhs_residual_obs": None},
}

# Bookkeeping fields where an equal number is a coincidence, not a copy.
SKIP_FIELD = re.compile(
    r"wall|^seeds?$|^index$|^n$|^n_|_n$|grid|^m$|^years?|future_t|^t$|^k$|"
    r"sobol_base_n",
    re.I)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def array_sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    if a.dtype.kind == "O":
        return sha(repr(a.tolist()).encode())
    return sha(a.tobytes())


def significant_digits(x: float) -> int:
    return len(f"{abs(x):.15g}".split("e")[0].replace(".", "").strip("0"))


def load_reference(xlsx: Path):
    sys.path.insert(0, str(FM))
    import zinc_colloc_v5 as z  # noqa: E402  (imports JAX)
    cfg = json.loads((FM / "anchor_v4" / "config_used.json").read_text())
    d = z.load_zinc_data(str(xlsx), extra_exog_cols=cfg.get("extra_exog_cols"))
    ref = {k: np.asarray(d[k], float).ravel() for k in REF_KEYS}
    full, strict = {}, {}
    for k in REF_KEYS:
        for i, x in enumerate(ref[k]):
            if not np.isfinite(x) or x == 0:
                continue
            full.setdefault(float(x), (k, i))
            if significant_digits(x) >= 3 and not (
                    float(x).is_integer() and 1900 <= x <= 2100):
                strict.setdefault(float(x), (k, i))
    return as_table(full), as_table(strict)


REL_TOL = 1e-12   # a copy may differ from the loader value by a few ULPs
                  # (e.g. a mean over 35 seeds of the same pinned value)


def as_table(table: dict):
    vals = np.array(sorted(table), float)
    keys = np.array([REF_KEYS.index(table[v][0]) for v in vals], np.int8)
    pos = np.array([table[v][1] for v in vals], np.int32)
    return vals, keys, pos


def find(values: np.ndarray, table):
    """Positions in ``values`` that equal a reference value to within REL_TOL.

    Returns the positions, the source array and position of each match, and
    the offset in units in the last place (ULPs) between the stored value and
    the reference value, so that restoration is bit-exact.
    """
    vals, keys, pos = table
    v = np.asarray(values, float)
    ok = np.isfinite(v) & (v != 0)
    j = np.clip(np.searchsorted(vals, v), 1, len(vals) - 1)
    left, right = vals[j - 1], vals[j]
    j = np.where(np.abs(v - left) <= np.abs(v - right), j - 1, j)
    near = vals[j]
    hit = ok & (np.abs(v - near) <= REL_TOL * np.abs(near))
    idx = np.flatnonzero(hit)
    ulp = (v[idx].view(np.int64) - near[idx].view(np.int64))
    if np.abs(ulp).max(initial=0) > 2**31 - 1:
        raise SystemExit("ULP offset out of range")
    return (idx.astype(np.int64), keys[j[idx]], pos[j[idx]],
            ulp.astype(np.int32))


def plan_field(rel, name, values, full, strict):
    """Return (mode, masked positions, mapped positions, (key, pos, ulp), recipe)."""
    values = np.asarray(values, float).ravel()
    finite = np.isfinite(values)
    if rel in DERIVED and name in DERIVED[rel]:
        return "derived", np.flatnonzero(finite), None, None, DERIVED[rel][name]
    if name in RAW_NAMES:
        idx, k, p, u = find(values, full)
        nonzero = finite & (values != 0)      # zeros carry no information
        if nonzero.sum() and len(idx) >= 0.5 * nonzero.sum():
            return "raw", np.flatnonzero(nonzero), idx, (k, p, u), None
    if SKIP_FIELD.search(str(name)):
        return None
    idx, k, p, u = find(values, strict)
    if len(idx):
        return "copies", idx, idx, (k, p, u), None
    return None


def write_atomic(dst: Path, writer) -> None:
    """Write through a temporary file and rename it into place."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as fh:
        writer(fh)
    os.replace(tmp, dst)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", type=Path,
                    help="result files, relative to final_model/")
    ap.add_argument("--xlsx", type=Path, default=FM / "zinc_dataset.xlsx")
    args = ap.parse_args(argv)

    full, strict = load_reference(args.xlsx)
    manifest, maps = {"reference_keys": list(REF_KEYS), "files": {}}, {}
    for rel_path in args.files:
        rel = rel_path.as_posix()
        src = FM / rel
        entry = {"fields": {}}
        if rel.endswith(".npz"):
            with np.load(src, allow_pickle=True) as d:
                arrays = {k: d[k] for k in d.files}
            out_arrays = dict(arrays)
            for name, a in arrays.items():
                if a.dtype.kind != "f":
                    continue
                plan = plan_field(rel, name, a, full, strict)
                if plan is None:
                    continue
                mode, masked_idx, map_idx, src_kp, recipe = plan
                b = a.copy().ravel()
                b[masked_idx] = np.nan
                out_arrays[name] = b.reshape(a.shape)
                entry["fields"][name] = {
                    "mode": mode, "n_masked": int(len(masked_idx)),
                    "recipe": recipe, "sha256": array_sha(a),
                    "dtype": str(a.dtype), "shape": list(a.shape)}
                if mode != "derived":
                    k, p, u = src_kp
                    tag = f"{rel}::{name}"
                    maps[tag + "::idx"] = map_idx
                    maps[tag + "::key"] = k
                    maps[tag + "::pos"] = p
                    maps[tag + "::ulp"] = u
                    entry["fields"][name]["n_mapped"] = int(len(map_idx))
            if not entry["fields"]:
                continue
            entry.update(kind="npz", keys=list(arrays))
            write_atomic(OUT / rel, lambda fh: np.savez_compressed(fh, **out_arrays))
        elif rel.endswith(".csv"):
            raw = src.read_bytes()
            text = raw.decode()
            rows = list(csv.reader(io.StringIO(text, newline="")))
            buf = io.StringIO(newline="")
            csv.writer(buf, lineterminator="\n").writerows(rows)
            if buf.getvalue() != text:
                raise SystemExit(f"{rel}: csv does not round-trip; mask by hand")
            df = pd.read_csv(src, low_memory=False, float_precision="round_trip")
            if len(df) != len(rows) - 1 or list(df.columns) != rows[0]:
                raise SystemExit(f"{rel}: header/row alignment check failed")
            for j, name in enumerate(df.columns):
                v = pd.to_numeric(df[name], errors="coerce").to_numpy(float)
                plan = plan_field(rel, name, v, full, strict)
                if plan is None:
                    continue
                mode, masked_idx, map_idx, src_kp, recipe = plan
                for i in masked_idx:
                    rows[i + 1][j] = ""
                entry["fields"][name] = {
                    "mode": mode, "n_masked": int(len(masked_idx)),
                    "recipe": recipe, "column": j}
                if mode != "derived":
                    k, p, u = src_kp
                    tag = f"{rel}::{name}"
                    maps[tag + "::idx"] = map_idx
                    maps[tag + "::key"] = k
                    maps[tag + "::pos"] = p
                    maps[tag + "::ulp"] = u
                    entry["fields"][name]["n_mapped"] = int(len(map_idx))
            if not entry["fields"]:
                continue
            entry.update(kind="csv", sha256=sha(raw))
            buf = io.StringIO(newline="")
            csv.writer(buf, lineterminator="\n").writerows(rows)
            write_atomic(OUT / rel, lambda fh: fh.write(buf.getvalue().encode()))
        else:
            continue
        manifest["files"][rel] = entry
        n = sum(f["n_masked"] for f in entry["fields"].values())
        print(f"masked {n:>7d} values in {rel}")
    write_atomic(OUT / "restore_map.npz", lambda fh: np.savez_compressed(fh, **maps))
    write_atomic(OUT / "manifest.json",
                 lambda fh: fh.write((json.dumps(manifest, indent=1) + "\n").encode()))
    print(f"{len(manifest['files'])} files masked -> {OUT.relative_to(FM)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
