"""Stage 2 - data validation.

Row-level rules split every table into valid rows (data/interim) and rejected rows
(data/errors, with the reason). Rules with severity "warning" are only logged and the
rows are fixed later in preparation. The surviving rows must then pass a pandera
schema, which acts as the contract for the next stages.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pandera.pandas as pa

from src.common.config import get_logger, path

log = get_logger("validation")
YEAR_RE = r"\(\d{4}\)\s*$"


class ValidationError(RuntimeError):
    pass


def build_schemas(ds: dict) -> dict:
    lo, hi = ds["rating_min"], ds["rating_max"]
    ts_lo = int(pd.Timestamp(ds["timestamp_min"]).timestamp())
    ts_hi = int(time.time())
    return {
        "users": pa.DataFrameSchema({"user_id": pa.Column(str, nullable=False, unique=True)}, strict=False),
        "items": pa.DataFrameSchema({"item_id": pa.Column(str, nullable=False, unique=True),
                                     "title": pa.Column(str, nullable=False)}, strict=False),
        "interactions": pa.DataFrameSchema({
            "user_id": pa.Column(str, nullable=False),
            "item_id": pa.Column(str, nullable=False),
            "rating": pa.Column(float, pa.Check.in_range(lo, hi), nullable=False),
            "timestamp": pa.Column("int64", pa.Check.in_range(ts_lo, ts_hi), nullable=False),
        }, strict=False),
    }


class _Collector:
    def __init__(self, table: str, df: pd.DataFrame, checks: list):
        self.table, self.df, self.checks = table, df, checks
        self.reasons = pd.Series("", index=df.index, dtype=object)
        self.flagged: set[int] = set()

    def add(self, mask: pd.Series, check: str, rule: str, severity: str = "error"):
        mask = mask.fillna(False).astype(bool)
        n = int(mask.sum())
        self.checks.append({"table": self.table, "check": check, "rule": rule, "severity": severity, "failed": n})
        if n:
            self.flagged.update(self.df.loc[mask, "raw_row"].tolist())
            if severity == "error":
                cur = self.reasons[mask]
                self.reasons[mask] = np.where(cur == "", check, cur + ";" + check)

    @property
    def bad(self) -> pd.Series:
        return self.reasons != ""


def _read(raw, name):
    df = pd.read_csv(raw / f"{name}.csv", dtype=str, keep_default_na=False, na_values=[""])
    df.insert(0, "raw_row", np.arange(len(df)))
    return df


def validate(cfg: dict) -> dict:
    ds, raw = cfg["ds"], path(cfg, "raw")
    interim, errors, reports = path(cfg, "interim"), path(cfg, "errors"), path(cfg, "reports")
    users, items, inter = _read(raw, "users"), _read(raw, "items"), _read(raw, "interactions")
    checks: list[dict] = []
    now = int(time.time())
    ts_lo = int(pd.Timestamp(ds["timestamp_min"]).timestamp())

    # users
    cu = _Collector("users", users, checks)
    cu.add(users["user_id"].isna(), "missing_user_id", "user_id must not be null")
    cu.add(users["user_id"].notna() & ~users["user_id"].str.fullmatch(r"\d+").fillna(True),
           "user_id_not_numeric", "user_id must be an integer string")
    cu.add(users["user_id"].notna() & users["user_id"].duplicated(keep="first"),
           "duplicate_user_id", "user_id must be unique (first copy kept)")

    # items
    ci = _Collector("items", items, checks)
    ci.add(items["item_id"].isna(), "missing_item_id", "item_id must not be null")
    ci.add(items["item_id"].notna() & ~items["item_id"].str.fullmatch(r"\d+").fillna(True),
           "item_id_not_numeric", "item_id must be an integer string")
    ci.add(items["item_id"].notna() & items["item_id"].duplicated(keep="first"),
           "duplicate_item_id", "item_id must be unique (first copy kept)")
    ci.add(items["title"].isna(), "missing_title", "title must not be null")
    ci.add(items["genres"].isna(), "missing_genres", "genres empty, filled with 'Unknown' later", "warning")
    ci.add(items["title"].notna() & ~items["title"].str.contains(YEAR_RE, regex=True).fillna(False)
           & (items["release_date"].isna() if "release_date" in items else True),
           "missing_release_year", "no year in title, imputed later", "warning")

    valid_users = users[~cu.bad].copy()
    valid_items = items[~ci.bad].copy()

    # interactions
    cx = _Collector("interactions", inter, checks)
    for col in ["user_id", "item_id", "rating", "timestamp"]:
        cx.add(inter[col].isna(), f"missing_{col}", f"{col} must not be null")
    for col in ["user_id", "item_id"]:
        cx.add(inter[col].notna() & ~inter[col].str.fullmatch(r"\d+").fillna(True),
               f"{col}_not_numeric", f"{col} must be an integer string")
    rating = pd.to_numeric(inter["rating"], errors="coerce")
    ts = pd.to_numeric(inter["timestamp"], errors="coerce")
    cx.add(inter["rating"].notna() & rating.isna(), "rating_not_numeric", "rating must be numeric (wrong data type)")
    cx.add(inter["timestamp"].notna() & (ts.isna() | (ts % 1 != 0)), "timestamp_not_integer",
           "timestamp must be an integer unix time (wrong data type)")
    cx.add(rating.notna() & ~rating.between(ds["rating_min"], ds["rating_max"]), "rating_out_of_range",
           f"rating must be in [{ds['rating_min']}, {ds['rating_max']}]")
    cx.add(ts.notna() & ~ts.between(ts_lo, now), "invalid_timestamp",
           f"timestamp must be between {ds['timestamp_min']} and now")
    cx.add(inter["user_id"].notna() & ~inter["user_id"].isin(valid_users["user_id"]),
           "orphan_user", "user_id must exist in users (referential integrity)")
    cx.add(inter["item_id"].notna() & ~inter["item_id"].isin(valid_items["item_id"]),
           "orphan_item", "item_id must exist in items (referential integrity)")
    order = ts.fillna(-1).sort_values(kind="stable").index
    dup_sorted = inter.loc[order].duplicated(["user_id", "item_id"], keep="last")
    cx.add(dup_sorted.reindex(inter.index) & inter["user_id"].notna() & inter["item_id"].notna(),
           "duplicate_user_item", "one rating per user and item; older copy dropped in preparation", "warning")

    # split valid / rejected
    tables = {}
    for name, col, df in (("users", cu, users), ("items", ci, items), ("interactions", cx, inter)):
        rejected = df[col.bad].assign(error_reason=col.reasons[col.bad])
        rejected.to_csv(errors / f"{name}_errors.csv", index=False)
        good = df[~col.bad].drop(columns="raw_row").reset_index(drop=True)
        tables[name] = {"rows_in": len(df), "rows_valid": len(good), "rows_rejected": len(rejected),
                        "reject_ratio": round(len(rejected) / max(len(df), 1), 5)}
        if tables[name]["reject_ratio"] > cfg["validation"]["max_error_ratio"]:
            raise ValidationError(f"{name}: {tables[name]['reject_ratio']:.1%} rejected, above the allowed limit")
        tables[name]["_frame"] = good

    good_u, good_i, good_x = (tables[n].pop("_frame") for n in ("users", "items", "interactions"))
    good_x["rating"] = good_x["rating"].astype(float)
    good_x["timestamp"] = good_x["timestamp"].astype("int64")
    schemas = build_schemas(ds)
    try:
        schemas["users"].validate(good_u, lazy=True)
        schemas["items"].validate(good_i, lazy=True)
        schemas["interactions"].validate(good_x, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise ValidationError(f"pandera contract failed:\n{exc.failure_cases.head(20)}") from exc
    good_u.to_parquet(interim / "users.parquet", index=False)
    good_i.to_parquet(interim / "items.parquet", index=False)
    good_x.to_parquet(interim / "interactions.parquet", index=False)

    # did the checks catch the defects that ingestion injected?
    catch = None
    fault_file = raw / "injected_faults.csv"
    if fault_file.exists() and fault_file.stat().st_size > 20:
        faults = pd.read_csv(fault_file)
        flagged = {"users": cu.flagged, "items": ci.flagged, "interactions": cx.flagged}
        faults["caught"] = [r in flagged[t] for t, r in zip(faults["table"], faults["raw_row"])]
        catch = {"injected": int(len(faults)), "caught": int(faults["caught"].sum()),
                 "by_kind": {k: {"injected": int(len(g)), "caught": int(g["caught"].sum())}
                             for k, g in faults.groupby("kind")}}

    report = {"dataset": cfg["dataset"]["name"], "tables": tables, "checks": checks, "injected_defect_check": catch}
    (reports / "validation_report.json").write_text(json.dumps(report, indent=2))
    _write_markdown(report, reports / "validation_report.md")
    log.info("validation done: %s", {k: (v["rows_valid"], v["rows_rejected"]) for k, v in tables.items()})
    if catch:
        log.info("injected defects caught: %d / %d", catch["caught"], catch["injected"])
    return report


def _write_markdown(report: dict, out) -> None:
    lines = [f"# Data validation report ({report['dataset']})", "", "## Rows per table", "",
             "| table | rows in | valid | rejected | reject ratio |", "|---|---:|---:|---:|---:|"]
    for t, v in report["tables"].items():
        lines.append(f"| {t} | {v['rows_in']:,} | {v['rows_valid']:,} | {v['rows_rejected']:,} | {v['reject_ratio']:.2%} |")
    lines += ["", "## Checks", "", "| table | check | severity | failed rows | rule |", "|---|---|---|---:|---|"]
    for c in report["checks"]:
        lines.append(f"| {c['table']} | `{c['check']}` | {c['severity']} | {c['failed']:,} | {c['rule']} |")
    lines += ["", "Severity `error`: row moved to `data/errors/`. Severity `warning`: row kept, fixed in preparation.", ""]
    c = report.get("injected_defect_check")
    if c:
        lines += ["## Injected defects", "",
                  "Ingestion adds a small logged set of defects (`data/raw/injected_faults.csv`) to prove the checks work.",
                  "", f"Caught {c['caught']} of {c['injected']}.", "", "| defect | injected | caught |", "|---|---:|---:|"]
        for k, v in c["by_kind"].items():
            lines.append(f"| {k} | {v['injected']} | {v['caught']} |")
    out.write_text("\n".join(lines) + "\n")
