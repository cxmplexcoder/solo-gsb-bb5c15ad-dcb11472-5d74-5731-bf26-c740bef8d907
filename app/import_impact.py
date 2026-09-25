"""导入影响预演 (import impact preview).

Before confirming a previewed CSV upload, the user can preview *what would
change* for recent matching traffic: the most recent 100 distinct raw
queries (taken from the saved match logs, newest first by log time) are
re-scored twice with the default rules and the existing review-consensus
logic -- once against the **current** standard address table and once
against an in-memory **merged** table containing temporary versions of the
voucher's valid rows, normalised exactly as the real import would normalise
them (``ingest.planned_address_fields``).

Invalid and duplicate rows never participate: only rows classified
``valid`` by the preview become temporary addresses.  Every temporary
address keeps the CSV physical line number it came from (line 1 is the
header), so a changed auto-match target can be traced back to the upload.

Each query is classified by comparing its before/after verdict:

* ``improved``   改善 - the status moves up the ladder
  (``no_match`` < ``needs_review`` < ``auto_matched``);
* ``regressed``  退化 - the status moves down that ladder;
* ``retargeted`` 目标改派 - both runs auto-match, but on different targets;
* ``unchanged``  不变 - everything else.

The preview is strictly read-only: nothing is inserted into the address
table (the temporary rows live only in memory) and no match log is
written.  The voucher must still exist, must not have expired and the
address-table snapshot bound to it must still match the live table;
otherwise the caller is told to re-preview, exactly like the confirm path.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from . import consensus, db, ingest
from .config import Settings
from .matcher import match_address, result_to_dict

# How many distinct recent raw queries participate in one preview run.
RECENT_QUERY_LIMIT = 100

CATEGORY_IMPROVED = "improved"
CATEGORY_REGRESSED = "regressed"
CATEGORY_RETARGETED = "retargeted"
CATEGORY_UNCHANGED = "unchanged"

VALID_CATEGORIES = (
    CATEGORY_IMPROVED,
    CATEGORY_REGRESSED,
    CATEGORY_RETARGETED,
    CATEGORY_UNCHANGED,
)

CATEGORY_LABELS = {
    CATEGORY_IMPROVED: "改善",
    CATEGORY_REGRESSED: "退化",
    CATEGORY_RETARGETED: "目标改派",
    CATEGORY_UNCHANGED: "无变化",
}

STATUS_LABELS = {
    "auto_matched": "自动命中",
    "needs_review": "待确认",
    "no_match": "无匹配",
}

# Status quality ladder used to decide improvement vs regression.
_STATUS_RANK = {"no_match": 0, "needs_review": 1, "auto_matched": 2}


class ImpactPreviewExpired(Exception):
    """The preview token does not exist or has expired."""


class ImpactTableChanged(Exception):
    """The address table changed since the preview was created."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _temporary_address(line_no: int, address_id: int,
                       planned_fields: dict[str, Any]) -> dict[str, Any]:
    """Build an in-memory address row shaped exactly like an ``addresses`` row.

    ``match_address`` only reads these columns, and the token fields are the
    ones ``ingest.planned_address_fields`` computed with the same
    normalisation the confirm-time import uses -- so scoring against this
    row is indistinguishable from scoring against a really imported one.
    """
    fields = planned_fields["fields"]
    return {
        "id": address_id,
        "full_text": planned_fields["full_text"],
        "tokens_json": json.dumps(planned_fields["tokens"], ensure_ascii=False),
        "street_tokens_json": json.dumps(
            planned_fields["street_tokens"], ensure_ascii=False),
        "locality_tokens_json": json.dumps(
            planned_fields["locality_tokens"], ensure_ascii=False),
        "house": planned_fields["house"],
        "line1": fields.get("line1"),
        "city": fields.get("city"),
        "state": fields.get("state"),
        "postal_code": fields.get("postal_code"),
        "country": fields.get("country"),
        "external_id": fields.get("external_id"),
        "raw_json": json.dumps(planned_fields["raw"], ensure_ascii=False),
        # The synthetic id is never displayed on its own; the CSV line is
        # the identity users recognise from the preview panel.
        "csv_line": line_no,
        "is_new": True,
    }


def _recent_distinct_queries(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """The most recent 100 distinct raw queries, newest log first.

    Logs are scanned by descending id (the log-time order, ids are
    monotonic) and the first occurrence of each raw query text wins, so
    each distinct query carries the time of its most recent log.
    """
    rows = conn.execute(
        """
        SELECT id, query, created_at FROM match_logs
         ORDER BY id DESC
        """
    ).fetchall()
    seen: set[str] = set()
    queries: list[dict[str, Any]] = []
    for row in rows:
        if row["query"] in seen:
            continue
        seen.add(row["query"])
        queries.append({
            "query": row["query"],
            "last_match_id": row["id"],
            "last_seen_at": row["created_at"],
        })
        if len(queries) >= RECENT_QUERY_LIMIT:
            break
    return queries


def _run_once(query: str, rows: list[Any], settings: Settings,
              conn: sqlite3.Connection) -> dict[str, Any]:
    """Score + review-consensus pass against one address-row snapshot."""
    result = match_address(query, rows, settings)
    # Same post-scoring step the regular /api/match endpoints run: the
    # reviewed history is consulted live and may redirect a needs_review
    # verdict to a vouched-for consensus target.
    verdict = consensus.lookup_consensus(conn, query)
    consensus.apply_consensus(result, verdict)
    return result_to_dict(result)


def _candidate_by_id(match: dict[str, Any],
                     address_id: int | None) -> dict[str, Any] | None:
    if address_id is None:
        return None
    for cand in match.get("candidates", []):
        if cand["address"]["id"] == address_id:
            return cand
    return None


def _annotate_new_addresses(match: dict[str, Any],
                            new_ids: dict[int, int]) -> None:
    """Tag every candidate that is one of the temporary new addresses."""
    for cand in match.get("candidates", []):
        address_id = cand["address"]["id"]
        if address_id in new_ids:
            cand["address"]["is_new"] = True
            cand["address"]["csv_line"] = new_ids[address_id]


def _snapshot(match: dict[str, Any], new_ids: dict[int, int]) -> dict[str, Any]:
    """Compact before/after view used by the summary table and filters."""
    best_id = match.get("best_id")
    candidate = _candidate_by_id(match, best_id)
    return {
        "status": match["status"],
        "status_label": STATUS_LABELS.get(match["status"], match["status"]),
        "reason": match["reason"],
        "best_id": best_id,
        "best_score": match.get("best_score"),
        "best_full_text": (
            candidate["address"]["full_text"] if candidate else None),
        "best_is_new": best_id in new_ids if best_id is not None else False,
        "best_csv_line": new_ids.get(best_id) if best_id is not None else None,
        "consensus_applied": bool(
            (match.get("consensus") or {}).get("applied")),
        "match": match,
    }


def _target_text(view: dict[str, Any]) -> str:
    if view["best_id"] is None:
        return "无目标"
    if view["best_is_new"]:
        return f"CSV 第 {view['best_csv_line']} 行新地址"
    return f"#{view['best_id']}"


def _classify(before: dict[str, Any], after: dict[str, Any]) -> tuple[str, str]:
    before_rank = _STATUS_RANK[before["status"]]
    after_rank = _STATUS_RANK[after["status"]]
    if after_rank > before_rank:
        reason = (
            f"状态改善：{before['status_label']} → {after['status_label']}"
            f"（自动命中目标：{_target_text(after)}）"
        )
        return CATEGORY_IMPROVED, reason
    if after_rank < before_rank:
        reason = (
            f"状态退化：{before['status_label']} → {after['status_label']}"
        )
        return CATEGORY_REGRESSED, reason
    if (
        before["status"] == "auto_matched"
        and after["status"] == "auto_matched"
        and before["best_id"] != after["best_id"]
    ):
        reason = (
            f"两次均自动命中但目标不同：{_target_text(before)} → "
            f"{_target_text(after)}"
        )
        return CATEGORY_RETARGETED, reason
    reason = f"前后均为{before['status_label']}，判定不变"
    return CATEGORY_UNCHANGED, reason


def compute_impact(conn: sqlite3.Connection, settings: Settings,
                   token: str) -> dict[str, Any]:
    """Re-run recent queries against the current and the merged table.

    Raises ``ImpactPreviewExpired`` when the voucher is missing or expired
    and ``ImpactTableChanged`` when the live address table no longer matches
    the snapshot bound to the voucher.  Both mean the client must re-preview.
    Nothing is written: temporary addresses exist only in memory and no
    match log is inserted or modified.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    voucher = db.get_upload_preview(conn, token)
    if voucher is None:
        raise ImpactPreviewExpired("预览凭证不存在，须重新预览")
    if voucher["expires_at"] < now:
        raise ImpactPreviewExpired("预览已过期，须重新预览")
    if db.address_table_snapshot(conn) != voucher["table_snapshot"]:
        raise ImpactTableChanged("标准地址表在预览后发生变化，须重新预览")

    current_rows = db.all_addresses(conn)
    max_id = max((int(row["id"]) for row in current_rows), default=0)

    # Build the temporary merged table from the voucher's valid rows, using
    # the exact normalisation the confirm-time import writes with.  The
    # synthetic ids continue the current id sequence, mirroring the ids the
    # rows would receive if the import were confirmed right now.
    new_ids: dict[int, int] = {}
    new_addresses: list[dict[str, Any]] = []
    temporary_rows: list[dict[str, Any]] = []
    for offset, (line_no, row) in enumerate(
        json.loads(voucher["valid_rows_json"]), start=1
    ):
        planned = ingest.PlannedRow(line_no=line_no, status="valid", row=row)
        planned_fields = ingest.planned_address_fields(planned)
        address_id = max_id + offset
        new_ids[address_id] = line_no
        temp_row = _temporary_address(line_no, address_id, planned_fields)
        temporary_rows.append(temp_row)
        new_addresses.append({
            "id": address_id,
            "csv_line": line_no,
            "full_text": planned_fields["full_text"],
        })
    merged_rows = [dict(row) for row in current_rows] + temporary_rows

    queries = _recent_distinct_queries(conn)
    items: list[dict[str, Any]] = []
    for entry in queries:
        query = entry["query"]
        before_match = _run_once(query, current_rows, settings, conn)
        after_match = _run_once(query, merged_rows, settings, conn)
        _annotate_new_addresses(before_match, new_ids)
        _annotate_new_addresses(after_match, new_ids)
        before = _snapshot(before_match, new_ids)
        after = _snapshot(after_match, new_ids)
        category, reason = _classify(before, after)
        items.append({
            "query": query,
            "last_match_id": entry["last_match_id"],
            "last_seen_at": entry["last_seen_at"],
            "category": category,
            "category_label": CATEGORY_LABELS[category],
            "reason": reason,
            "before": before,
            "after": after,
        })

    counts = {
        cat: sum(1 for it in items if it["category"] == cat)
        for cat in VALID_CATEGORIES
    }
    return {
        "generated_at": _now(),
        "preview_token": token,
        "filename": voucher["filename"],
        "valid": voucher["valid_count"],
        "duplicates": voucher["duplicate_count"],
        "invalid": voucher["invalid_count"],
        "invalid_lines": json.loads(voucher["invalid_lines"]),
        "new_address_count": len(new_addresses),
        "new_addresses": new_addresses,
        "query_limit": RECENT_QUERY_LIMIT,
        "total_queries": len(items),
        **counts,
        "thresholds": {
            "auto_threshold": settings.auto_threshold,
            "margin_threshold": settings.margin_threshold,
            "reject_threshold": settings.reject_threshold,
            "token_weight": settings.token_weight,
        },
        "items": items,
    }
