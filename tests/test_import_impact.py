"""Tests for the import impact preview
(``GET /api/addresses/preview/{token}/impact``).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import db  # noqa: E402
from app.main import app, get_conn  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --- CSV fixtures -----------------------------------------------------------

# All scenario addresses carry the distinctive "Zxqwv" sentinel so they can
# never collide with sample rows or rows inserted by other test modules that
# share the throwaway database.  Physical lines:
#   2  IMP-IMP-1  93 Zxqwv Improved Way ...    (valid: covers a no_match query)
#   3  IMP-DUP    123 Main Street ...          (dup: full_text == sample STD-001)
#   4  ,,,                                     (invalid blank row)
#   5  IMP-TIE    91 zxqwv regression rd ...   (valid: normalises identically
#                                               to the regression seed -> tie)
#   6  IMP-RET    92 Zxqwv Retarget Road North (valid: strictly better match
#                                               than the retarget seed)
#   7  IMP-EXTRA  96 Zxqwv Extra Court ...     (valid: unrelated extra row)
#   8  IMP-IMP-1  93 Zxqwv Improved Way ...    (in-file repeat of line 2)
_IMPACT_CSV = (
    "external_id,address,city,state,postal_code\n"
    "IMP-IMP-1,93 Zxqwv Improved Way,Improvedton,IL,98071\n"
    "IMP-DUP,123 Main Street,New York,NY,10001\n"
    ",,,\n"
    "IMP-TIE,91 zxqwv regression rd,Regressionton,IL,98051\n"
    "IMP-RET,92 Zxqwv Retarget Road North,Retargetton,IL,98061\n"
    "IMP-EXTRA,96 Zxqwv Extra Court,Extraton,IL,98091\n"
    "IMP-IMP-1,93 Zxqwv Improved Way,Improvedton,IL,98071\n"
).encode()


def _seed_address(client, external_id: str, address: str) -> int:
    r = client.post("/api/addresses", json={
        "addresses": [{"external_id": external_id, "address": address}],
    })
    assert r.status_code == 200, r.text
    assert r.json()["inserted"] == 1, r.text
    return next(
        a["id"] for a in client.get("/api/addresses?limit=500").json()["items"]
        if a["external_id"] == external_id
    )


def _preview(client, csv_bytes: bytes = _IMPACT_CSV, name: str = "impact.csv"):
    r = client.post("/api/addresses/preview",
                    files={"file": (name, csv_bytes, "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()


def _match_save(client, query: str) -> dict:
    r = client.post("/api/match", json={"query": query, "save": True})
    assert r.status_code == 200, r.text
    return r.json()


def _review(client, match_id: int, action: str, selected_id: int | None = None):
    body = {"action": action}
    if selected_id is not None:
        body["selected_id"] = selected_id
    r = client.post(f"/api/matches/{match_id}/review", json=body)
    assert r.status_code == 200, r.text


def _impact(client, token: str):
    return client.get(f"/api/addresses/preview/{token}/impact")


def _item_for(data: dict, query: str) -> dict:
    matches = [it for it in data["items"] if it["query"] == query]
    assert len(matches) == 1, [it["query"] for it in data["items"]]
    return matches[0]


# --- tests ------------------------------------------------------------------


def test_impact_categories_and_read_only(client):
    conn = get_conn()
    addresses_before = db.count_addresses(conn)
    logs_before = db.count_match_logs(conn)

    # improved: a no_match query that the line-2 new address will cover.
    q_improved = "93 Zxqwv Improved Way, Improvedton IL 98071"
    imp = _match_save(client, q_improved)
    assert imp["status"] == "no_match", imp

    # regressed: seed the fully-written address; the line-5 CSV row is its
    # case/abbreviation twin with a different external_id, so the merged
    # table ties two identical-score rows at the top -> needs_review.
    reg_seed_id = _seed_address(
        client, "IMP-SEED-REG",
        "91 Zxqwv Regression Road, Regressionton, IL 98051",
    )
    q_regressed = "91 Zxqwv Regression Rd, Regressionton IL"
    reg = _match_save(client, q_regressed)
    assert reg["status"] == "auto_matched"
    assert reg["best_id"] == reg_seed_id

    # retargeted: the seed auto-matches the query, but the line-6 new
    # address carries the query's directional token "N" and postcode, so
    # after the merge both auto-match and the new address wins.
    ret_seed_id = _seed_address(
        client, "IMP-SEED-RET",
        "92 Zxqwv Retarget Road, Retargetton, IL",
    )
    q_retargeted = "92 Zxqwv Retarget Road N, Retargetton IL 98061"
    ret = _match_save(client, q_retargeted)
    assert ret["status"] == "auto_matched"
    assert ret["best_id"] == ret_seed_id

    # unchanged via review consensus: both before/after runs go through the
    # existing review-consensus logic.  A seeded address with a wrong house
    # number leaves the matching queries in needs_review; two differently
    # written logs with the same normalised key are reviewed onto the
    # seeded address, so subsequent matches auto-redirect there both with
    # and without the temporary merged rows.
    con_seed_id = _seed_address(
        client, "IMP-SEED-CON",
        "94 Zxqwv Consensus Road, Consensuston, IL 98081",
    )
    q_consensus_a = "95 Zxqwv Consensus Road, Consensuston IL"
    q_consensus_b = "95 Zxqwv Consensus Rd, Consensuston IL"
    c1 = _match_save(client, q_consensus_a)
    c2 = _match_save(client, q_consensus_b)
    assert c1["status"] == "needs_review"
    assert c2["status"] == "needs_review"
    assert c1["best_id"] == con_seed_id
    _review(client, c1["match_id"], "confirmed", con_seed_id)
    _review(client, c2["match_id"], "confirmed", con_seed_id)

    data = _preview(client)
    assert data["valid"] == 4
    assert data["duplicates"] == 2
    assert data["invalid"] == 1
    assert data["invalid_lines"] == [4]
    token = data["preview_token"]

    # Reviews only touch match logs, which never enter the table snapshot.
    assert db.count_addresses(conn) == addresses_before + 3
    assert db.count_match_logs(conn) == logs_before + 5

    r = _impact(client, token)
    assert r.status_code == 200, r.text
    impact = r.json()

    # Summary counts add up.
    assert impact["new_address_count"] == 4
    assert impact["total_queries"] <= 100
    assert (
        impact["improved"] + impact["regressed"] + impact["retargeted"]
        + impact["unchanged"]
    ) == impact["total_queries"]

    improved_item = _item_for(impact, q_improved)
    assert improved_item["category"] == "improved"
    assert improved_item["before"]["status"] == "no_match"
    assert improved_item["after"]["status"] == "auto_matched"
    assert improved_item["after"]["best_is_new"] is True
    assert improved_item["after"]["best_csv_line"] == 2

    regressed_item = _item_for(impact, q_regressed)
    assert regressed_item["category"] == "regressed"
    assert regressed_item["before"]["status"] == "auto_matched"
    assert regressed_item["after"]["status"] == "needs_review"

    retargeted_item = _item_for(impact, q_retargeted)
    assert retargeted_item["category"] == "retargeted"
    assert retargeted_item["before"]["best_id"] == ret_seed_id
    assert retargeted_item["after"]["best_is_new"] is True
    assert retargeted_item["after"]["best_csv_line"] == 6
    assert retargeted_item["after"]["best_id"] != ret_seed_id

    # The review-consensus pass runs for both snapshots: this query keeps
    # auto-matching the existing consensus target (none of the temporary
    # addresses share its normalised key), so it is classified unchanged.
    consensus_item = _item_for(impact, q_consensus_b)
    assert consensus_item["category"] == "unchanged"
    assert consensus_item["before"]["status"] == "auto_matched"
    assert consensus_item["after"]["status"] == "auto_matched"
    assert consensus_item["before"]["best_id"] == con_seed_id
    assert consensus_item["after"]["best_id"] == con_seed_id
    assert consensus_item["before"]["consensus_applied"] is True
    assert consensus_item["after"]["consensus_applied"] is True

    # New-address candidates are tagged with their CSV line; the
    # before-run candidate set contains no temporary addresses.
    for cand in improved_item["before"]["match"]["candidates"]:
        assert cand["address"].get("is_new") is not True
    after_cands = improved_item["after"]["match"]["candidates"]
    best = next(c for c in after_cands if c["address"]["is_new"])
    assert best["address"]["csv_line"] == 2

    # The preview was strictly read-only: no addresses and no logs written.
    assert db.count_addresses(conn) == addresses_before + 3
    assert db.count_match_logs(conn) == logs_before + 5


def test_impact_newest_distinct_queries_and_limit(client):
    r = _impact(client, _preview(client)["preview_token"])
    assert r.status_code == 200
    data = r.json()
    assert data["query_limit"] == 100
    # Distinct queries only, newest first by log time.
    queries = [it["query"] for it in data["items"]]
    assert len(queries) == len(set(queries))
    match_ids = [it["last_match_id"] for it in data["items"]]
    assert match_ids == sorted(match_ids, reverse=True)


def test_impact_unknown_token_404(client):
    r = _impact(client, "no-such-token")
    assert r.status_code == 404


def test_impact_rejects_changed_table(client):
    token = _preview(client)["preview_token"]
    # An unrelated direct upload mutates the address-table snapshot.
    interloper = b"id,address\nIMPINT,7 Zxqwv Interloper Lane,Intruder City\n"
    assert client.post("/api/addresses/upload",
                       files={"file": ("i.csv", interloper, "text/csv")}
                       ).status_code == 200
    r = _impact(client, token)
    assert r.status_code == 409
    # Nothing was written by the impact call itself (only the interloper).


def test_impact_does_not_consume_voucher(client):
    token = _preview(client)["preview_token"]
    assert _impact(client, token).status_code == 200
    assert _impact(client, token).status_code == 200
    # A fresh voucher survives a preview run and can still be confirmed.
    fresh = (
        "external_id,address\n"
        "IMPCF1,1 Zxqwv Confirmable Road,Confirmtown\n"
        "IMPCF2,2 Zxqwv Confirmable Plaza,Confirmtown\n"
    ).encode()
    token = _preview(client, fresh, "f.csv")["preview_token"]
    assert _impact(client, token).status_code == 200
    r = client.post(
        "/api/addresses/confirm",
        files={"file": ("f.csv", fresh, "text/csv")},
        data={"preview_token": token},
    )
    assert r.status_code == 200, r.text
    assert r.json()["inserted"] == 2


def test_impact_expired_token_404(client):
    token = _preview(client)["preview_token"]
    # Force the voucher's expiry into the past and re-check.
    conn = get_conn()
    with db.transaction(conn):
        conn.execute(
            "UPDATE upload_previews SET expires_at = '1970-01-01T00:00:00+00:00'"
            " WHERE token = ?",
            (token,),
        )
    assert _impact(client, token).status_code == 404
