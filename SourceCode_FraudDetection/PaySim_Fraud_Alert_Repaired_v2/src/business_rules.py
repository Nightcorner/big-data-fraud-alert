from __future__ import annotations

from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path
import re
import json
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "business_rules.json"
with CONFIG_PATH.open("r", encoding="utf-8") as _config_file:
    RULE_CONFIG = json.load(_config_file)

REQUIRED_COLUMNS = {
    "transaction_id",
    "user_id",
    "event_time",
    "country",
    "home_country",
    "transaction_type",
    "status",
    "amount_idr_equiv",
}
RULE_IDS = ("R1", "R2", "R3")
AMOUNT_THRESHOLD_IDR = float(RULE_CONFIG["rule_3"]["daily_amount_threshold_idr_equivalent"])
WINDOW_MINUTES = float(RULE_CONFIG["rule_2"]["window_minutes"])
FIFTH_TRANSACTION_NUMBER = int(RULE_CONFIG["rule_3"]["transaction_number_trigger"])


def _normalize_token(value: Any) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"[\s-]+", "_", value)
    return value


def _normalize_country(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _is_cash_withdrawal(value: Any) -> bool:
    return _normalize_token(value) in {"cash_withdrawal", "cashwithdrawal"}


def _parse_events(events: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(events.columns))
    if missing:
        raise ValueError(
            "Kolom yang diperlukan untuk business rule tidak tersedia: "
            f"{missing}. Dataset Kaggle asli tidak memiliki semua kolom ini; "
            "gunakan skema event rule atau data simulasi yang terdokumentasi."
        )
    if events.empty:
        raise ValueError("Input business rule kosong.")

    df = events.copy().reset_index(drop=True)
    df["_input_order"] = np.arange(len(df))
    for col in ("transaction_id", "user_id"):
        if df[col].isna().any():
            raise ValueError(f"Kolom {col} tidak boleh memiliki nilai kosong.")
        df[col] = df[col].astype(str).str.strip()
        if (df[col] == "").any():
            raise ValueError(f"Kolom {col} tidak boleh kosong.")
    if df["transaction_id"].duplicated().any():
        duplicates = df.loc[df["transaction_id"].duplicated(), "transaction_id"].head(5).tolist()
        raise ValueError(f"transaction_id harus unik. Contoh duplikasi: {duplicates}")

    # Input event_time must already use one consistent business timezone. Naive
    # times in the bundled synthetic fixtures are interpreted in the same local zone.
    parsed = pd.to_datetime(df["event_time"], errors="coerce")
    if parsed.isna().any():
        bad = df.loc[parsed.isna(), "event_time"].head(5).tolist()
        raise ValueError(f"event_time tidak dapat dibaca sebagai timestamp: {bad}")
    try:
        # Mixed timezone offsets / mixed naive and aware values are rejected rather
        # than silently assigning a wrong business day.
        df["_event_time"] = parsed
        df["_business_date"] = df["_event_time"].dt.date
    except (AttributeError, TypeError) as exc:
        raise ValueError(
            "event_time harus memiliki format dan zona waktu yang konsisten. "
            "Normalisasikan semua timestamp ke zona waktu operasional yang sama terlebih dahulu."
        ) from exc

    df["country"] = df["country"].map(_normalize_country)
    df["home_country"] = df["home_country"].map(_normalize_country)
    if ((df["country"] == "") | (df["home_country"] == "")).any():
        raise ValueError("country dan home_country tidak boleh kosong.")

    # Keep user home-country context consistent for all events of that user.
    home_values_per_user = df.groupby("user_id")["home_country"].nunique(dropna=False)
    if (home_values_per_user > 1).any():
        bad_users = home_values_per_user[home_values_per_user > 1].index.tolist()[:5]
        raise ValueError(f"home_country berubah untuk user yang sama: {bad_users}")

    df["transaction_type"] = df["transaction_type"].map(_normalize_token)
    df["status"] = df["status"].map(lambda x: str(x).strip().upper() if not pd.isna(x) else "")
    df["amount_idr_equiv"] = pd.to_numeric(df["amount_idr_equiv"], errors="coerce")
    if df["amount_idr_equiv"].isna().any() or not np.isfinite(df["amount_idr_equiv"].to_numpy(dtype=float)).all():
        raise ValueError("amount_idr_equiv harus numerik dan tidak boleh kosong.")
    if (df["amount_idr_equiv"] < 0).any():
        raise ValueError("amount_idr_equiv tidak boleh negatif.")

    df["_is_foreign"] = df["country"] != df["home_country"]
    df["_is_cash_withdrawal"] = df["transaction_type"].map(_is_cash_withdrawal)
    # Deterministic event ordering; same-time events have no intrinsic order in
    # the scenario schema, so transaction_id is only a technical tie-breaker.
    return df.sort_values(["user_id", "_event_time", "transaction_id"], kind="mergesort").reset_index(drop=True)


def evaluate_business_rules(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate proposed notification and risk-screening rules over event records.

    Proposed rule semantics after the public-source review:
    - R1 is a customer notification, not a fraud decision: the current event is
      the user's first foreign event of the business day, is a cash withdrawal,
      and status is APPROVED or DECLINED. Any earlier foreign event suppresses R1.
    - R2 is a potential-risk indicator: a foreign cash withdrawal is paired with
      any same-user event in a different country within 0 <= delta < 90 minutes,
      before or after the withdrawal. In streaming terms the alert is emitted when
      the second event arrives; the alert refers to the withdrawal. Country-only
      difference is coarse and does not prove physically impossible travel.
    - R3 is a potential-risk indicator: current event is a foreign cash-withdrawal
      attempt and either it is the user's fifth foreign cash-withdrawal event that
      business day (one count alert per day), or foreign cash-withdrawal amounts
      for that user/day first reach/exceed Rp6,000,000 (one amount alert per day).
      Scope and threshold are proposed assumptions, not bank-approved policy.

    Amounts are accumulated as event amounts regardless of status; if the intended
    business rule concerns only settled/approved cash disbursements, this must be
    changed after the source-system semantics are confirmed. Notifications are
    output records only; this function does not send SMS/email.
    """
    df = _parse_events(events)

    user_day_count: dict[tuple[str, date], int] = defaultdict(int)
    user_day_amount: dict[tuple[str, date], float] = defaultdict(float)
    foreign_withdrawal_count: dict[tuple[str, date], int] = defaultdict(int)
    foreign_withdrawal_amount: dict[tuple[str, date], float] = defaultdict(float)
    foreign_seen_today: set[tuple[str, date]] = set()
    amount_alerted_today: set[tuple[str, date]] = set()
    count_alerted_today: set[tuple[str, date]] = set()
    prior_events: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    r2_alerted_transaction_ids: set[str] = set()

    eval_rows: list[dict[str, Any]] = []
    alert_rows: list[dict[str, Any]] = []

    for row in df.to_dict(orient="records"):
        txid = row["transaction_id"]
        user = row["user_id"]
        event_time = row["_event_time"]
        day_key = (user, row["_business_date"])
        is_foreign = bool(row["_is_foreign"])
        is_cash = bool(row["_is_cash_withdrawal"])
        is_foreign_cash = is_foreign and is_cash
        status = row["status"]
        amount = float(row["amount_idr_equiv"])
        country = row["country"]

        # Retain only prior events within the R2 lookback window.
        history = prior_events[user]
        while history:
            delta = (event_time - history[0]["event_time"]).total_seconds() / 60.0
            if delta > WINDOW_MINUTES:
                history.popleft()
            else:
                break
        prior_different_country = [
            hist for hist in history
            if hist["country"] != country
            and 0 <= (event_time - hist["event_time"]).total_seconds() / 60.0 < WINDOW_MINUTES
        ]

        # R1: customer notification on first foreign event of the business day.
        r1 = is_foreign and day_key not in foreign_seen_today and is_cash and status in {"APPROVED", "DECLINED"}

        # General user/day counters remain in the output for context, but do not
        # drive R3 in the proposed version.
        tx_count_today = user_day_count[day_key] + 1
        amount_total_today = user_day_amount[day_key] + amount

        withdrawal_count_today = foreign_withdrawal_count[day_key] + (1 if is_foreign_cash else 0)
        withdrawal_amount_before = foreign_withdrawal_amount[day_key]
        withdrawal_amount_today = withdrawal_amount_before + (amount if is_foreign_cash else 0.0)

        r3_reasons: list[str] = []
        r3_count_condition = (
            is_foreign_cash
            and withdrawal_count_today >= FIFTH_TRANSACTION_NUMBER
            and day_key not in count_alerted_today
        )
        r3_amount_condition = (
            is_foreign_cash
            and withdrawal_amount_before < AMOUNT_THRESHOLD_IDR <= withdrawal_amount_today
            and day_key not in amount_alerted_today
        )
        if r3_count_condition:
            r3_reasons.append("fifth_or_later_foreign_cash_withdrawal_of_day")
        if r3_amount_condition:
            r3_reasons.append("foreign_cash_withdrawal_amount_reached_rp6000000")

        common = {
            "transaction_id": txid,
            "user_id": user,
            "event_time": str(event_time),
            "business_date": str(row["_business_date"]),
            "country": country,
            "home_country": row["home_country"],
            "transaction_type": row["transaction_type"],
            "status": status,
            "amount_idr_equiv": amount,
            "transaction_number_today": tx_count_today,
            "cumulative_amount_idr_today": amount_total_today,
            "foreign_cash_withdrawal_number_today": withdrawal_count_today,
            "cumulative_foreign_cash_withdrawal_amount_idr_today": withdrawal_amount_today,
        }
        alerts_caused_by_current_event: list[str] = []

        # R1: SMS/customer-notification record, with no minimum amount.
        if r1:
            alert_rows.append({
                **common,
                "rule_id": "R1",
                "notification_channel": "SMS_CUSTOMER",
                "reason": "first_foreign_transaction_of_day_with_approved_or_declined_cash_withdrawal",
                "detected_by_transaction_id": txid,
                "detected_at_event_time": str(event_time),
                "matched_transaction_id": "",
                "matched_event_time": "",
            })
            alerts_caused_by_current_event.append("R1")

        # R2 direction 1: current event is a foreign cash withdrawal and the
        # country-different event is in its preceding <90-minute window.
        r2_current = is_foreign_cash and bool(prior_different_country)
        if r2_current and txid not in r2_alerted_transaction_ids:
            closest = min(
                prior_different_country,
                key=lambda h: abs((event_time - h["event_time"]).total_seconds()),
            )
            alert_rows.append({
                **common,
                "rule_id": "R2",
                "notification_channel": "EMAIL_FRAUD_TEAM",
                "reason": "foreign_cash_withdrawal_with_different_country_transaction_within_less_than_90_minutes",
                "detected_by_transaction_id": txid,
                "detected_at_event_time": str(event_time),
                "matched_transaction_id": closest["transaction_id"],
                "matched_event_time": str(closest["event_time"]),
            })
            r2_alerted_transaction_ids.add(txid)
            alerts_caused_by_current_event.append("R2")

        # R2 direction 2: the current event completes a pair for a prior qualifying
        # foreign cash withdrawal. This is causal/stream-compatible: no future event
        # is used before it arrives. Only one alert per qualifying withdrawal.
        prior_withdrawals_matched = [
            hist for hist in history
            if hist["is_foreign_cash_withdrawal"]
            and hist["country"] != country
            and 0 <= (event_time - hist["event_time"]).total_seconds() / 60.0 < WINDOW_MINUTES
            and hist["transaction_id"] not in r2_alerted_transaction_ids
        ]
        for withdrawal in prior_withdrawals_matched:
            alert_rows.append({
                **withdrawal["alert_context"],
                "rule_id": "R2",
                "notification_channel": "EMAIL_FRAUD_TEAM",
                "reason": "prior_foreign_cash_withdrawal_matched_by_different_country_transaction_within_less_than_90_minutes",
                "detected_by_transaction_id": txid,
                "detected_at_event_time": str(event_time),
                "matched_transaction_id": txid,
                "matched_event_time": str(event_time),
            })
            r2_alerted_transaction_ids.add(withdrawal["transaction_id"])
            alerts_caused_by_current_event.append("R2")

        # R3: risk screening. The count and amount are scoped to foreign cash
        # withdrawals, and count/amount alerts are each emitted at most once/day.
        if r3_reasons:
            alert_rows.append({
                **common,
                "rule_id": "R3",
                "notification_channel": "EMAIL_FRAUD_TEAM",
                "reason": "+".join(r3_reasons),
                "detected_by_transaction_id": txid,
                "detected_at_event_time": str(event_time),
                "matched_transaction_id": "",
                "matched_event_time": "",
            })
            alerts_caused_by_current_event.append("R3")
            if "fifth_or_later_foreign_cash_withdrawal_of_day" in r3_reasons:
                count_alerted_today.add(day_key)
            if "foreign_cash_withdrawal_amount_reached_rp6000000" in r3_reasons:
                amount_alerted_today.add(day_key)

        r2_triggered = "R2" in alerts_caused_by_current_event
        eval_rows.append({
            **common,
            "is_foreign": is_foreign,
            "is_cash_withdrawal": is_cash,
            "is_foreign_cash_withdrawal": is_foreign_cash,
            "rule1_triggered": r1,
            "rule2_triggered": r2_triggered,
            "rule3_triggered": bool(r3_reasons),
            "rule3_reasons": "+".join(r3_reasons),
            "triggered_rule_ids": "|".join(rule_id for rule_id, triggered in (("R1", r1), ("R2", r2_triggered), ("R3", bool(r3_reasons))) if triggered) or "NONE",
        })

        # Update counters only after evaluating the current row.
        user_day_count[day_key] += 1
        user_day_amount[day_key] += amount
        if is_foreign:
            foreign_seen_today.add(day_key)
        if is_foreign_cash:
            foreign_withdrawal_count[day_key] += 1
            foreign_withdrawal_amount[day_key] += amount
        history.append({
            "transaction_id": txid,
            "event_time": event_time,
            "country": country,
            "is_foreign_cash_withdrawal": is_foreign_cash,
            "alert_context": common,
        })

    evaluation = pd.DataFrame(eval_rows)
    alerts = pd.DataFrame(alert_rows)
    if alerts.empty:
        alerts = pd.DataFrame(columns=[
            "transaction_id", "user_id", "event_time", "business_date", "country",
            "home_country", "transaction_type", "status", "amount_idr_equiv",
            "transaction_number_today", "cumulative_amount_idr_today",
            "foreign_cash_withdrawal_number_today", "cumulative_foreign_cash_withdrawal_amount_idr_today",
            "rule_id", "notification_channel", "reason", "detected_by_transaction_id",
            "detected_at_event_time", "matched_transaction_id", "matched_event_time"
        ])

    # Restore input order for easier tracing to the input CSV.
    order_map = dict(zip(df["transaction_id"], df["_input_order"]))
    evaluation["_input_order"] = evaluation["transaction_id"].map(order_map)
    evaluation = evaluation.sort_values("_input_order").drop(columns=["_input_order"]).reset_index(drop=True)
    if not alerts.empty:
        alerts["_input_order"] = alerts["transaction_id"].map(order_map)
        alerts = alerts.sort_values(["_input_order", "rule_id"]).drop(columns=["_input_order"]).reset_index(drop=True)
    return evaluation, alerts


def expected_rules_by_transaction(events: pd.DataFrame) -> dict[str, set[str]]:
    """Helper for fixture validation; expects an `expected_rules` column."""
    if "expected_rules" not in events.columns:
        raise ValueError("Fixture validasi harus memiliki kolom expected_rules.")
    output: dict[str, set[str]] = {}
    for row in events[["transaction_id", "expected_rules"]].itertuples(index=False):
        raw = "" if pd.isna(row.expected_rules) else str(row.expected_rules).strip()
        output[str(row.transaction_id)] = set() if raw in {"", "NONE"} else set(raw.split("|"))
    return output
