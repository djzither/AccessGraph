"""
DataLayer/servicenow_loader.py
─────────────────────────────────────────────────────────────────────────────
ServiceNow Table API client for CE Tickets (New Employee workflow).

Fetches rows from ``x_bryu_continuin_0_ce_ticket`` with pagination, flattens
reference-style JSON cells using ``display_value``, and normalizes strings for
downstream pandas / AccessGraph demos.

Auth: set ``SN_USER`` and ``SN_PASS`` in the environment (Basic auth).

This module is intentionally limited to ingestion — no recommendation logic.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

DEFAULT_TABLE_API_URL = (
    "https://support-test.byu.edu/api/now/table/x_bryu_continuin_0_ce_ticket"
)

NEW_EMPLOYEE_QUERY = "ticket_type=New Employee"

DEFAULT_SYSPARM_FIELDS: tuple[str, ...] = (
    "number",
    "ticket_type",
    "short_description",
    "description",
    "new_employee",
    "preferred_full_name",
    "name",
    "hiring_type",
    "employee_type",
    "copy_rights_from",
    "employee_job_title",
    "requester_department",
    "workday_driver",
    "supervisor",
    "netid",
    "approval",
    "sys_created_on",
)


class ServiceNowAuthError(RuntimeError):
    """Raised when ServiceNow credentials are missing or invalid."""


class ServiceNowAPIError(RuntimeError):
    """Raised when the ServiceNow API returns an unexpected or error response."""


def _require_credentials() -> tuple[str, str]:
    user = (os.environ.get("SN_USER") or "").strip()
    password = os.environ.get("SN_PASS")
    if password is not None:
        password = str(password)
    if not user or not password:
        raise ServiceNowAuthError(
            "Set SN_USER and SN_PASS in the environment before calling ServiceNow."
        )
    return user, password


def flatten_servicenow_cell(value: Any) -> Any:
    """
    Turn ServiceNow JSON cells into Parquet-friendly scalars.

    Reference fields typically look like
    ``{"display_value": "...", "value": "<sys_id>"}`` when
    ``sysparm_display_value=true``. We keep ``display_value`` and recurse in
    case of nested structures. Lists are joined with ``, `` after flattening
    each element. Empty strings become ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        t = value.strip()
        return None if t == "" else t
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        if not value:
            return None
        if "display_value" in value:
            return flatten_servicenow_cell(value.get("display_value"))
        if "value" in value:
            return flatten_servicenow_cell(value.get("value"))
        return None
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            flat = flatten_servicenow_cell(item)
            if flat is None or flat == "":
                continue
            parts.append(str(flat))
        if not parts:
            return None
        return ", ".join(parts) if len(parts) > 1 else parts[0]
    s = str(value).strip()
    return None if s == "" else s


def normalize_record(
    record: Mapping[str, Any],
    *,
    fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Apply ``flatten_servicenow_cell`` to every value in a row."""
    keys = fields if fields is not None else tuple(record.keys())
    out: dict[str, Any] = {}
    for key in keys:
        out[key] = flatten_servicenow_cell(record.get(key))
    return out


def _parse_result_json(resp: requests.Response) -> list[dict[str, Any]]:
    if not resp.ok:
        snippet = (resp.text or "")[:500]
        raise ServiceNowAPIError(
            f"ServiceNow HTTP {resp.status_code} for {resp.url}: {snippet}"
        )
    try:
        payload = resp.json()
    except ValueError as exc:
        raise ServiceNowAPIError(f"ServiceNow response is not JSON: {resp.text[:200]}") from exc
    result = payload.get("result")
    if result is None:
        raise ServiceNowAPIError(f"ServiceNow JSON missing 'result' key: {payload.keys()}")
    if not isinstance(result, list):
        raise ServiceNowAPIError("ServiceNow 'result' is not a list")
    return result


@dataclass(frozen=True)
class ServiceNowTableClient:
    """Thin Table API client with shared session and auth."""

    table_api_url: str
    user: str
    password: str
    session: requests.Session

    @classmethod
    def from_env(
        cls,
        table_api_url: str = DEFAULT_TABLE_API_URL,
        *,
        session: requests.Session | None = None,
    ) -> ServiceNowTableClient:
        user, password = _require_credentials()
        sess = session or requests.Session()
        return cls(
            table_api_url=table_api_url.rstrip("/"),
            user=user,
            password=password,
            session=sess,
        )

    def get_page(
        self,
        *,
        sysparm_query: str,
        sysparm_fields: Sequence[str],
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        params: MutableMapping[str, str] = {
            "sysparm_query": sysparm_query,
            "sysparm_fields": ",".join(sysparm_fields),
            "sysparm_display_value": "true",
            "sysparm_limit": str(limit),
            "sysparm_offset": str(offset),
        }
        resp = self.session.get(
            self.table_api_url,
            params=params,
            auth=HTTPBasicAuth(self.user, self.password),
            headers={"Accept": "application/json"},
            timeout=120,
        )
        if resp.status_code in (401, 403):
            raise ServiceNowAuthError(
                "ServiceNow rejected credentials (HTTP "
                f"{resp.status_code}). Check SN_USER / SN_PASS."
            )
        return _parse_result_json(resp)


def iter_new_employee_ticket_pages(
    client: ServiceNowTableClient,
    *,
    sysparm_fields: Sequence[str] = DEFAULT_SYSPARM_FIELDS,
    page_size: int = 500,
    sysparm_query: str = NEW_EMPLOYEE_QUERY,
) -> Iterator[tuple[int, int, list[dict[str, Any]]]]:
    """
    Yield ``(offset, len(rows), rows)`` for each Table API page.

    Stops when a page returns fewer than ``page_size`` rows.
    """
    offset = 0
    while True:
        rows = client.get_page(
            sysparm_query=sysparm_query,
            sysparm_fields=sysparm_fields,
            limit=page_size,
            offset=offset,
        )
        yield offset, len(rows), rows
        if len(rows) < page_size:
            break
        offset += page_size


def pull_new_employee_tickets_normalized(
    client: ServiceNowTableClient,
    *,
    sysparm_fields: Sequence[str] = DEFAULT_SYSPARM_FIELDS,
    page_size: int = 500,
    sysparm_query: str = NEW_EMPLOYEE_QUERY,
    progress_log: bool = True,
) -> pd.DataFrame:
    """
    Fetch all matching tickets, normalize cells, and return a DataFrame.

    If ``progress_log`` is True, prints a short line per page to stderr-like
    consumer — the CLI passes ``print`` here.
    """
    records: list[dict[str, Any]] = []
    for offset, n, rows in iter_new_employee_ticket_pages(
        client,
        sysparm_fields=sysparm_fields,
        page_size=page_size,
        sysparm_query=sysparm_query,
    ):
        for raw in rows:
            records.append(normalize_record(raw, fields=sysparm_fields))
        if progress_log:
            print(
                f"  Page: offset={offset}, rows_this_page={n}, cumulative={len(records)}"
            )
    if not records:
        return pd.DataFrame(columns=list(sysparm_fields))
    return pd.DataFrame.from_records(records, columns=list(sysparm_fields))
