"""
DataLayer/pdf_ticket_parser.py
─────────────────────────────────────────────────────────────────────────────
Temporary **demo** ingestion for CE Tickets exported as PDF from ServiceNow.

┌─────────────────────────────────────────────────────────────────────────────┐
│ Future ServiceNow REST API integration (see DataLayer/servicenow_loader.py): │
│                                                                              │
│ • Replace ONLY the **source** step: PDF text extraction → Table API JSON.    │
│ • Keep this module's **normalization** and **demo identity resolution**      │
│   as a shared layer so ``demo_servicenow_tickets.parquet`` and API-backed    │
│   parquet share column semantics for downstream AccessGraph demos.           │
│ • API rows map naturally: ``short_description`` ↔ Title, ``description`` ↔   │
│   Description, reference fields ↔ Requester / Supervisor display values.     │
└─────────────────────────────────────────────────────────────────────────────┘

Deterministic: sorted PDF paths, stable regex, reproducible matching scores.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Defaults (demo layout under repo)
# -----------------------------------------------------------------------------

DEFAULT_DEMO_PDF_DIR = Path("data/demo_processed/demo_pdf")
DEFAULT_SANITIZED_USERS_PATH = Path("data/demo_processed/sanitized_clean_users.parquet")
DEFAULT_MAPPING_DIR = Path("private/demo_mapping")

# Labels that terminate a multiline field value (longest first for regex alternation).
_TICKET_FIELD_LABELS: tuple[str, ...] = (
    "Work Log(comments)",
    "Internal Notes",
    "Employee Job Title",
    "Employee Type",
    "Requested Participants [only one]",
    "Correlation display",
    "Website Access Notes",
    "Website Access BYU ID",
    "Website Access Name",
    "Drupal Site",
    "Technology Approval Step",
    "Orion Issue #",
    "Workday Driver",
    "Developer Ticket Info",
    "Room #",
    "Assigned To",
    "Scheduled",
    "Ticket Type",
    "Opened by",
    "Requester",
    "Priority",
    "Job Title",
    "Title",
    "State",
    "Number",
    "Supervisor",
    "Employee",
    "Description",
    "Short description",
    "Data Request Type",
)

# Noise lines / section banners (not ``Label: value`` pairs).
_BOILERPLATE_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*CE Tickets Details\s*$", re.I),
    re.compile(r"^\s*Report Title:\s*$", re.I),
    re.compile(r"^\s*Run Date and Time:\s*$", re.I),
    re.compile(r"^\s*Run by:\s*$", re.I),
    re.compile(r"^\s*Run By\s*:.*$", re.I),
    re.compile(r"^\s*Table Name:\s*$", re.I),
    re.compile(r"^\s*x_bryu_continuin_0_ce_ticket\s*$", re.I),
    re.compile(r"^\s*Page\s+\d+\s*$", re.I),
    re.compile(r"^\s*sys_popup\s*$", re.I),
    re.compile(r"^\s*Data Request\s*$", re.I),
    re.compile(r"^\s*CE Tickets\s*$"),  # repeated header; real fields use "CE Tickets\nTitle:"
    re.compile(r"^\s*\d{4}-\d{2}-\d{2}.+Mountain\b.*Time\s*$"),
    re.compile(r"^\s*Other Projects Priority:\s*$", re.I),
    re.compile(r"^\s*REDA Survey Development Priority:\s*$", re.I),
    re.compile(r"^\s*REDA Data/Visualization Report\s*$", re.I),
    re.compile(r"^\s*Development Priority:\s*$", re.I),
)

_SECTION_ONLY_LINES: frozenset[str] = frozenset(
    {
        "employee information",
        "description",
        "developer ticket info",
    }
)

# Door / access keywords for lightweight demo analytics (deterministic keyword sets).
_DOOR_KEYWORDS: tuple[str, ...] = (
    "door",
    "doors",
    "hceb",
    "hcen",
    "access",
    "badge",
    "reader",
    "lock",
    "floor",
    "room",
)

_ONBOARDING_TICKET_TYPES: frozenset[str] = frozenset(
    {
        "new employee",
        "onboarding",
        "orientation",
    }
)


class PDFExtractionError(RuntimeError):
    """Neither PyMuPDF nor pdfplumber could read the PDF."""


def extract_pdf_text(path: Path) -> str:
    """
    Extract plain text from one PDF, preferring PyMuPDF (fitz), then pdfplumber.

    This is the **demo substitute** for ``ServiceNowTableClient.get_page`` —
    both produce unstructured/semi-structured content that the normalization
    layer below turns into rows.
    """
    errors: list[str] = []

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(path)
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text())
        doc.close()
        return "\n\n".join(parts)
    except Exception as exc:
        errors.append(f"PyMuPDF: {exc}")

    try:
        import pdfplumber

        parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return "\n\n".join(parts)
    except Exception as exc:
        errors.append(f"pdfplumber: {exc}")

    raise PDFExtractionError("; ".join(errors))


def clean_pdf_text(text: str) -> str:
    """
    Drop repeated CE export headers/footers and report chrome.

    ServiceNow UI exports often repeat titles per page — analogous to API
    pagination fields we will filter out when mapping JSON metadata later.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.rstrip()
        if any(p.match(stripped) for p in _BOILERPLATE_LINE_PATTERNS):
            continue
        low = stripped.strip().lower()
        if low in _SECTION_ONLY_LINES:
            continue
        cleaned.append(line.rstrip())
    # Collapse excessive blank lines while preserving paragraph breaks in notes.
    out_lines: list[str] = []
    blank_run = 0
    for ln in cleaned:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                out_lines.append("")
        else:
            blank_run = 0
            out_lines.append(ln.rstrip())
    return "\n".join(out_lines).strip()


def _scrub_orphan_section_banners(block: str) -> str:
    """Remove lone ``Description`` banner lines that precede ``Description:``."""
    return re.sub(r"(?ms)^Description\s*\n(?=^\s*Description:\s*)", "", block)


def extract_field(block: str, label: str, *, all_labels: Sequence[str] | None = None) -> str | None:
    """
    Extract a ``Label:\\nvalue`` field up to the next known ticket label line.

    Tolerates multiline values (doors lists, internal notes). Missing labels
    return None (caller sets parsed_successfully / validation flags).
    """
    labels = tuple(all_labels) if all_labels is not None else _TICKET_FIELD_LABELS
    others = [x for x in labels if x != label]
    others_sorted = sorted(others, key=len, reverse=True)
    alt = "|".join(re.escape(o) for o in others_sorted)
    pattern = rf"(?ms)^{re.escape(label)}:\s*\n?(.*?)(?=^\s*(?:{alt}):\s*)"
    m = re.search(pattern, block)
    if not m:
        # Single trailing field (no following label)
        end_anchor = rf"(?ms)^{re.escape(label)}:\s*\n?(.*)$"
        m2 = re.search(end_anchor, block)
        return m2.group(1).strip() if m2 else None
    val = m.group(1).strip()
    return None if val == "" else val


def parse_ticket_blocks(cleaned_text: str) -> list[tuple[str | None, str]]:
    """
    Split cleaned export text into blocks, one per distinct ``Number`` field.

    Repeated ``Number:`` sections (multiple PDF pages / repeated headers) are
    merged by ticket id so door/access notes stay intact — similar to merging
    paginated API ``result`` arrays that reference the same ``sys_id``.

    Text **before** the first ``Number:`` (title, requester, ticket type, …) is
    prepended to every ticket block — exports often place fields above the
    first ``Number`` line; the REST API returns those as sibling columns
    instead.
    """
    # Primary split: ``Number:`` then ticket id (often on the next line in exports).
    part_re = re.compile(r"(?ms)^Number:\s*\n\s*(TK\d+)\s*$", re.MULTILINE)
    matches = list(part_re.finditer(cleaned_text))
    preamble = ""
    if matches:
        preamble = cleaned_text[: matches[0].start()].rstrip()

    if not matches:
        # Fallback: any TK mention — whole body as one anonymous block
        if re.search(r"\bTK\d{7}\b", cleaned_text):
            return [(None, cleaned_text)]
        return [(None, cleaned_text)]

    blocks_by_number: dict[str, list[str]] = defaultdict(list)
    for i, m in enumerate(matches):
        tid = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned_text)
        chunk = cleaned_text[start:end].strip()
        if preamble:
            chunk = f"{preamble}\n\n{chunk}".strip()
        blocks_by_number[tid].append(chunk)

    ordered_ids = sorted(blocks_by_number.keys())
    out: list[tuple[str | None, str]] = []
    for tid in ordered_ids:
        merged = _merge_ticket_chunks(blocks_by_number[tid])
        out.append((tid, merged))
    return out


def _merge_ticket_chunks(chunks: Sequence[str]) -> str:
    """
    Merge duplicate exports of the same ticket from multiple pages.

    Deduplicate **whole chunks** only — line-level dedupe would collapse repeated
    legitimate values (e.g. Requester and Opened_by sharing the same name).
    """
    seen_blocks: set[str] = set()
    merged_parts: list[str] = []
    for ch in chunks:
        block = ch.strip()
        if not block or block in seen_blocks:
            continue
        seen_blocks.add(block)
        merged_parts.append(block)
    text = "\n\n".join(merged_parts).strip()
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


def _strip_duplicate_notes_prefix_from_work_log(
    work_log: str | None,
    internal_notes: str | None,
) -> str | None:
    """When the PDF nests ``Internal Notes`` inside the work-log region, keep only tail."""
    if not work_log or not internal_notes:
        return work_log
    wl = work_log.strip()
    notes = internal_notes.strip()
    first_note = notes.split("\n---\n", 1)[0].strip()
    if not wl.lower().startswith("internal notes:"):
        return work_log
    rest = re.sub(r"(?is)^\s*internal notes:\s*", "", wl).strip()
    if first_note and rest.startswith(first_note):
        tail = rest[len(first_note) :].strip()
        return tail or None
    return work_log


def _normalize_display_fields(rec: dict[str, Any]) -> None:
    """In-place tidy for parquet (single-line title)."""
    t = rec.get("Title")
    if isinstance(t, str) and "\n" in t:
        rec["Title"] = re.sub(r"\s+", " ", t.replace("\n", " ")).strip()


def _extract_internal_notes(block: str) -> str | None:
    """Internal Notes → next ``Work Log(comments):`` or EOF (supports repeats)."""
    markers = list(
        re.finditer(r"(?ms)^Internal Notes:\s*", block),
    )
    if not markers:
        return None
    chunks: list[str] = []
    for i, mk in enumerate(markers):
        start = mk.end()
        tail = block[start:]
        wl = re.search(r"(?ms)^Work Log\(comments\):\s*", tail)
        end = wl.start() if wl else len(tail)
        piece = tail[:end].strip()
        if piece:
            chunks.append(piece)
    if not chunks:
        return None
    if len(chunks) > 1 and len({c.strip() for c in chunks}) == 1:
        return chunks[0]
    return "\n---\n".join(chunks) if len(chunks) > 1 else chunks[0]


def _extract_work_log(block: str) -> str | None:
    """
    Choose the best ``Work Log(comments):`` section.

    Exports often embed a **nested** mini ticket (``CE Tickets`` …) inside the
    first log — skip that copy when a cleaner section exists (e.g. log
    immediately followed by ``Internal Notes`` with door lists).
    """
    headers = list(re.finditer(r"(?ms)^Work Log\(comments\):\s*", block))
    if not headers:
        return None

    bodies: list[tuple[int, str]] = []
    for i, hk in enumerate(headers):
        body_start = hk.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(block)
        body = block[body_start:body_end].strip()
        if body:
            bodies.append((i, body))

    def _is_nested_echo(text: str) -> bool:
        head = "\n".join(text.split("\n")[:3]).lower()
        return "ce tickets" in head and "number:" in head

    non_echo = [(i, b) for i, b in bodies if not _is_nested_echo(b)]
    chosen = non_echo[-1][1] if non_echo else bodies[-1][1]
    return chosen.strip() or None


def _parse_block_to_record(block: str, *, inferred_number: str | None) -> dict[str, Any]:
    """Apply ``extract_field`` for each semantic column."""
    block = _scrub_orphan_section_banners(block)
    labels = _TICKET_FIELD_LABELS
    number = extract_field(block, "Number", all_labels=labels)
    if not number and inferred_number:
        number = inferred_number
    title = extract_field(block, "Title", all_labels=labels)
    requester = extract_field(block, "Requester", all_labels=labels)
    opened_by = extract_field(block, "Opened by", all_labels=labels)
    ticket_type = extract_field(block, "Ticket Type", all_labels=labels)
    state = extract_field(block, "State", all_labels=labels)
    employee_type = extract_field(block, "Employee Type", all_labels=labels)
    employee_job_title = extract_field(block, "Employee Job Title", all_labels=labels)
    supervisor = extract_field(block, "Supervisor", all_labels=labels)

    desc_short = extract_field(block, "Description", all_labels=labels)

    internal_notes = _extract_internal_notes(block)
    work_log = _extract_work_log(block)
    work_log = _strip_duplicate_notes_prefix_from_work_log(work_log, internal_notes)

    core_present = sum(
        1
        for x in (number, title, ticket_type)
        if x and str(x).strip()
    )
    parsed_ok = core_present >= 2 and bool(number)

    out = {
        "Number": (number or "").strip() or None,
        "Title": title,
        "Requester": requester,
        "Opened by": opened_by,
        "Ticket Type": ticket_type,
        "State": state,
        "Employee Type": employee_type,
        "Employee Job Title": employee_job_title,
        "Supervisor": supervisor,
        "Description": desc_short,
        "Internal Notes": internal_notes,
        "Work Log/comments": work_log,
        "parsed_successfully": parsed_ok,
    }
    _normalize_display_fields(out)
    return out


def parse_pdf_to_records(
    pdf_path: Path,
    *,
    extract_text_fn: Callable[[Path], str] | None = None,
) -> list[dict[str, Any]]:
    """
    Full parse pipeline for one file: extract → clean → blocks → record dicts.

    Returns the list of record dicts directly. Each dict contains the parsed
    ticket fields plus ``raw_text`` and ``source_pdf`` for provenance.
    """
    extract_text_fn = extract_text_fn or extract_pdf_text
    raw_text = extract_text_fn(pdf_path)
    cleaned = clean_pdf_text(raw_text)
    blocks = parse_ticket_blocks(cleaned)
    records: list[dict[str, Any]] = []
    for inferred_id, block in blocks:
        rec = _parse_block_to_record(block, inferred_number=inferred_id)
        rec["raw_text"] = raw_text
        rec["source_pdf"] = str(pdf_path.as_posix())
        records.append(rec)
    return records


def load_demo_mapping_tables(
    mapping_dir: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load optional CSV maps produced by ``scripts/build_demo_dataset``."""
    user_map_path = mapping_dir / "user_map.csv"
    person_map_path = mapping_dir / "person_name_map.csv"
    um = pm = None
    if user_map_path.is_file():
        um = pd.read_csv(user_map_path, dtype=str).fillna("")
    if person_map_path.is_file():
        pm = pd.read_csv(person_map_path, dtype=str).fillna("")
    return um, pm


def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _fake_display_for_real_name(real_name: str, person_map: pd.DataFrame | None) -> str | None:
    if person_map is None or not real_name.strip():
        return None
    key = _norm_key(real_name)
    if key == "(various)" or key == "":
        return None
    subset = person_map[
        person_map["real_person_name"].map(_norm_key) == key
    ]
    if len(subset) == 1:
        return str(subset.iloc[0]["fake_person_name"])
    return None


def _fake_sam_for_real_netid(netid: str, user_map: pd.DataFrame | None) -> str | None:
    if user_map is None or not netid.strip():
        return None
    key = netid.strip().lower()
    subset = user_map[user_map["real_sam_account"].str.lower() == key]
    if len(subset) == 1:
        return str(subset.iloc[0]["fake_sam_account"])
    return None


def _extract_netid_from_title(title: str | None) -> str | None:
    if not title:
        return None
    # Example: "Student - Seojin Mun - seojinm (doors)"
    m = re.search(
        r"-\s*([A-Za-z][A-Za-z0-9_.-]{2,})\s*(?:\(doors\)|\(door\))\s*$",
        title.strip(),
        re.I,
    )
    if m:
        return m.group(1).strip()
    m2 = re.search(r"-\s*([a-z0-9_.-]{3,})\s*$", title.strip(), re.I)
    return m2.group(1).strip() if m2 else None


def resolve_demo_identity(
    row: dict[str, Any],
    *,
    demo_users: pd.DataFrame,
    user_map: pd.DataFrame | None,
    person_map: pd.DataFrame | None,
) -> dict[str, Any]:
    """
    Map PDF **real** names / netids to sanitized demo users via mapping CSVs.

    When no confident join exists, flags only — **never invent** fake names.
    """
    title = row.get("Title")
    netid = _extract_netid_from_title(title)

    demo_sam: str | None = None
    demo_name: str | None = None
    confidence = "none"
    notes: list[str] = []

    if netid:
        fs = _fake_sam_for_real_netid(netid, user_map)
        if fs:
            matches = demo_users[demo_users["SamAccountName"].astype(str).str.lower() == fs.lower()]
            if len(matches) == 1:
                demo_sam = str(matches.iloc[0]["SamAccountName"])
                demo_name = str(matches.iloc[0]["DisplayName"])
                confidence = "high"
                notes.append(f"netid:{netid}->user_map")
            elif len(matches) == 0:
                notes.append(f"netid mapped to {fs} but not in sanitized_clean_users")

    if confidence == "none":
        # Try primary employee name embedded in Title before netid segment.
        mname = re.match(
            r"^\s*\S+\s*-\s*(.+?)\s*-\s*\S+(?:\s*\([^)]*\))?\s*$",
            (title or "").strip(),
        )
        employee_guess = mname.group(1).strip() if mname else None
        for candidate in filter(
            None,
            (employee_guess, row.get("Requester"), row.get("Opened by")),
        ):
            fd = _fake_display_for_real_name(str(candidate), person_map)
            if fd:
                dm = demo_users[demo_users["DisplayName"].astype(str).str.strip() == fd]
                if len(dm) == 1:
                    demo_sam = str(dm.iloc[0]["SamAccountName"])
                    demo_name = str(dm.iloc[0]["DisplayName"])
                    confidence = "medium"
                    notes.append(f"display:{candidate}->person_map")
                    break

    matched = confidence in {"high", "medium"}

    out = {
        "DemoSamAccountName": demo_sam,
        "DemoDisplayName": demo_name,
        "MatchedDemoUser": matched,
        "MatchConfidence": confidence,
        "IdentityMatchNotes": "; ".join(notes) if notes else None,
    }
    return out


def build_ticket_dataframe(
    pdf_dir: Path,
    *,
    sanitized_users_path: Path = DEFAULT_SANITIZED_USERS_PATH,
    mapping_dir: Path = DEFAULT_MAPPING_DIR,
    extract_text_fn: Callable[[Path], str] | None = None,
) -> pd.DataFrame:
    """
    Parser → normalization → dataframe.

    Swap ``pdf_dir`` / ``extract_pdf_text`` for API JSON ingestion later while
    keeping ``resolve_demo_identity`` + output columns stable.
    """
    pdf_dir = pdf_dir.resolve()
    paths = sorted(pdf_dir.glob("*.pdf"))
    rows: list[dict[str, Any]] = []

    demo_users = pd.read_parquet(sanitized_users_path)
    user_map, person_map = load_demo_mapping_tables(mapping_dir)

    extract_text_fn = extract_text_fn or extract_pdf_text

    for path in paths:
        try:
            records = parse_pdf_to_records(path, extract_text_fn=extract_text_fn)
        except PDFExtractionError as exc:
            logger.warning("Failed to read PDF %s: %s", path, exc)
            rows.append(
                {
                    "Number": None,
                    "Title": None,
                    "Requester": None,
                    "Opened by": None,
                    "Ticket Type": None,
                    "State": None,
                    "Employee Type": None,
                    "Employee Job Title": None,
                    "Supervisor": None,
                    "Description": None,
                    "Internal Notes": None,
                    "Work Log/comments": None,
                    "raw_text": "",
                    "source_pdf": str(path.as_posix()),
                    "parsed_successfully": False,
                    "parse_error": str(exc),
                }
            )
            continue

        for rec in records:
            id_map = resolve_demo_identity(rec, demo_users=demo_users, user_map=user_map, person_map=person_map)
            merged = {**rec, **id_map}
            rows.append(merged)

    df = pd.DataFrame(rows)
    return df


def validate_ticket_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Emit counts for logging: parses, failures, critical gaps, duplicates."""
    total = len(df)
    ok = int(df["parsed_successfully"].sum()) if "parsed_successfully" in df.columns else 0
    failed = total - ok
    critical_missing = 0
    if total:
        num_empty = df["Number"].isna() | (df["Number"].astype(str).str.strip() == "")
        crit = num_empty | df["Title"].isna() | (df["Title"].astype(str).str.strip() == "")
        critical_missing = int(crit.sum())

    dup_ids: list[str] = []
    if "Number" in df.columns and total:
        vc = df["Number"].dropna().astype(str)
        vc = vc[vc.str.strip() != ""]
        dup_ids = vc[vc.duplicated(keep=False)].unique().tolist()

    return {
        "tickets_total": total,
        "parsed_successfully": ok,
        "parse_failed": failed,
        "missing_critical_fields": critical_missing,
        "duplicate_ticket_numbers": dup_ids,
    }


def analyze_demo_tickets_for_access_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example analysis pass for demos — **not** production routing.

    • Surface probable access/door language in Internal Notes (regex keywords).
    • Flag onboarding-oriented ticket types for future recommendation queues.

    When ServiceNow API ingestion lands, run the same functions on the column
    ``internal_notes`` filled from ``u_internal_notes`` (or equivalent).
    """
    def _kw_hits(text: str | None) -> str:
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""
        low = str(text).lower()
        hits = [k for k in _DOOR_KEYWORDS if k in low]
        return ",".join(sorted(set(hits)))

    def _access_like(text: str | None) -> bool:
        if not text or (isinstance(text, float) and pd.isna(text)):
            return False
        low = str(text).lower()
        return ("door" in low or "access" in low or "hceb" in low or "hcen" in low)

    def _onboarding(row: pd.Series) -> bool:
        tt = str(row.get("Ticket Type") or "").strip().lower()
        return tt in _ONBOARDING_TICKET_TYPES or "new employee" in tt

    out = df.copy()
    notes_col = "Internal Notes"
    if notes_col not in out.columns:
        out["_demo_internal_notes_keywords"] = ""
        out["_demo_internal_notes_access_like"] = False
    else:
        out["_demo_internal_notes_keywords"] = out[notes_col].map(_kw_hits)
        out["_demo_internal_notes_access_like"] = out[notes_col].map(_access_like)

    title_col = "Title"
    out["_demo_title_access_like"] = (
        out[title_col].map(_access_like) if title_col in out.columns else False
    )
    out["_demo_onboarding_ticket"] = out.apply(_onboarding, axis=1)

    # Deterministic pseudo-queue label for narrative demos (hash ticket number).
    def _route_hint(num: object) -> str:
        s = str(num or "").strip() or "UNKNOWN"
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]
        return f"demo_route_{h}"

    if "Number" in out.columns:
        out["_demo_example_route_hint"] = out["Number"].map(_route_hint)
    else:
        out["_demo_example_route_hint"] = "demo_route_unknown"

    return out


def log_validation_summary(stats: dict[str, Any]) -> None:
    dup = stats.get("duplicate_ticket_numbers") or []
    logger.info(
        "Demo PDF ticket validation — total=%s ok=%s failed=%s missing_critical=%s duplicates=%s",
        stats.get("tickets_total"),
        stats.get("parsed_successfully"),
        stats.get("parse_failed"),
        stats.get("missing_critical_fields"),
        dup if dup else "(none)",
    )


def analyze_demo_tickets_example_report(df: pd.DataFrame) -> None:
    """
    Printable narrative for demos — highlights access/onboarding signals.

    Replace PDF source with API rows later without changing this report layout.
    """
    if df.empty:
        print("[analysis] No ticket rows to analyze.")
        return
    enriched = analyze_demo_tickets_for_access_signals(df)
    access_notes = int(enriched["_demo_internal_notes_access_like"].sum())
    onboarding = int(enriched["_demo_onboarding_ticket"].sum())
    title_access = int(enriched["_demo_title_access_like"].sum())
    print("[analysis] --- demo access / onboarding signals ---")
    print(f"  Tickets with door/access-like Internal Notes: {access_notes}")
    print(f"  Tickets with door/access-like Title:          {title_access}")
    print(f"  Onboarding-oriented ticket types:             {onboarding}")
    sample = enriched.loc[
        enriched["_demo_internal_notes_access_like"] | enriched["_demo_onboarding_ticket"],
        [
            "Number",
            "Ticket Type",
            "_demo_internal_notes_keywords",
            "_demo_onboarding_ticket",
        ],
    ].head(15)
    if not sample.empty:
        print("[analysis] Sample rows (first 15 matches):")
        print(sample.to_string(index=False))
    else:
        print("[analysis] No combined matches in sample slice.")
