"""Minimal CityUHK taught-postgraduate programme parser.

Design based on ``docs/programme_schema.md``.

Converts a programme markdown file (e.g. ``data/markdown/p66.md``) into a
structured Programme object:

    {
      "programme_id": "P66",
      "name": "MSc Mechanical Engineering",
      "name_zh": "理學碩士(機械工程學)",
      "apply_now_url": "...",
      "source_file": "data/markdown/p66.md",
      "metadata": { ... Category A structured fields ... },
      "contacts": [ {role, name, qualification, email, phone, fax}, ... ],
      "outline": [ ... TOC section titles ... ],
      "sections": [ {title, category, content, char_count}, ... ],
      "footnotes": [ ... trailing footnote paragraphs ... ]
    }

Four parts (this file):

  1. ProgrammeParser   -- Category A structured metadata:
                           programme id, EN/CN name, ``##`` metadata fields,
                           tuition fee, contact blocks.
  2. OutlineExtractor  -- reads the TOC under ``### Programme Outlines``
                           (the ground truth for section segmentation).
  3. SectionSegmenter  -- splits the body (after the ``Outline`` marker) into
                           section chunks using the TOC titles.
  4. parse()/parse_all() -- orchestration: single file + all 64 files,
                           JSON output (``data/programmes.json``).

Known simplifications (documented, out of scope for this version):
  * body section titles are matched exactly against the TOC; nested
    sub-headings (e.g. PILOs, stream tables) stay inside their parent chunk;
  * footnotes that are *not* at the end of the file (e.g. p70's mid-file
    "Remarks: ..." line) remain inside their enclosing section;
  * ``## Local/Non-local Applicants`` headings (p17/p53, not in the TOC) are
    not split out.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# constants (from docs/programme_schema.md §3)
# ---------------------------------------------------------------------------

RAG_DIR = Path(__file__).resolve().parent
ROOT_DIR = RAG_DIR.parent
DATA_DIR = ROOT_DIR / "data" / "markdown"
OUTPUT_PATH = ROOT_DIR / "data" / "programmes.json"

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ISO_DEADLINE_RE = re.compile(r"^\d{8}T\d{6}Z$")
PHONE_TAGGED_RE = re.compile(r"^\+?[\d ]+ \((Phone|Fax)\)$")
PHONE_BARE_RE = re.compile(r"^\+?[\d\s\-]+$")

METADATA_HEADERS = (
    "Year of Entry",
    "Application Deadline",
    "Mode of Study",
    "Mode of Funding",
    "Indicative Intake Target",
    "Minimum No. of Credits Required",
    "Class Schedule",
    "Normal Study Period",
    "Maximum Study Period",
    "Mode of Processing",
    "Tuition Fee",
    "Programme Website",
    "Intermediate Award",
)

HEADER_TO_KEY = {
    "Year of Entry": "year_of_entry",
    "Application Deadline": "application_deadline",
    "Mode of Study": "mode_of_study",
    "Mode of Funding": "mode_of_funding",
    "Indicative Intake Target": "indicative_intake_target",
    "Minimum No. of Credits Required": "minimum_no_of_credits_required",
    "Class Schedule": "class_schedule",
    "Normal Study Period": "normal_study_period",
    "Maximum Study Period": "maximum_study_period",
    "Mode of Processing": "mode_of_processing",
    "Tuition Fee": "tuition_fee",
    "Programme Website": "programme_website",
    "Intermediate Award": "intermediate_award",
}

CONTACT_ROLE_RE = re.compile(
    r"^[A-Z][A-Za-z ]*(?:Leader|Director|Tutor|Enquiries)$"
)


def is_contact_label(s: str) -> bool:
    return bool(CONTACT_ROLE_RE.match(s))

# Section category per docs/programme_schema.md §4 (B) / §5 (C).
SECTION_CATEGORIES = {
    # B -- retrieval knowledge sections (embed as child documents)
    "Programme Aims and Objectives": "B",
    "Programme Intended Learning Outcomes (PILOs)": "B",
    "Entrance Requirements": "B",
    "English Proficiency Requirements": "B",
    "Local Applicants": "B",
    "Non-local Applicants": "B",
    "Course Description": "B",
    "Programme Content": "B",
    "Programme Structure & Courses": "B",
    # C -- optional information (store separately)
    "Useful Links": "C",
    "Scholarship": "C",
    "Scholarships": "C",
    "Hong Kong Future Talents Scholarship Scheme for Advanced Studies": "C",
    "Bonus Features": "C",
    "Programme Features": "C",
    "Professional Accreditation": "C",
    "Professional Recognition": "C",
    "Career": "C",
    "Career Prospects": "C",
    "Did You Know?": "C",
    "Remarks": "C",
    "Credit Unit Transfer": "C",
    "Medium of Instruction and Assessment": "C",
    "Admissions Interview": "C",
    "Curriculum Design": "C",
}
DEFAULT_CATEGORY = "C"


# ---------------------------------------------------------------------------
# Part 1 -- ProgrammeParser: Category A structured metadata
# ---------------------------------------------------------------------------


class ProgrammeParser:
    """Extracts identity, ``##`` metadata fields, tuition fee and contacts."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lines = self.path.read_text(encoding="utf-8").splitlines()
        # metadata + contacts live before the outlines heading
        self._outlines_idx = self._find_outlines_index()
        self._meta_region = (
            self.lines[: self._outlines_idx]
            if self._outlines_idx is not None
            else self.lines
        )

    def parse(self) -> dict:
        front = self._parse_front_matter()
        metadata = self._parse_metadata()
        contacts = self._parse_contacts()
        return {
            "programme_id": front["programme_id"],
            "name": front["name"],
            "name_zh": front["name_zh"],
            "apply_now_url": front["apply_now_url"],
            "metadata": metadata,
            "contacts": contacts,
            "source_file": str(
                self.path.resolve().relative_to(ROOT_DIR.resolve())
            ),
        }

    # -- identity -----------------------------------------------------------

    def _find_outlines_index(self):
        for i, line in enumerate(self.lines):
            if line.strip() == "### Programme Outlines":
                return i
        return None

    def _parse_front_matter(self) -> dict:
        pid = name = name_zh = apply_url = None
        seen_pid = False
        for line in self.lines[:12]:
            s = line.strip()
            if not s:
                continue
            if not seen_pid:
                if re.match(r"^P\d+$", s):
                    pid = s
                    seen_pid = True
                continue
            if name is None:
                name = s
                continue
            if name_zh is None and CJK_RE.search(s):
                name_zh = s
                continue
            if apply_url is None:
                m = re.match(r"^\[Apply Now\]\((.*)\)$", s)
                if m:
                    apply_url = m.group(1)
        return {
            "programme_id": pid,
            "name": name,
            "name_zh": name_zh,
            "apply_now_url": apply_url,
        }

    # -- metadata -----------------------------------------------------------

    def _parse_metadata(self) -> dict:
        parsed = {HEADER_TO_KEY[h]: None for h in METADATA_HEADERS}
        lines, n = self._meta_region, len(self._meta_region)
        i = 0
        while i < n:
            m = re.match(r"^##\s+(.+)$", lines[i])
            if m:
                header, link = self._split_header(m.group(1).strip())
                if header in HEADER_TO_KEY:
                    values, i = self._collect_values(i + 1)
                    parsed[HEADER_TO_KEY[header]] = self._clean_value(
                        header, values, link
                    )
                    continue
            i += 1
        return parsed

    @staticmethod
    def _split_header(raw: str):
        """'[Tuition Fee](http://...)' -> ('Tuition Fee', 'http://...')."""
        m = re.match(r"^\[([^\]]+)\]\(([^)]+)\)$", raw)
        if m:
            return m.group(1), m.group(2)
        return raw, None

    def _collect_values(self, start: int):
        """Lines after a ``##`` header until next header / contact / outlines."""
        values = []
        j = start
        while j < len(self._meta_region):
            line = self._meta_region[j]
            s = line.strip()
            if (
                re.match(r"^##\s+", line)
                or is_contact_label(s)
                or s.startswith("### ")
            ):
                break
            if s:
                values.append(s)
            j += 1
        return values, j

    def _clean_value(self, header: str, values: list, link: str | None):
        if header == "Tuition Fee":
            return self._parse_tuition_fee(values, link)
        if header == "Application Deadline":
            return self._parse_deadline(values)
        if header == "Programme Website":
            return link or (values[0] if values else None)
        if header == "Year of Entry":
            return values[0] if values else None
        # generic text: strip footnote markers, collapse whitespace
        text = " ".join(values)
        text = text.replace("†", "").replace("^", "")
        text = re.sub(r"\s+", " ", text).strip()
        return text or None

    @staticmethod
    def _parse_tuition_fee(values: list, link: str | None):
        fee = {"local": None, "non_local": None, "source": link}
        for v in values:
            m = re.match(r"^-\s*Local Students:\s*(.+)$", v)
            if m:
                fee["local"] = m.group(1).strip()
                continue
            m = re.match(r"^-\s*Non-local Students:\s*(.+)$", v)
            if m:
                fee["non_local"] = m.group(1).strip()
                continue
            m = re.match(r"^-\s*Source:\s*(.+)$", v)
            if m:
                fee["source"] = m.group(1).strip()
        return fee

    @staticmethod
    def _parse_deadline(values: list):
        iso = [v for v in values if ISO_DEADLINE_RE.match(v)]
        raw = []
        for v in values:
            if not ISO_DEADLINE_RE.match(v) and v not in raw:
                raw.append(v)
        return {"iso": iso, "raw": raw}

    # -- contacts -----------------------------------------------------------

    def _parse_contacts(self) -> list:
        """Contact blocks as a list (roles repeat: co-leaders, multiple offices)."""
        contacts = []
        lines, n = self._meta_region, len(self._meta_region)
        i = 0
        while i < n:
            s = lines[i].strip()
            if is_contact_label(s):
                j = i + 1
                block = []
                while j < n:
                    ls = lines[j].strip()
                    if is_contact_label(ls) or ls.startswith("### ") or re.match(
                        r"^##\s+", lines[j]
                    ):
                        break
                    block.append(lines[j])
                    j += 1
                info = self._parse_contact_block(block)
                info["role"] = s
                contacts.append(info)
                i = j
                continue
            i += 1
        return contacts

    @staticmethod
    def _parse_contact_block(block: list) -> dict:
        info = {
            "name": None,
            "qualification": None,
            "email": None,
            "phone": [],
            "fax": [],
        }
        for raw in block:
            line = raw.strip()
            if not line:
                continue
            em = re.match(r"^\[([^\]]+)\]\(mailto:([^)]+)\)$", line)
            if em:
                info["email"] = em.group(1)
                continue
            tm = PHONE_TAGGED_RE.match(line)
            if tm:
                num = line.split("(")[0].strip()
                (info["phone"] if tm.group(1) == "Phone" else info["fax"]).append(num)
                continue
            if PHONE_BARE_RE.match(line):
                info["phone"].append(line)
                continue
            if line.startswith("*") and line.endswith("*") and len(line) > 2:
                if info["qualification"] is None:
                    info["qualification"] = line.strip("*").strip()
                continue
            if info["name"] is None:
                info["name"] = line
        return info


# ---------------------------------------------------------------------------
# Part 2 -- OutlineExtractor: TOC under "### Programme Outlines"
# ---------------------------------------------------------------------------


class OutlineExtractor:
    """Reads the section titles listed under ``### Programme Outlines``."""

    def __init__(self, lines: list):
        self.lines = lines

    def extract(self) -> list:
        start = None
        for i, line in enumerate(self.lines):
            if line.strip() == "### Programme Outlines":
                start = i
                break
        if start is None:
            return []
        titles = []
        for line in self.lines[start + 1 :]:
            s = line.strip()
            if s == "Outline":
                break
            m = re.match(r"^\*\s+(.+)$", s)
            if m:
                titles.append(m.group(1).strip())
        return titles


# ---------------------------------------------------------------------------
# Part 3 -- SectionSegmenter: TOC-driven body segmentation
# ---------------------------------------------------------------------------


class SectionSegmenter:
    """Splits the body into chunks using the TOC titles as ground truth."""

    def __init__(self, lines: list, outline_titles: list):
        self.lines = lines
        self.titles = set(outline_titles)
        self.order = outline_titles

    def segment(self):
        body = self._body()
        sections = []
        footnotes = []
        current = None
        buffer = []

        for line in body:
            s = line.strip()
            title = self._match_title(s)
            if title is not None:
                if current:
                    sections.append(self._make_section(current, buffer))
                current = title
                buffer = []
            else:
                buffer.append(line)

        if current:
            sections.append(self._make_section(current, buffer))

        sections, footnotes = self._peel_trailing_footnotes(sections, footnotes)
        return sections, footnotes

    def _body(self) -> list:
        # body starts after the standalone "Outline" marker
        for i, line in enumerate(self.lines):
            if line.strip() == "Outline":
                return self.lines[i + 1 :]
        # fallback: right after the last TOC bullet
        for i, line in enumerate(self.lines):
            if line.strip() == "### Programme Outlines":
                return self.lines[i + 1 :]
        return []

    def _match_title(self, s: str):
        if not s:
            return None
        s = s.lstrip("#").strip()
        if s in self.titles:
            return s
        return None

    @staticmethod
    def _make_section(title: str, buffer: list) -> dict:
        content = "\n".join(l.rstrip() for l in buffer)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()
        return {
            "title": title,
            "category": SECTION_CATEGORIES.get(title, DEFAULT_CATEGORY),
            "content": content,
            "char_count": len(content),
        }

    @staticmethod
    def _peel_trailing_footnotes(sections: list, footnotes: list):
        """Peel ``†``/``^``-prefixed footnote paragraphs from the file end.

        ``† Combined mode: ...`` is always the final line (55/64 files);
        ``^``-prefixed lines only when part of the trailing block (p66).
        Mid-file ``^``/``Remarks:`` lines (p46, p70, p93) are left in place.
        """
        if not sections:
            return sections, footnotes
        last = sections[-1]
        lines = last["content"].splitlines()
        peeled = []
        i = len(lines)
        while i > 0:
            s = lines[i - 1].strip()
            if not s:
                i -= 1
                continue
            if s.startswith("†") or s.startswith("^"):
                peeled.append(s)
                i -= 1
            else:
                break
        if not peeled:
            return sections, footnotes
        footnotes = list(reversed(peeled)) + footnotes
        rest = "\n".join(lines[:i]).strip()
        # drop a bare trailing "Remarks:" line left by the footnote block
        rest = re.sub(r"\n?Remarks:\s*$", "", rest).strip()
        sections[-1] = {
            "title": last["title"],
            "category": last["category"],
            "content": rest,
            "char_count": len(rest),
        }
        if not rest:
            sections.pop()
        return sections, footnotes


# ---------------------------------------------------------------------------
# Part 4 -- orchestration
# ---------------------------------------------------------------------------


def parse(path) -> dict:
    """Parse a single programme markdown file into a Programme object."""
    p = Path(path)
    if not p.exists() and not p.is_absolute():
        alt = DATA_DIR / p
        if alt.exists():
            p = alt
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()

    programme = ProgrammeParser(p).parse()
    outline = OutlineExtractor(lines).extract()
    sections, footnotes = SectionSegmenter(lines, outline).segment()

    programme["outline"] = outline
    programme["sections"] = sections
    programme["footnotes"] = footnotes
    return programme


def parse_all(markdown_dir=DATA_DIR) -> list:
    return [parse(f) for f in sorted(Path(markdown_dir).glob("p*.md"))]


def summarize(programmes: list) -> dict:
    total = len(programmes)
    missing_id = [p["programme_id"] for p in programmes if not p["programme_id"]]
    empty = [p["programme_id"] for p in programmes if not p["sections"]]
    partial = [
        (p["programme_id"], len(p["outline"]), len(p["sections"]))
        for p in programmes
        if p["outline"] and len(p["sections"]) != len(p["outline"])
    ]
    cat_counts: dict = {}
    for p in programmes:
        for s in p["sections"]:
            cat_counts[s["category"]] = cat_counts.get(s["category"], 0) + 1
    return {
        "files": total,
        "missing_programme_id": missing_id,
        "files_without_sections": empty,
        "outline_vs_sections_mismatch": partial,
        "section_categories": cat_counts,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Parse CityUHK programme markdown")
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="single markdown file to parse (default: run all and write JSON)",
    )
    args = parser.parse_args()

    if args.file:
        print(json.dumps(parse(args.file), ensure_ascii=False, indent=2))
        return

    programmes = parse_all()
    OUTPUT_PATH.write_text(
        json.dumps(programmes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(programmes)} programmes -> {OUTPUT_PATH}")
    print(json.dumps(summarize(programmes), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
