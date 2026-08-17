# CityUHK Taught Postgraduate Programme Pages — Information Schema

Source: 64 scraped programme markdown files in `data/markdown/` (`p02.md` … `p99.md`).
Purpose: a general schema for indexing, retrieval and metadata extraction of programme pages.

---

## 1. Document anatomy

Every programme file has the same skeleton:

```
[front matter]  P-code / English title / Chinese title / Apply Now link   (unheaded)
## metadata fields ...                                                     (11–13 ## headings)
[contact blocks] Programme Leader / Admissions Tutor / General Enquiries   (unheaded)
### Programme Outlines                                                     (TOC, 64/64)
[body knowledge sections]                                                  (plain-text section titles)
[footnotes]                                                                (misparsed as ## headings)
```

Body section titles are **plain text lines** (not markdown headings) in most files; a few use
`###` or `**bold**` pseudo-headings. The `Programme Outlines` TOC lists the section titles and is
the ground truth for segmentation.

## 2. Category overview

| Category | Meaning | Handling |
|---|---|---|
| **A — Structured metadata** | Fixed fields, consistent values | Parse to fields; **never embed** |
| **B — Retrieval knowledge** | Prose/table content users ask about | **Embed as child documents** |
| **C — Optional information** | Programme-specific extras | Store separately; embed tagged `optional` |

## 3. A — Structured metadata (never embedded)

| canonical_name | aliases | frequency | notes |
|---|---|---|---|
| Programme Identity | Programme Code (P02…), English/Chinese title, Apply Now | 64/64 (100%) | bilingual titles must be preserved |
| Year of Entry | — | 64/64 (100%) | single year (e.g. 2026) |
| Application Deadline | — | 64/64 (100%) | ISO timestamps + duplicated human-readable dates; some programmes have multiple rounds |
| Mode of Study | — | 64/64 (100%) | Full-time / Part-time / Combined; `†` footnote marker |
| Mode of Funding | Funding Mode | 64/64 (100%) | Govt / Non-govt funded |
| Indicative Intake Target | Intake Quota | 64/64 (100%) | integer |
| Minimum No. of Credits Required | Credits Required | 64/64 (100%) | may be per-award |
| Class Schedule | — | 60/64 (94%) | **absent in 4 files** |
| Normal Study Period | Duration | 64/64 (100%) | per-mode durations |
| Maximum Study Period | — | 64/64 (100%) | per-mode durations |
| Mode of Processing | Rolling Basis | 64/64 (100%) | prose policy, mostly boilerplate |
| Tuition Fee | Fees | 64/64 (100%) | local vs non-local rates + source URL |
| Programme Website | Official Website | 50/64 (78%) | heading is the hyperlink; **absent in 14 files** |
| Intermediate Award | Exit Award | 10/64 (16%) | optional field |
| Programme Contacts | Programme Leader, Associate Programme Leader, Admissions Tutor, General Enquiries | 56–63/64 | name / role / qualification / email blocks |
| Programme Outlines (TOC) | Outline, Contents | 64/64 (100%) | **exclude from index**; use only for section detection |
| IELTS/TOEFL footnote paragraphs | "Applicants are required to arrange for sending their IELTS result(s)…" | ~30/64 (47%) | **processing artifact** — merge into Entrance Requirements, not a section |

## 4. B — Retrieval knowledge sections (embed as child documents)

| canonical_name | aliases | frequency | notes |
|---|---|---|---|
| Programme Aims and Objectives | Programme Aims, Aims | 62/64 (97%) | missing in p18, p89; p96 adds ### Programme Aims |
| Programme Intended Learning Outcomes (PILOs) | PILOs, ILOs | 12/64 (19%) | format varies (### / bold line / bullets); sometimes nested in Aims or Course Description |
| Entrance Requirements | Admission Requirements, General Entrance Requirements, "To be eligible for admission, you must:" | 64/64 (100%) | universal; absorbs footnote paragraphs |
| English Proficiency Requirements | Language Requirements | 14/64 (22%) explicit, else embedded | TOEFL 79 / IELTS 6.0 cutoffs; keep as sub-chunk of Entrance Requirements |
| Local / Non-local Applicant Info | Local Applicants, Non-local Applicants | 2/64 (3%) | p17, p53 only; applicant-type-specific guidance |
| Course Description | Course List, Programme Structure & Courses, Core/Elective Courses | 57/64 (89%) | richest section; course tables (code/title/credits) |
| Programme Content | Programme Structure | 11/64 (17%) | **not a synonym** of Course Description — both coexist in p66, p77, p79, p96 |

## 5. C — Optional information (store separately)

| canonical_name | aliases | frequency | notes |
|---|---|---|---|
| Useful Links | Links, Related Links | 57/64 (89%) | store as link metadata; low-priority embedding |
| Scholarship | Hong Kong Future Talents Scholarship Scheme for Advanced Studies, Scholarships | 25/64 (39%) | 23 × scheme name + 2 × generic; may carry CGPA criteria |
| Bonus Features | Programme Features | 22/64 (34%) | marketing highlights |
| Professional Accreditation | Professional Recognition | 12/64 (19%) | 10 Accreditation + 2 Recognition; professional-body references |
| Career Prospects | Career, Graduate Outcomes | 6/64 (9%) | p43, p70, p76, p77, p79, p81 |
| Did You Know? | — | 7/64 (11%) | trivia; low retrieval value |
| Remarks | Additional Notes, Note | 7/64 (11%) | p12, p56, p66, p70, p77, p86, p93; p20 uses "Additional Notes" |
| Credit Unit Transfer | Credit Transfer | 6/64 (9%) | p52, p60, p64, p78, p82, p93 |
| Medium of Instruction and Assessment | Teaching Language | 2/64 (3%) | p46, p83 |
| Admissions Interview | Interview | 1/64 (2%) | p71 only |
| Curriculum Design | — | 1/64 (2%) | p71 only |

## 6. Extraction caveats

1. **Do not assume uniformity** — metadata presence varies (Class Schedule 94%, Programme Website 78%, Intermediate Award 16%).
2. **Programme Content ≠ Course Description** — keep the distinction; both can appear in one file.
3. **`##` footnote headings** (IELTS/TOEFL delivery text) are scrape artifacts in ~47% of files — normalize into Entrance Requirements.
4. **Duplicate dates** under Application Deadline (ISO + human-readable, repeated 2–3×) — dedupe during parsing.
5. **TOC duplication** — `Programme Outlines` repeats every body title; strip before embedding.
6. **Body titles are unmarked lines** — section segmentation must match against the TOC anchor list rather than markdown heading levels.

## 7. Recommended processing pipeline

1. Parse front matter + metadata `##` fields → structured metadata (category A).
2. Extract `Programme Outlines` TOC → section boundary map.
3. Segment body by TOC titles; merge stray footnotes into Entrance Requirements (A).
4. Classify segments → B child documents (embed with programme-id + section-name metadata) or C optional docs (embed tagged `optional: true`).
5. Index programme node with A fields; child docs carry parent programme id for filtered retrieval.
