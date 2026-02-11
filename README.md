# OpenReview Rating – Variable Reference

This project pulls ICLR conference review data from the OpenReview API and exports it to CSV (and optionally JSONL) for analysis. Below are the variables (columns) produced by the scripts.

---

## Output: `iclr_2025_detailed_reviews.csv` (from `OpenReviewDataExtract.py`)

One **row per official review**. Each row includes paper-level metadata (repeated for every review of that paper) and review-specific fields.

### Paper-level (same for all reviews of a paper)

| Variable | Type | Description |
|----------|------|-------------|
| `paper_id` | string | OpenReview note ID of the submission (e.g. `zzR1Uskhj0`). |
| `title` | string | Paper title. |
| `submission_time` | string | When the paper was submitted (`YYYY-MM-DD HH:MM:SS`), or `N/A`. |
| `official_label` | string | Final decision: e.g. `Accept (Poster)`, `Reject`, `Pending`. |
| `meta_review` | string | Meta-review text (reason for acceptance/rejection), cleaned (no newlines/tabs). |

### Review-level (per reviewer)

| Variable | Type | Description |
|----------|------|-------------|
| `reviewer_id` | string | Reviewer identifier (from OpenReview signature, e.g. `~Reviewer_ABC1`). |
| `original_rating` | int or null | Numeric rating from the **first** version of the review (from note revisions). |
| `final_rating` | int or null | Numeric rating from the **current** version of the review (e.g. 1–10). |
| `score_did_change` | bool | `True` if the reviewer changed their numeric rating between first and final version. |
| `summary` | string | Reviewer’s summary of the paper (cleaned text). |
| `official_comment` | string | Official comment field from the review form (cleaned text). |
| `strengths` | string | Strengths section (cleaned text). |
| `weaknesses` | string | Weaknesses section (cleaned text). |
| `questions` | string | Questions for the authors (cleaned text). |
| `soundness` | string | Raw form value for soundness (e.g. `"3: Good"`). |
| `presentation` | string | Raw form value for presentation. |
| `contribution` | string | Raw form value for contribution. |
| `confidence` | string | Raw form value for confidence (e.g. `"4: You are confident in your assessment"`). |
| `rating_full` | string | Full rating string as stored (e.g. `"8: Accept"`). |
| `flag_for_ethics_review` | string | Whether the review flagged the paper for ethics review (form value). |
| `code_of_conduct` | string | Code-of-conduct / compliance field from the form. |
| `author_response_to_this_review` | string | All author rebuttal text for this review, joined with ` [SEP] `. |
| `review_last_modified` | string | Last modification time of the review note (`YYYY-MM-DD HH:MM:SS` or `N/A`). |

**Note:** “Cleaned” text has newlines and tabs replaced by spaces and collapsed so it stays on one line in CSV.

## Quick reference: where each variable comes from

- **OpenReview submission note:** `paper_id`, `title`, `submission_time` (from `cdate`).
- **OpenReview official review reply (content):** `summary`, `official_comment`, `strengths`, `weaknesses`, `questions`, `soundness`, `presentation`, `contribution`, `confidence`, `rating_full`, `flag_for_ethics_review`, `code_of_conduct`.
- **OpenReview note revisions:** `original_rating` (first revision), `final_rating` (current).
- **OpenReview author comments with `replyto` = review id:** `author_response_to_this_review`.
- **OpenReview decision / meta-review replies:** `official_label`, `meta_review`.

If a field is missing in the OpenReview form for a given venue or year, the corresponding column will be empty in the CSV.
