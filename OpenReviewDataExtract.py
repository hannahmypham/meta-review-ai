import openreview
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
USERNAME = '' 
PASSWORD = '!'
VENUE_ID = 'ICLR.cc/2025/Conference'
OUTPUT_CSV = 'iclr_2025_detailed_reviews.csv'

client = openreview.api.OpenReviewClient(
    baseurl='https://api2.openreview.net', 
    username=USERNAME, 
    password=PASSWORD
)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def format_date(ms):
    """Converts millisecond timestamps to YYYY-MM-DD format."""
    if not ms: return "N/A"
    return datetime.fromtimestamp(ms / 1000.0).strftime('%Y-%m-%d %H:%M:%S')

def parse_rating(r_obj):
    """Extracts the leading integer from rating strings."""
    if r_obj is None: return None
    try:
        val = str(r_obj).split(':')[0]
        return int(val)
    except (ValueError, IndexError):
        return None

def clean_text(text):
    """Removes newlines and tabs to keep the CSV rows aligned."""
    if not text: return ""
    return " ".join(str(text).replace('\n', ' ').replace('\t', ' ').split())

# ==========================================
# 3. MAIN SCRAPER
# ==========================================
def scrape_iclr_data():
    all_rows = []

    print(f"Downloading ICLR 2025 data...")
    submissions = client.get_all_notes(
        invitation=f'{VENUE_ID}/-/Submission',
        details='replies'
    )

    for sub in tqdm(submissions, desc="Processing Papers"):
        # Filter out withdrawn papers
        if sub.ddate or 'Withdrawn' in sub.content.get('venue', {}).get('value', ''):
            continue

        replies = sub.details.get('replies', [])
        
        # --- A. Extract Paper Metadata (Shared across reviews) ---
        paper_metadata = {
            'paper_id': sub.id,
            'title': sub.content.get('title', {}).get('value', ''),
            'submission_time': format_date(sub.cdate), # CREATION DATE
            'official_label': 'Pending',
            'meta_review': '',
        }
        
        # Pre-scan for Decision and Meta-Review
        for r in replies:
            inv = r.get('invitations', [])
            if any('Decision' in i for i in inv):
                paper_metadata['official_label'] = r.get('content', {}).get('decision', {}).get('value', 'Reject')
            if any('Meta_Review' in i for i in inv):
                paper_metadata['meta_review'] = clean_text(r.get('content', {}).get('metareview', {}).get('value', ''))

        # --- B. Create a Row for Each Official Review ---
        for reply in replies:
            inv = reply.get('invitations', [])
            
            if any('Official_Review' in i for i in inv):
                reply_id = reply.get('id')
                content = reply.get('content', {})
                
                sig = reply.get('signatures', [''])[0]
                reviewer_label = sig.split('/')[-1] if '/' in sig else sig

                # --- C. Match Author Rebuttal to THIS Specific Review ---
                author_rebuttals = []
                for comment in replies:
                    is_author = 'Authors' in str(comment.get('signatures'))
                    if comment.get('replyto') == reply_id and is_author:
                        text = comment.get('content', {}).get('comment', {}).get('value', '')
                        author_rebuttals.append(clean_text(text))

                # --- D. Rating Evolution ---
                final_rating = parse_rating(content.get('rating', {}).get('value'))
                try:
                    revisions = client.get_note_revisions(note_id=reply_id)
                    original_rating = parse_rating(revisions[-1].content.get('rating', {}).get('value'))
                except:
                    original_rating = final_rating

                # --- E. Final Row Construction ---
                row = paper_metadata.copy()
                row.update({
                    'reviewer_id': reviewer_label,
                    'original_rating': original_rating,
                    'final_rating': final_rating,
                    'score_did_change': (original_rating != final_rating) if original_rating else False,
                    # Core text fields
                    'summary': clean_text(content.get('summary', {}).get('value', '')),
                    'official_comment': clean_text(content.get('official_comment', {}).get('value', '')),
                    'strengths': clean_text(content.get('strengths', {}).get('value', '')),
                    'weaknesses': clean_text(content.get('weaknesses', {}).get('value', '')),
                    'questions': clean_text(content.get('questions', {}).get('value', '')),
                    # Categorical / numerical (raw form values, e.g. "8: Accept")
                    'soundness': content.get('soundness', {}).get('value', ''),
                    'presentation': content.get('presentation', {}).get('value', ''),
                    'contribution': content.get('contribution', {}).get('value', ''),
                    'confidence': content.get('confidence', {}).get('value', ''),
                    'rating_full': content.get('rating', {}).get('value', ''),
                    # Ethics & compliance
                    'flag_for_ethics_review': content.get('flag_for_ethics_review', {}).get('value', ''),
                    'code_of_conduct': content.get('code_of_conduct', {}).get('value', ''),
                    # Metadata
                    'author_response_to_this_review': " [SEP] ".join(author_rebuttals),
                    'review_last_modified': format_date(reply.get('mdate') or reply.get('cdate'))
                })
                all_rows.append(row)

    # Export to CSV
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print(f"\nSuccess! File saved as {OUTPUT_CSV}")

if __name__ == "__main__":
    scrape_iclr_data()