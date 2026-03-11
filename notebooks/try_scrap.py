import requests
import csv
import time
from datetime import datetime

headers = {
    "User-Agent": "Mozilla/5.0 (RedditCommentScraper/1.0)"
}

SUBREDDIT = "politics"
LIMIT_POST = 25
MIN_WORDS = 15
OUTPUT_FILE = "reddit_politics_comments.csv"


def get_posts(subreddit=SUBREDDIT, limit=LIMIT_POST):
    posts = []

    for sort in ["hot", "new", "top"]:
        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}"
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"Gagal fetch {sort}")
            continue

        data = response.json()

        for post in data["data"]["children"]:
            p = post["data"]

            posts.append({
                "post_id": p["id"],
                "post_title": p.get("title", "")
            })

        time.sleep(1)

    # deduplicate
    unique = {p["post_id"]: p for p in posts}

    return list(unique.values())


def extract_comments(comment_list, post_meta):
    results = []

    for c in comment_list:

        if c["kind"] != "t1":
            continue

        data = c["data"]

        body = data.get("body", "").strip()

        if not body or body in ("[deleted]", "[removed]"):
            continue

        word_count = len(body.split())

        if word_count < MIN_WORDS:
            continue

        timestamp = datetime.utcfromtimestamp(
            data.get("created_utc")
        ).isoformat()

        results.append({
            "post_id": post_meta["post_id"],
            "post_title": post_meta["post_title"],
            "timestamp": timestamp,
            "platform": "reddit",
            "user_id": data.get("author"),
            "text": body,
            "word_count": word_count,
            "label": ""   # unlabeled
        })

    return results


def get_comments(post_meta, subreddit=SUBREDDIT):

    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_meta['post_id']}.json?limit=500"

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return []

    data = response.json()

    comment_list = data[1]["data"]["children"]

    return extract_comments(comment_list, post_meta)


# ===== MAIN =====

print("Fetching posts...")
posts = get_posts()

print(f"Total post: {len(posts)}")

all_comments = []

for i, post in enumerate(posts):

    print(f"[{i+1}/{len(posts)}] Scraping: {post['post_title'][:60]}")

    comments = get_comments(post)

    all_comments.extend(comments)

    print(f"  → Didapat {len(comments)} komentar")

    time.sleep(1.5)


fieldnames = [
    "post_id",
    "post_title",
    "timestamp",
    "platform",
    "user_id",
    "text",
    "word_count",
    "label"
]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_comments)


print("\nSelesai!")
print(f"Total komentar: {len(all_comments)}")
print(f"File: {OUTPUT_FILE}")