import requests
import csv
import time
import random
from datetime import datetime
from tqdm import tqdm


BASE_URL = "https://api.reddit.com"

headers = {
    "User-Agent": "Mozilla/5.0 (RedditCommentScraper/1.0)"
}

SUBREDDITS = ["politics", "TrueOffMyChest", "mildlyinfuriating"]
LIMIT_POST = 10
MIN_WORDS = 15
MAX_WORDS = 150
MAX_RETRIES = 5
MAX_COMMENTS = 200


def safe_request(url):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                return response

            elif response.status_code == 429:
                wait = 2 ** attempt + random.uniform(0.5, 1.5)
                print(f"Rate limited. Retrying in {wait:.2f}s...")
                time.sleep(wait)

            elif response.status_code == 403:
                wait = 5 + random.uniform(1, 3)
                print(f"Blocked (403). Cooling down {wait:.2f}s...")
                time.sleep(wait)

            else:
                print(f"Status {response.status_code}, retrying...")

        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            print(f"Network error: {e}, retrying in {wait}s...")
            time.sleep(wait)

    print(f"Failed after {MAX_RETRIES} attempts: {url}")
    return None


def get_posts(subreddit):
    posts = []

    for sort in ["hot", "new", "top"]:
        url = f"{BASE_URL}/r/{subreddit}/{sort}?limit={LIMIT_POST}"

        response = safe_request(url)
        if not response:
            continue

        data = response.json()

        for post in data["data"]["children"]:
            p = post["data"]
            posts.append({
                "post_id": p["id"],
                "post_title": p.get("title", ""),
                "subreddit": subreddit
            })

        time.sleep(random.uniform(0.8, 1.5))

    unique = {p["post_id"]: p for p in posts}
    return list(unique.values())


def extract_comments(comment_list, post_meta, depth=0):
    results = []

    for c in comment_list:
        if c["kind"] == "more":
            continue

        if c["kind"] != "t1":
            continue

        data = c["data"]
        body = data.get("body", "").strip()

        if not body or body in ("[deleted]", "[removed]"):
            pass
        else:
            word_count = len(body.split())

            if MIN_WORDS <= word_count <= MAX_WORDS:
                timestamp = datetime.utcfromtimestamp(
                    data.get("created_utc")
                ).isoformat()

                results.append({
                    "post_id": post_meta["post_id"],
                    "post_title": post_meta["post_title"],
                    "subreddit": post_meta["subreddit"],
                    "timestamp": timestamp,
                    "platform": "reddit",
                    "user_id": data.get("author"),
                    "text": body,
                    "word_count": word_count,
                    "depth": depth,
                    "label": ""
                })

        replies = data.get("replies")
        if replies and isinstance(replies, dict):
            reply_list = replies["data"]["children"]
            results.extend(extract_comments(reply_list, post_meta, depth=depth + 1))

    return results


def get_comments(post_meta):
    url = f"{BASE_URL}/r/{post_meta['subreddit']}/comments/{post_meta['post_id']}?limit=500"

    response = safe_request(url)
    if not response:
        return []

    data = response.json()
    comment_list = data[1]["data"]["children"]

    return extract_comments(comment_list, post_meta, depth=0)


def run_extract(output_file):
    all_comments = []

    quota_per_sub = MAX_COMMENTS // len(SUBREDDITS)
    remainder = MAX_COMMENTS % len(SUBREDDITS)

    for i, subreddit in enumerate(SUBREDDITS):
        sub_quota = quota_per_sub + (remainder if i == 0 else 0)
        sub_comments = []

        print(f"Fetching posts from r/{subreddit} (quota: {sub_quota})...")
        posts = get_posts(subreddit)
        print(f"Total posts: {len(posts)}")

        for post in tqdm(posts, desc=f"r/{subreddit}"):
            if len(sub_comments) >= sub_quota:
                print(f"r/{subreddit} quota reached ({sub_quota}), moving on.")
                break

            comments = get_comments(post)

            for c in comments:
                if len(sub_comments) >= sub_quota:
                    break
                sub_comments.append(c)

            time.sleep(random.uniform(1.0, 2.0))

        if len(sub_comments) < sub_quota:
            print(f"r/{subreddit} only got {len(sub_comments)}/{sub_quota} — try increasing LIMIT_POST.")

        print(f"r/{subreddit} collected: {len(sub_comments)}")
        all_comments.extend(sub_comments)

    print("Saving to CSV...")

    fieldnames = [
        "post_id",
        "post_title",
        "subreddit",
        "timestamp",
        "platform",
        "user_id",
        "text",
        "word_count",
        "depth",
        "label"
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_comments)

    print(f"Done. Total comments saved: {len(all_comments)}")