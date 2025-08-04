import json
import re

import requests


class PullRequest:
    def __init__(self, repo, pr_number, github_token):
        self.repo = repo
        self.pr_number = pr_number
        self.github_token = github_token
        self._fetch_data()
        self._parse_data()

    def _fetch_data(self):
        url = f"https://api.github.com/repos/{self.repo}/pulls/{self.pr_number}"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.github_token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        self.data = response.json()

        # Fetch PR comments (issue comments)
        comments_url = f"https://api.github.com/repos/{self.repo}/issues/{self.pr_number}/comments"
        comments_response = requests.get(comments_url, headers=headers)
        if comments_response.status_code == 200:
            self.data["retrieved_comments"] = comments_response.json()
        else:
            print(f"Failed to fetch PR comments: {comments_response.status_code} - {comments_response.text}")
            self.data["retrieved_comments"] = []

        # Fetch PR reviews
        reviews_url = f"https://api.github.com/repos/{self.repo}/pulls/{self.pr_number}/reviews"
        reviews_response = requests.get(reviews_url, headers=headers)
        if reviews_response.status_code == 200:
            self.data["retrieved_reviews"] = reviews_response.json()
        else:
            print(f"Failed to fetch PR reviews: {reviews_response.status_code} - {reviews_response.text}")
            self.data["retrieved_reviews"] = []

        # Fetch PR timeline
        timeline_url = f"https://api.github.com/repos/{self.repo}/issues/{self.pr_number}/timeline"
        timeline_response = requests.get(timeline_url, headers=headers)
        if timeline_response.status_code == 200:
            self.data["retrieved_timeline"] = timeline_response.json()
        else:
            print(f"Failed to fetch PR timeline: {timeline_response.status_code} - {timeline_response.text}")
            self.data["retrieved_timeline"] = []

    def _parse_data(self):
        d = self.data
        self.body = d.get("body", "") or ""
        self.title = d.get("title", "") or ""
        self.author = (d.get("user", "") or {}).get("login", "") or ""
        self.assignees = [assignee["login"] for assignee in d.get("assignees", [])] or []
        self.draft = d.get("draft", False)
        self.state = d.get("state", "") or ""
        self.is_open = self.state == "open"
        self.task_completed = not self.is_open

        self.reviewers = [
            review["user"]["login"]
            for review in d.get("retrieved_reviews", [])
            if review.get("user") and review["user"].get("login")
        ]
        self.commenters = [
            review["user"]["login"]
            for review in d.get("retrieved_comments", [])
            if review.get("user") and review["user"].get("login")
        ]
        self.github_logins = set(self.assignees + self.reviewers + self.commenters + [self.author])

        self.retrieved_reviews = d.get("retrieved_reviews", [])
        self.retrieved_timeline = d.get("retrieved_timeline", [])
        self.events = self._build_events()

    def _build_events(self):
        reviews = [
            {
                "type": "review",
                "timestamp": r["submitted_at"],
                "user": r["user"]["login"],
                "text": r["body"],
                "action": r["state"],
                "id": r["id"],
                "html_url": r["html_url"],
            }
            for r in self.retrieved_reviews
            if r.get("state") in ["APPROVED", "CHANGES_REQUESTED"]
        ]
        review_requested = [
            {
                "type": "review_requested",
                "timestamp": e["created_at"],
                "user": e["actor"]["login"],
                "requested_reviewer": e["requested_reviewer"]["login"],
                "action": "re-requested",
                "text": f" review from {e['requested_reviewer']['login']}",
                "id": e["id"],
            }
            for e in self.retrieved_timeline
            if e.get("event") == "review_requested"
        ]
        events = sorted(reviews + review_requested, key=lambda x: x["timestamp"])
        seen_review_users = set()
        filtered_events = []
        for event in events:
            if event["type"] == "review":
                seen_review_users.add(event["user"])
                filtered_events.append(event)
            elif event["type"] == "review_requested" and event["requested_reviewer"] in seen_review_users:
                filtered_events.append(event)
        return filtered_events

    @property
    def last_event(self):
        return self.events[-1] if self.events else None

    # probably want to remove some of this as it's now being written to the VCR cassette
    def print_debug_info(self):
        print("Pull request JSON:")
        print(json.dumps(self.data, indent=2))
        print(f"assignees: {self.assignees}")
        print(f"author: {self.author}")
        print(f"reviewers: {self.reviewers}")
        print(f"commenters: {self.commenters}")
        print(f"github_logins: {self.github_logins}")
        print(f"event stream: {self.events}")
        print(f"last event: {self.last_event}")

    @staticmethod
    def _extract_asana_urls_from_description(description: str) -> list[str]:
        """Extract Asana task URLs from the description text."""
        if not description:
            return []

        # Pattern to match Asana task URLs in the format used by update-pr-description
        # Matches: [Asana Task](https://app.asana.com/0/123456789/123456789)

        asana_pattern = r"\[Asana Task\]\((https://app\.asana\.com/\d+/\d+/project/\d+/task/\d+\))"

        urls = re.findall(asana_pattern, description)
        print(f"Found {len(urls)} Asana URLs in description: {description}")

        return urls

    @property
    def asana_urls(self):
        return self._extract_asana_urls_from_description(self.body)
