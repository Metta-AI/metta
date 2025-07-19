import re

import requests
import vcr

ASANA_GITHUB_ATTACHMENT_ACTION_URL = "https://github.integrations.asana.plus/custom/v1/actions/widget"

# Configure VCR for Asana API calls
asana_vcr = vcr.VCR(
    cassette_library_dir="cassettes",
    record_mode=vcr.mode.ONCE,
    match_on=["uri", "method"],
    filter_headers=["authorization"],
    decode_compressed_response=True,
)


class AsanaTask:
    def __init__(
        self,
        asana_token: str,
        github_url_field_id: str,
        pr_author_field_id: str,
        asana_email_field_id: str,
        project_id: str,
        workspace_id: str,
        attachment_secret: str,
    ):
        self.asana_token = asana_token
        self.github_url_field_id = github_url_field_id
        self.pr_author_field_id = pr_author_field_id
        self.asana_email_field_id = asana_email_field_id
        self.project_id = project_id
        self.workspace_id = workspace_id
        self.attachment_secret = attachment_secret
        self._task_url = None

    @property
    def task_url(self):
        if self._task_url is None:
            raise Exception("task_url is not set on AsanaTask instance")
        return self._task_url

    @task_url.setter
    def task_url(self, value):
        self._task_url = value

    @property
    def task_gid(self):
        return self.extract_gid_from_url(self.task_url)

    def ensure(
        self,
        title: str,
        description: str,
        task_completed: bool,
        assignee: str | None,
        collaborators: list[str],
        github_url: str,
        pr_author: str | None,
        urls: list[str],
    ) -> str:
        existing = None
        if urls:
            for url in urls:
                validated = self.validate(url, github_url)
                if validated:
                    existing = validated
                    break
            print("None of the existing Asana URLs in description are valid for this project")

        if not existing:
            existing = self.search(
                github_url=github_url,
            )

        if existing:
            self.update_if_needed(
                existing,
                title,
                description,
                task_completed,
                assignee,
                github_url,
                pr_author,
            )
            self.task_url = existing["permalink_url"]
            return self.task_url

        new_url = self.create(
            title,
            description,
            task_completed,
            assignee,
            collaborators,
            github_url,
            pr_author,
        )
        self.task_url = new_url
        return new_url

    def validate(
        self,
        url: str,
        github_url: str,
    ) -> dict | None:
        gid = self.extract_gid_from_url(url)
        if not gid:
            return None
        api_url = f"https://app.asana.com/api/1.0/tasks/{gid}"
        headers = {
            "Authorization": f"Bearer {self.asana_token}",
            "Content-Type": "application/json",
        }
        params = {
            "opt_fields": "permalink_url,custom_fields,name,notes,modified_at,completed,"
            "assignee.email,followers.email,projects.gid",
        }

        with asana_vcr.use_cassette(f"validate_task_{gid}.yaml"):
            response = requests.get(api_url, headers=headers, params=params, timeout=30)
            if response.status_code != 200:
                return None
            data = response.json()["data"]
            projects = [project["gid"] for project in data.get("projects", [])]
            if self.project_id not in projects:
                return None
            custom_fields = data.get("custom_fields", [])
            task_github_url = None
            for field in custom_fields:
                if field.get("gid") == self.github_url_field_id:
                    task_github_url = field.get("text_value")
                    break
            if task_github_url != github_url:
                return None
            return data

    def search(
        self,
        github_url: str,
    ) -> dict | None:
        api_url = f"https://app.asana.com/api/1.0/workspaces/{self.workspace_id}/tasks/search"
        headers = {
            "Authorization": f"Bearer {self.asana_token}",
            "Content-Type": "application/json",
        }
        params = {
            "opt_fields": "permalink_url,custom_fields,name,notes,modified_at,completed,assignee.email,followers.email",
            "limit": 100,
            "sort_by": "created_at",
            "projects.any": self.project_id,
            f"custom_fields.{self.github_url_field_id}.value": github_url,
        }

        with asana_vcr.use_cassette("search_tasks.yaml"):
            response = requests.get(api_url, headers=headers, params=params, timeout=30)
            if response.status_code != 200:
                return None
            data = response.json()
            results = data["data"]
            if results:
                return results[0]
            return None

    def create(
        self,
        title: str,
        description: str,
        completed: bool,
        assignee: str | None,
        collaborators: list[str],
        github_url: str,
        pr_author: str | None,
    ) -> str:
        api_url = "https://app.asana.com/api/1.0/tasks"
        headers = {
            "Authorization": f"Bearer {self.asana_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "data": {
                "name": title,
                "notes": description,
                "workspace": self.workspace_id,
                "projects": [self.project_id],
                "followers": collaborators,
            }
        }
        if assignee:
            payload["data"]["assignee"] = assignee
        custom_fields = {}
        custom_fields[self.github_url_field_id] = github_url
        custom_fields[self.pr_author_field_id] = pr_author
        payload["data"]["custom_fields"] = custom_fields
        if completed:
            payload["data"]["completed"] = True

        with asana_vcr.use_cassette("create_task.yaml"):
            response = requests.post(api_url, json=payload, headers=headers, timeout=30)
            if response.status_code == 201:
                url = response.json()["data"]["permalink_url"]
                self.ensure_github_url_in_task(url, title, github_url)
                return url
            else:
                raise Exception(f"Asana API Error (create): {response.status_code} - {response.text}")

    def update(
        self,
        gid: str,
        title: str,
        description: str,
        completed: bool,
        assignee: str | None,
        github_url: str,
        pr_author: str | None,
    ):
        api_url = f"https://app.asana.com/api/1.0/tasks/{gid}"
        headers = {
            "Authorization": f"Bearer {self.asana_token}",
            "Content-Type": "application/json",
        }
        payload = {"data": {"name": title, "notes": description, "completed": completed}}
        if assignee:
            payload["data"]["assignee"] = assignee
        custom_fields = {}
        custom_fields[self.github_url_field_id] = github_url
        custom_fields[self.pr_author_field_id] = pr_author
        if custom_fields:
            payload["data"]["custom_fields"] = custom_fields

        with asana_vcr.use_cassette(f"update_task_{gid}.yaml"):
            response = requests.put(api_url, json=payload, headers=headers, timeout=30)
            if response.status_code != 200:
                raise Exception(f"Asana API Error (update): {response.status_code} - {response.text}")

    def update_if_needed(
        self,
        data: dict,
        title: str,
        description: str,
        task_completed: bool,
        assignee: str | None,
        github_url: str,
        pr_author: str | None,
    ) -> None:
        current_title = data.get("name") or ""
        current_notes = data.get("notes") or ""
        current_completed = data.get("completed") or False
        current_assignee = (data.get("assignee") or {}).get("email") or ""
        if (
            current_title != title
            or current_notes != description
            or current_completed != task_completed
            or current_assignee != assignee
        ):
            self.update(
                data["gid"],
                title,
                description,
                task_completed,
                assignee,
                github_url,
                pr_author,
            )

    def ensure_github_url_in_task(
        self,
        url: str,
        title: str,
        github_url: str,
    ) -> dict | None:
        github_url_number = github_url.split("pull/")[-1]
        if not github_url_number.isdigit():
            return None
        headers = {
            "Authorization": f"Bearer {self.attachment_secret}",
            "Content-Type": "application/json",
        }
        payload = {
            "allowedProjects": [self.project_id],
            "blockedProjects": [],
            "pullRequestName": title,
            "pullRequestDescription": url,
            "pullRequestNumber": int(github_url_number),
            "pullRequestURL": github_url,
        }

        with asana_vcr.use_cassette("ensure_github_url_in_task.yaml"):
            response = requests.post(ASANA_GITHUB_ATTACHMENT_ACTION_URL, json=payload, headers=headers, timeout=30)
            if response.status_code == 201:
                return response.json()
            else:
                return None

    def get_comments(self, task_id: str):
        """
        Fetches comments for an Asana task
        Args:
            task_id (str): Asana task ID
        Returns:
            List[Dict]: List of comment dictionaries
        """
        from datetime import datetime

        url = f"https://app.asana.com/api/1.0/tasks/{task_id}/stories"
        headers = {"Authorization": f"Bearer {self.asana_token}", "Content-Type": "application/json"}
        params = {
            "opt_fields": "text,html_text,created_by.name,created_by.email,created_at,type,resource_subtype,is_pinned"
        }

        with asana_vcr.use_cassette(f"get_comments_{task_id}.yaml"):
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            comments = [
                story
                for story in data["data"]
                if story.get("type") == "comment" or story.get("resource_subtype") == "comment_added"
            ]
            ret = [
                {
                    "id": comment.get("gid"),
                    "text": comment.get("text", ""),
                    "html_text": comment.get("html_text", ""),
                    "author": {
                        "name": comment.get("created_by", {}).get("name", "Unknown"),
                        "email": comment.get("created_by", {}).get("email"),
                    },
                    "created_at": datetime.fromisoformat(comment.get("created_at", "").replace("Z", "+00:00")),
                    "is_pinned": comment.get("is_pinned", False),
                    "review_id": None,  # You may want to add review_id extraction logic if needed
                }
                for comment in comments
            ]
            print(f"comments in asana: {ret}")

            return ret

    def asana_comments_with_links(self):
        comments = self.get_comments(self.task_gid)
        return [comment for comment in comments if comment["review_id"] is not None]

    def synchronize_comments_in_asana_as_multiple_blocks(self, comments_from_github: list[dict]) -> None:
        """
        Synchronize review comments in Asana as multiple blocks.
        Args:
            comments_from_github (list[dict]): List of review comments from GitHub
        """
        print("[synchronize_comments_in_asana_as_multiple_blocks] Starting with:")
        asana_comments_with_links = self.asana_comments_with_links()
        print(f"  asana_comments_with_links: {len(asana_comments_with_links)} comments")
        print(f"  comments_from_github: {len(comments_from_github)} reviews")

        if not comments_from_github:
            print("[synchronize_comments_in_asana_as_multiple_blocks] No GitHub comments to process")
            return

        api_url = f"https://app.asana.com/api/1.0/tasks/{self.task_gid}"
        headers = {"Authorization": f"Bearer {self.asana_token}", "Content-Type": "application/json"}

        # Create a map of review IDs to existing Asana comments
        existing_comments_by_review_id = {
            comment["review_id"]: comment for comment in asana_comments_with_links if comment["review_id"] is not None
        }
        print(f"[s] Existing comments by review ID: {list(existing_comments_by_review_id.keys())}")

        # Process each GitHub review
        for github_review in comments_from_github:
            review_id = github_review["id"]
            review_body = github_review.get("text", "")
            github_user = github_review["user"]
            review_state = github_review["action"]
            github_timestamp = github_review["timestamp"]

            print(f"[s] Processing review {review_id} from {github_user} ({review_state})")

            # Format the review for Asana
            from pr_gh_to_asana import format_github_review_body_for_asana

            formatted_comment = format_github_review_body_for_asana(
                review_body, github_user, review_state, review_id, github_timestamp
            )

            if review_id in existing_comments_by_review_id:
                print(
                    f"[synchronize_comments_in_asana_as_multiple_blocks] Review {review_id} has existing Asana comment"
                )
                # Update existing comment if content differs
                existing_comment = existing_comments_by_review_id[review_id]
                if existing_comment["text"] != formatted_comment:
                    print(f"[s] Updating existing comment for review {review_id}")
                    story_id = existing_comment["id"]
                    url = f"https://app.asana.com/api/1.0/stories/{story_id}"
                    payload = {"data": {"html_text": formatted_comment}}
                    try:
                        print(payload)
                        with asana_vcr.use_cassette(f"update_comment_{story_id}.yaml"):
                            response = requests.put(url, headers=headers, json=payload)
                            if response.status_code == 200:
                                print(f"Updated Asana comment {story_id} for review {review_id}")
                            else:
                                print(f"Failed to update comment {story_id}: {response.status_code} - {response.text}")
                    except requests.exceptions.RequestException as e:
                        print(f"Error updating Asana comment {story_id}: {e}")
                else:
                    print(
                        f"[synchronize_comments_in_asana_as_multiple_blocks] Review {review_id} comment is up to date"
                    )
            else:
                print(f"[s] Review {review_id} has no existing Asana comment")
                # Check if we should add this comment (don't add out of order)
                # Find the last existing comment that matches a GitHub review
                last_matching_review_id = None
                for existing_comment in asana_comments_with_links:
                    if existing_comment["review_id"] is not None:
                        if existing_comment["review_id"] in {r["id"] for r in comments_from_github}:
                            last_matching_review_id = existing_comment["review_id"]

                # Check if this review comes before the last matching review
                should_add = True
                if last_matching_review_id is not None:
                    # Find the position of the last matching review in the GitHub reviews list
                    last_matching_index = -1
                    for i, review in enumerate(comments_from_github):
                        if review["id"] == last_matching_review_id:
                            last_matching_index = i
                            break

                    # Find the position of current review in the GitHub reviews list
                    current_review_index = -1
                    for i, review in enumerate(comments_from_github):
                        if review["id"] == review_id:
                            current_review_index = i
                            break

                    # Only add if current review comes after the last matching review
                    if current_review_index <= last_matching_index:
                        print(f"[s] Review {review_id} comes before/at last review {last_matching_review_id}, skipping")
                        should_add = False

                if should_add:
                    print(f"[s] Adding new comment for review {review_id}")
                    # Create new comment
                    url = f"{api_url}/stories"
                    payload = {"data": {"html_text": formatted_comment, "type": "comment"}}
                    print(payload)

                    try:
                        with asana_vcr.use_cassette(f"add_comment_{review_id}.yaml"):
                            response = requests.post(url, headers=headers, json=payload)
                            response.raise_for_status()
                            print(f"Added new Asana comment for review {review_id}")
                    except requests.exceptions.RequestException as e:
                        print(f"Error adding Asana comment for review {review_id}: {e}")
                        print(payload)
                else:
                    print(f"[s] Skipped adding comment for review {review_id} due to ordering constraint")

    @staticmethod
    def extract_asana_gid_from_url(task_url: str) -> str:
        # Try Format 1: https://app.asana.com/0/project_id/task_id
        match = re.search(r"https://app\.asana\.com/\d+/\d+/(\d+)(?:/|$)", task_url)
        if match:
            return match.group(1)

        # Try Format 2: https://app.asana.com/1/workspace_id/project/project_id/task/task_id
        match = re.search(r"https://app\.asana\.com/\d+/\d+/project/\d+/task/(\d+)(?:/|$)", task_url)
        if match:
            return match.group(1)

        raise Exception(f"Could not extract task ID from URL: {task_url}")
