import re

import requests

ASANA_GITHUB_ATTACHMENT_ACTION_URL = "https://github.integrations.asana.plus/custom/v1/actions/widget"


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
        print(f"[__init__] Initializing AsanaTask with project_id={project_id}, workspace_id={workspace_id}")
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
        print(f"[task_url.setter] Setting task_url to: {value}")
        self._task_url = value

    @property
    def task_gid(self):
        gid = self.extract_gid_from_url(self.task_url)
        print(f"[task_gid] Extracted task_gid: {gid} from URL: {self.task_url}")
        return gid

    @staticmethod
    def extract_gid_from_url(task_url: str) -> str:
        print(f"[extract_gid_from_url] extract_gid_from_url() called with task_url='{task_url}'")
        # Try Format 1: https://app.asana.com/0/project_id/task_id
        match = re.search(r"https://app\.asana\.com/\d+/\d+/(\d+)(?:/|$)", task_url)
        if match:
            gid = match.group(1)
            print(f"[extract_gid_from_url] Extracted GID using Format 1: {gid}")
            return gid

        # Try Format 2: https://app.asana.com/1/workspace_id/project/project_id/task/task_id
        match = re.search(r"https://app\.asana\.com/\d+/\d+/project/\d+/task/(\d+)", task_url)
        if match:
            gid = match.group(1)
            print(f"[extract_gid_from_url] Extracted GID using Format 2: {gid}")
            return gid

        print(f"[extract_gid_from_url] Could not extract task ID from URL: {task_url}")
        raise Exception(f"Could not extract task ID from URL: {task_url}")

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
        print(f"[ensure] ensure() called with title='{title}', github_url='{github_url}', urls={urls}")
        existing = None
        if urls:
            print(f"[ensure] Checking {len(urls)} existing URLs for validation")
            for url in urls:
                print(f"[ensure] Validating URL: {url}")
                validated = self.validate(url, github_url)
                if validated:
                    print(f"[ensure] Found valid existing task: {validated.get('permalink_url')}")
                    existing = validated
                    break
            if not existing:
                print("[ensure] None of the existing Asana URLs in description are valid for this project")

        if not existing:
            print(f"[ensure] Searching for existing task with github_url: {github_url}")
            existing = self.search(
                github_url=github_url,
            )
            if existing:
                print(f"[ensure] Found existing task via search: {existing.get('permalink_url')}")
            else:
                print("[ensure] No existing task found via search")

        if existing:
            print(f"[ensure] Updating existing task: {existing.get('permalink_url')}")
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

        print(f"[ensure] Creating new task with title: {title}")
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
        print(f"[validate] validate() called with url='{url}', github_url='{github_url}'")
        gid = self.extract_gid_from_url(url)
        if not gid:
            print(f"[validate] Could not extract GID from URL: {url}")
            return None
        print(f"[validate] Extracted GID: {gid}")

        api_url = f"https://app.asana.com/api/1.0/tasks/{gid}"
        headers = {
            "Authorization": f"Bearer {self.asana_token}",
            "Content-Type": "application/json",
        }
        params = {
            "opt_fields": ",".join(
                [
                    "permalink_url",
                    "custom_fields",
                    "name",
                    "notes",
                    "modified_at",
                    "completed",
                    "assignee.email",
                    "followers.email",
                    "projects.gid",
                ]
            )
        }
        print(f"[validate] Making GET request to: {api_url}")
        response = requests.get(api_url, headers=headers, params=params, timeout=30)
        print(f"[validate] Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"[validate] API request failed: {response.text}")
            return None

        data = response.json()["data"]

        projects = [project["gid"] for project in data.get("projects", [])]
        if self.project_id not in projects:
            print(
                f"[validate] Task not in target project. Task projects: {projects}, target project: {self.project_id}"
            )
            return None

        custom_fields = data.get("custom_fields", [])
        task_github_url = None
        for field in custom_fields:
            if field.get("gid") == self.github_url_field_id:
                task_github_url = field.get("text_value")
                break

        print(f"[validate] Task GitHub URL: '{task_github_url}', expected: '{github_url}'")
        if task_github_url != github_url:
            print("[validate] GitHub URL mismatch")
            return None

        print("[validate] Task validation successful")
        return data

    def search(
        self,
        github_url: str,
    ) -> dict | None:
        print(f"[search] search() called with github_url='{github_url}'")
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
        print(f"[search] Making search request to: {api_url}")
        print(f"[search] Search params: {params}")
        response = requests.get(api_url, headers=headers, params=params, timeout=30)
        print(f"[search] Search response status: {response.status_code}")
        if response.status_code != 200:
            print(f"[search] Search API request failed: {response.text}")
            return None

        data = response.json()
        results = data["data"]
        print(f"[search] Search returned {len(results)} results")
        if results:
            print(f"[search] Returning first result: {results[0].get('permalink_url')}")
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
        print(f"[create] create() called with title='{title}', assignee='{assignee}', collaborators={collaborators}")
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

        print(f"[create] Creating task with payload: {payload}")
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        print(f"[create] Create response status: {response.status_code}")
        if response.status_code == 201:
            url = response.json()["data"]["permalink_url"]
            print(f"[create] Task created successfully: {url}")
            self.ensure_github_url_in_task(url, title, github_url)
            return url
        else:
            print(f"[create] Task creation failed: {response.text}")
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
        print(f"[update] update() called with gid='{gid}', title='{title}', completed={completed}")
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

        print(f"[update] Updating task with payload: {payload}")
        response = requests.put(api_url, json=payload, headers=headers, timeout=30)
        print(f"[update] Update response status: {response.status_code}")
        if response.status_code != 200:
            print(f"[update] Task update failed: {response.text}")
            raise Exception(f"Asana API Error (update): {response.status_code} - {response.text}")
        else:
            print("[update] Task updated successfully")

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
        print("[update_if_needed] update_if_needed() called")
        current_title = data.get("name") or ""
        current_notes = data.get("notes") or ""
        current_completed = data.get("completed") or False
        current_assignee = (data.get("assignee") or {}).get("email") or ""

        print("[update_if_needed] Current vs new values:")
        print(f"  title: '{current_title}' vs '{title}'")
        print(f"  notes: '{current_notes[:50]}...' vs '{description[:50]}...'")
        print(f"  completed: {current_completed} vs {task_completed}")
        print(f"  assignee: '{current_assignee}' vs '{assignee}'")

        if (
            current_title != title
            or current_notes != description
            or current_completed != task_completed
            or current_assignee != assignee
        ):
            print("[update_if_needed] Changes detected, updating task")
            self.update(
                data["gid"],
                title,
                description,
                task_completed,
                assignee,
                github_url,
                pr_author,
            )
        else:
            print("[update_if_needed] No changes needed")

    def ensure_github_url_in_task(
        self,
        url: str,
        title: str,
        github_url: str,
    ) -> dict | None:
        print(f"[ensure_github_url_in_task] ensure_github_url_in_task() called with url='{url}', title='{title}'")
        github_url_number = github_url.split("pull/")[-1]
        if not github_url_number.isdigit():
            print(f"[ensure_github_url_in_task] Could not extract PR number from GitHub URL: {github_url}")
            return None

        print(f"[ensure_github_url_in_task] PR number extracted: {github_url_number}")
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
        print(f"[ensure_github_url_in_task] Making attachment request with payload: {payload}")
        response = requests.post(ASANA_GITHUB_ATTACHMENT_ACTION_URL, json=payload, headers=headers, timeout=30)
        print(f"[ensure_github_url_in_task] Attachment response status: {response.status_code}")
        if response.status_code == 201:
            print("[ensure_github_url_in_task] GitHub attachment successful")
            return response.json()
        else:
            print(f"[ensure_github_url_in_task] GitHub attachment failed: {response.text}")
            return None

    def extract_github_url_from_comment(self, asana_comment_text, author_name):
        """
        Extract GitHub URL from Asana comment text, but only if it's a comment we created

        Args:
            asana_comment_text: The text content of an Asana comment
            author_name: The name of the comment author

        Returns:
            str: GitHub URL if found and comment is ours, None otherwise
        """
        if not asana_comment_text:
            return None

        # Check if this is a comment we created
        if author_name != "Github Sync":
            return None

        # Look for href="..." pattern in the comment
        pattern = r'href="([^"]+)"'
        match = re.search(pattern, asana_comment_text)

        if match:
            return match.group(1)

        return None

    def asana_comments_with_links(self):
        print("[asana_comments_with_links] asana_comments_with_links() called")
        comments = self.get_comments(self.task_gid)
        linked_comments = [comment for comment in comments if comment["github_url"] is not None]
        print(f"[asana_comments_with_links] Found {len(linked_comments)} comments with GitHub links of {len(comments)}")
        return linked_comments

    def get_comments(self, task_id: str):
        """
        Fetches comments for an Asana task
        Args:
            task_id (str): Asana task ID
        Returns:
            List[Dict]: List of comment dictionaries
        """
        print(f"[get_comments] get_comments() called with task_id='{task_id}'")
        from datetime import datetime

        import requests

        url = f"https://app.asana.com/api/1.0/tasks/{task_id}/stories"
        headers = {"Authorization": f"Bearer {self.asana_token}", "Content-Type": "application/json"}
        params = {
            "opt_fields": "text,html_text,created_by.name,created_by.email,created_at,type,resource_subtype,is_pinned"
        }
        print(f"[get_comments] Making GET request to: {url}")
        response = requests.get(url, headers=headers, params=params)
        print(f"[get_comments] Get comments response status: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        comments = [
            story
            for story in data["data"]
            if story.get("type") == "comment" or story.get("resource_subtype") == "comment_added"
        ]
        print(f"[get_comments] Found {len(comments)} comment stories out of {len(data['data'])} total stories")
        ret = []
        for comment in comments:
            author_name = comment.get("created_by", {}).get("name", "Unknown")
            github_url = self.extract_github_url_from_comment(comment.get("html_text", ""), author_name)
            if github_url:  # Only include comments that have GitHub URLs
                ret.append(
                    {
                        "id": comment.get("gid"),
                        "text": comment.get("text", ""),
                        "html_text": comment.get("html_text", ""),
                        "author": {
                            "name": author_name,
                            "email": comment.get("created_by", {}).get("email"),
                        },
                        "created_at": datetime.fromisoformat(comment.get("created_at", "").replace("Z", "+00:00")),
                        "is_pinned": comment.get("is_pinned", False),
                        "github_url": github_url,
                    }
                )
        print(f"[get_comments] Processed {len(ret)} comments with GitHub URLs")
        print(f"comments in asana: {ret}")

        return ret

    def synchronize_comments_in_asana(self, comments_from_github: list[dict]) -> None:
        """
        Synchronize review comments in Asana as multiple blocks.
        Args:
            comments_from_github (list[dict]): List of review comments from GitHub
        """
        print("[synchronize_comments_in_asana] Starting with:")
        asana_comments_with_links = self.asana_comments_with_links()
        print(f"  asana_comments_with_links: {len(asana_comments_with_links)} comments")
        print(f"  comments_from_github: {len(comments_from_github)} reviews")

        if not comments_from_github:
            print("[synchronize_comments_in_asana] No GitHub comments to process")
            return

        api_url = f"https://app.asana.com/api/1.0/tasks/{self.task_gid}"
        headers = {"Authorization": f"Bearer {self.asana_token}", "Content-Type": "application/json"}

        # Create a map of GitHub URLs to existing Asana comments
        existing_comments_by_github_url = {comment["github_url"]: comment for comment in asana_comments_with_links}
        print(f"[s] Existing comments by GitHub URL: {list(existing_comments_by_github_url.keys())}")

        # Process each GitHub review
        for github_review in comments_from_github:
            review_body = github_review.get("text", "")
            github_user = github_review["user"]
            review_state = github_review["action"]
            github_url = github_review.get("html_url")

            print(f"[s] Processing review from {github_user} ({review_state}) at {github_url}")

            # Format the review for Asana
            from pr_gh_to_asana import format_github_review_body_for_asana

            formatted_comment = format_github_review_body_for_asana(review_body, github_user, review_state, github_url)

            if github_url in existing_comments_by_github_url:
                print(f"[synchronize_comments_in_asana] Review at {github_url} has existing Asana comment")
                # Update existing comment if content differs
                existing_comment = existing_comments_by_github_url[github_url]
                if existing_comment["html_text"] != formatted_comment:
                    print(f"[s] Updating existing comment for review at {github_url}")
                    story_id = existing_comment["id"]
                    url = f"https://app.asana.com/api/1.0/stories/{story_id}"
                    payload = {"data": {"html_text": formatted_comment}}
                    try:
                        print(payload)
                        response = requests.put(url, headers=headers, json=payload)
                        if response.status_code == 200:
                            print(f"Updated Asana comment {story_id} for review at {github_url}")
                        else:
                            print(
                                f"Failed to update Asana comment {story_id}: {response.status_code} - {response.text}"
                            )
                    except requests.exceptions.RequestException as e:
                        print(f"Error updating Asana comment {story_id}: {e}")
                else:
                    print(f"[synchronize_comments_in_asana] Review at {github_url} comment is up to date")
            else:
                print(f"[s] Review at {github_url} has no existing Asana comment")
                # Create new comment
                url = f"{api_url}/stories"
                payload = {"data": {"html_text": formatted_comment, "type": "comment"}}
                print(payload)

                try:
                    response = requests.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    print(f"Added new Asana comment for review at {github_url}")
                except requests.exceptions.RequestException as e:
                    print(f"Error adding Asana comment for review at {github_url}: {e}")
                    print(payload)
