import re

import asana
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

        # Initialize Asana SDK client
        config = asana.Configuration()
        config.access_token = asana_token
        self.api_client = asana.ApiClient(config)

        # Initialize API instances
        self.tasks_api = asana.TasksApi(self.api_client)
        self.stories_api = asana.StoriesApi(self.api_client)

        # Store configuration
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
                collaborators,
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

        opts = {
            "opt_fields": (
                "permalink_url,custom_fields,name,notes,modified_at,completed,"
                "assignee.email,followers.email,projects.gid"
            )
        }

        try:
            print(f"[validate] Fetching task {gid}")
            data = self.tasks_api.get_task(gid, opts)

            projects = [project["gid"] for project in data.get("projects", [])]
            if self.project_id not in projects:
                print(
                    f"[validate] Task not in target project. "
                    f"Task projects: {projects}, target project: {self.project_id}"
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
        except Exception as e:
            print(f"[validate] Failed to fetch task: {e}")
            return None

    def search(
        self,
        github_url: str,
    ) -> dict | None:
        print(f"[search] search() called with github_url='{github_url}'")
        opts = {
            "opt_fields": "permalink_url,custom_fields,name,notes,modified_at,completed,assignee.email,followers.email",
            "projects.any": self.project_id,
            f"custom_fields.{self.github_url_field_id}.value": github_url,
        }

        try:
            print(f"[search] Searching for tasks in workspace {self.workspace_id}")
            print(f"[search] Search params: {opts}")
            # Use search_tasks_for_workspace and convert generator to list
            results = list(self.tasks_api.search_tasks_for_workspace(self.workspace_id, opts, item_limit=1))
            print(f"[search] Search returned {len(results)} results")
            if results:
                print(f"[search] Returning first result: {results[0].get('permalink_url')}")
                return results[0]
            return None
        except Exception as e:
            print(f"[search] Search failed: {e}")
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
        body = {
            "data": {
                "name": title,
                "notes": description,
                "workspace": self.workspace_id,
                "projects": [self.project_id],
                "followers": collaborators,
                "custom_fields": {
                    self.github_url_field_id: github_url,
                    self.pr_author_field_id: pr_author,
                },
            }
        }
        if assignee:
            body["data"]["assignee"] = assignee
        if completed:
            body["data"]["completed"] = True

        try:
            print(f"[create] Creating task with body: {body}")
            task = self.tasks_api.create_task(body, {})
            url = task["permalink_url"]
            print(f"[create] Task created successfully: {url}")
            self.ensure_github_url_in_task(url, title, github_url)
            return url
        except Exception as e:
            print(f"[create] Task creation failed: {e}")
            raise Exception(f"Asana API Error (create): {e}") from e

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
        body = {
            "data": {
                "name": title,
                "notes": description,
                "completed": completed,
                "custom_fields": {
                    self.github_url_field_id: github_url,
                    self.pr_author_field_id: pr_author,
                },
            }
        }
        if assignee:
            body["data"]["assignee"] = assignee

        try:
            print(f"[update] Updating task with body: {body}")
            self.tasks_api.update_task(body, gid, {})
            print("[update] Task updated successfully")
        except Exception as e:
            print(f"[update] Task update failed: {e}")
            raise Exception(f"Asana API Error (update): {e}") from e

    def add_followers_if_needed(self, gid: str, current_followers: list[str], new_followers: list[str]) -> None:
        """Add new followers to the task if they're not already following"""
        print(f"[add_followers_if_needed] Current followers: {current_followers}")
        print(f"[add_followers_if_needed] Desired followers: {new_followers}")

        # Find followers to add (those in new_followers but not in current_followers)
        followers_to_add = [f for f in new_followers if f not in current_followers]

        if not followers_to_add:
            print("[add_followers_if_needed] No new followers to add")
            return

        print(f"[add_followers_if_needed] Adding followers: {followers_to_add}")
        body = {"data": {"followers": followers_to_add}}

        try:
            self.tasks_api.add_followers_for_task(body, gid, {})
            print(f"[add_followers_if_needed] Successfully added {len(followers_to_add)} followers")
        except Exception as e:
            print(f"[add_followers_if_needed] Failed to add followers: {e}")

    def update_if_needed(
        self,
        data: dict,
        title: str,
        description: str,
        task_completed: bool,
        assignee: str | None,
        collaborators: list[str],
        github_url: str,
        pr_author: str | None,
    ) -> None:
        print("[update_if_needed] update_if_needed() called")
        current_title = data.get("name") or ""
        current_notes = data.get("notes") or ""
        current_completed = data.get("completed") or False
        current_assignee = (data.get("assignee") or {}).get("email") or ""
        current_permalink_url = data.get("permalink_url") or ""
        current_followers = [f.get("email") for f in data.get("followers", []) if f.get("email")]

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
            # Ensure GitHub URL is attached to the task
            self.ensure_github_url_in_task(
                current_permalink_url,
                title,
                github_url,
            )
        else:
            print("[update_if_needed] No changes needed")

        # Always sync followers (add new ones if needed)
        self.add_followers_if_needed(data["gid"], current_followers, collaborators)

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

        opts = {
            "opt_fields": "text,html_text,created_by.name,created_by.email,created_at,type,resource_subtype,is_pinned"
        }

        try:
            print(f"[get_comments] Fetching stories for task {task_id}")
            stories = list(self.stories_api.get_stories_for_task(task_id, opts))
            comments = [
                story
                for story in stories
                if story.get("type") == "comment" or story.get("resource_subtype") == "comment_added"
            ]
            print(f"[get_comments] Found {len(comments)} comment stories out of {len(stories)} total stories")
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
        except Exception as e:
            print(f"[get_comments] Failed to fetch comments: {e}")
            return []

    def create_subtask(
        self,
        name: str,
        assignee: str | None,
        notes: str = "",
    ) -> str:
        """Create a subtask under the current task"""
        print(f"[create_subtask] Creating subtask '{name}' assigned to '{assignee}'")
        body = {
            "data": {
                "name": name,
                "notes": notes,
            }
        }
        if assignee:
            body["data"]["assignee"] = assignee

        try:
            subtask = self.tasks_api.create_subtask_for_task(body, self.task_gid, {})
            gid = subtask["gid"]
            print(f"[create_subtask] Subtask created successfully with GID: {gid}")
            return gid
        except Exception as e:
            print(f"[create_subtask] Failed to create subtask: {e}")
            raise Exception(f"Asana API Error (create_subtask): {e}") from e

    def get_subtasks(self) -> list[dict]:
        """Get all subtasks of the current task"""
        print(f"[get_subtasks] Fetching subtasks for task {self.task_gid}")
        opts = {"opt_fields": "gid,name,assignee.email,completed,notes"}

        try:
            subtasks = list(self.tasks_api.get_subtasks_for_task(self.task_gid, opts))
            print(f"[get_subtasks] Found {len(subtasks)} subtasks")
            return subtasks
        except Exception as e:
            print(f"[get_subtasks] Failed to fetch subtasks: {e}")
            return []

    def synchronize_review_subtasks(
        self,
        requested_reviewers: list[str],
        github_asana_mapping: dict[str, str],
        pr_number: str,
        pr_author: str,
        pr_title: str,
        pr_url: str,
    ) -> None:
        """Synchronize review subtasks based on requested reviewers"""
        print(f"[synchronize_review_subtasks] Synchronizing subtasks for {len(requested_reviewers)} reviewers")

        # Get existing subtasks
        existing_subtasks = self.get_subtasks()

        # Build mapping of assignee email to subtask for review subtasks
        review_subtask_prefix = "Review "
        existing_review_subtasks = {}
        for subtask in existing_subtasks:
            if subtask["name"].startswith(review_subtask_prefix):
                assignee_email = subtask.get("assignee", {}).get("email") if subtask.get("assignee") else None
                if assignee_email:
                    # Keep the most recent subtask for each assignee (in case there are duplicates)
                    if assignee_email not in existing_review_subtasks:
                        existing_review_subtasks[assignee_email] = subtask

        print(f"[synchronize_review_subtasks] Found {len(existing_review_subtasks)} existing review subtasks")

        # Create subtasks for new requested reviewers
        for reviewer_login in requested_reviewers:
            reviewer_email = github_asana_mapping.get(reviewer_login)
            if not reviewer_email:
                print(f"[synchronize_review_subtasks] No Asana email found for reviewer {reviewer_login}, skipping")
                continue

            if reviewer_email in existing_review_subtasks:
                subtask = existing_review_subtasks[reviewer_email]
                if subtask["completed"]:
                    print(
                        f"[synchronize_review_subtasks] Creating new review subtask for "
                        f"re-requested review from {reviewer_login}"
                    )
                    # Create a new subtask for re-requested review (prefer new task over reopening)
                    subtask_name = f"Review #{pr_number} by {pr_author}: {pr_title}"
                    subtask_notes = pr_url
                    self.create_subtask(
                        name=subtask_name,
                        assignee=reviewer_email,
                        notes=subtask_notes,
                    )
                else:
                    print(f"[synchronize_review_subtasks] Review subtask already exists for {reviewer_login}")
            else:
                print(f"[synchronize_review_subtasks] Creating new review subtask for {reviewer_login}")
                subtask_name = f"Review #{pr_number} by {pr_author}: {pr_title}"
                subtask_notes = pr_url
                self.create_subtask(
                    name=subtask_name,
                    assignee=reviewer_email,
                    notes=subtask_notes,
                )

    def complete_review_subtask_for_reviewer(
        self,
        reviewer_login: str,
        github_asana_mapping: dict[str, str],
    ) -> None:
        """Mark review subtask as complete when reviewer submits a review"""
        print(f"[complete_review_subtask_for_reviewer] Marking review complete for {reviewer_login}")

        reviewer_email = github_asana_mapping.get(reviewer_login)
        if not reviewer_email:
            print(
                f"[complete_review_subtask_for_reviewer] No Asana email found for reviewer {reviewer_login}, skipping"
            )
            return

        # Get existing subtasks
        existing_subtasks = self.get_subtasks()
        review_subtask_prefix = "Review "

        # Find the subtask assigned to this reviewer
        for subtask in existing_subtasks:
            if subtask["name"].startswith(review_subtask_prefix) and not subtask["completed"]:
                assignee_email = subtask.get("assignee", {}).get("email") if subtask.get("assignee") else None
                if assignee_email == reviewer_email:
                    try:
                        body = {"data": {"completed": True}}
                        self.tasks_api.update_task(body, subtask["gid"], {})
                        print(f"[complete_review_subtask_for_reviewer] Completed subtask: {subtask['name']}")
                        return
                    except Exception as e:
                        print(
                            f"[complete_review_subtask_for_reviewer] Failed to complete subtask {subtask['gid']}: {e}"
                        )
                        return

        print(f"[complete_review_subtask_for_reviewer] No open review subtask found for {reviewer_login}")

    def close_all_review_subtasks(self) -> None:
        """Close all review subtasks when PR is merged or closed"""
        print("[close_all_review_subtasks] Closing all review subtasks")

        existing_subtasks = self.get_subtasks()
        review_subtask_prefix = "Review "

        closed_count = 0
        for subtask in existing_subtasks:
            if subtask["name"].startswith(review_subtask_prefix) and not subtask["completed"]:
                try:
                    body = {"data": {"completed": True}}
                    self.tasks_api.update_task(body, subtask["gid"], {})
                    print(f"[close_all_review_subtasks] Closed subtask: {subtask['name']}")
                    closed_count += 1
                except Exception as e:
                    print(f"[close_all_review_subtasks] Failed to close subtask {subtask['gid']}: {e}")

        print(f"[close_all_review_subtasks] Closed {closed_count} review subtasks")

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
                    body = {"data": {"html_text": formatted_comment}}
                    try:
                        print(body)
                        self.stories_api.update_story(body, story_id, {})
                        print(f"Updated Asana comment {story_id} for review at {github_url}")
                    except Exception as e:
                        print(f"Error updating Asana comment {story_id}: {e}")
                else:
                    print(f"[synchronize_comments_in_asana] Review at {github_url} comment is up to date")
            else:
                print(f"[s] Review at {github_url} has no existing Asana comment")
                # Create new comment
                body = {"data": {"html_text": formatted_comment}}
                print(body)

                try:
                    self.stories_api.create_story_for_task(body, self.task_gid, {})
                    print(f"Added new Asana comment for review at {github_url}")
                except Exception as e:
                    print(f"Error adding Asana comment for review at {github_url}: {e}")
                    print(body)
