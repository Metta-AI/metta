#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "asana>=5.2.2",
#   "requests>=2.31.0",
#   "vcrpy>=6.0.0",
#   "pyyaml>=6.0.0"
# ]
# ///

import os
import re
from datetime import datetime

import vcr
from asana_task import AsanaTask
from github_asana_mapping import GithubAsanaMapping
from pull_request import PullRequest

# VCR configuration for tracking REST traffic
# logging.basicConfig(level=logging.INFO)
# vcr_log = logging.getLogger("vcr")
# vcr_log.setLevel(logging.DEBUG)

record_mode = os.getenv("VCR_RECORD_MODE", "new_episodes")
print(f"VCR record mode: {record_mode}")

my_vcr = vcr.VCR(
    record_mode=record_mode,
    cassette_library_dir=".",
    filter_headers=["Authorization", "User-Agent"],
    match_on=["uri", "method"],
    filter_query_parameters=["access_token"],
)


def log_http_interactions(cassette_name):
    """Log HTTP interactions from the VCR cassette"""
    try:
        import os

        import yaml

        # Check if file exists
        if not os.path.exists(cassette_name):
            print(f"VCR cassette file not found: {cassette_name}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Files in current directory: {os.listdir('.')}")
            return

        with open(cassette_name, "r") as f:
            cassette_data = yaml.safe_load(f)

        if cassette_data and "interactions" in cassette_data:
            interactions = cassette_data["interactions"]
            print(f"Recorded {len(interactions)} HTTP interactions in {cassette_name}")

            for i, interaction in enumerate(interactions):
                request = interaction.get("request", {})
                response = interaction.get("response", {})
                method = request.get("method", "UNKNOWN")
                uri = request.get("uri", "UNKNOWN")
                status = response.get("status", {}).get("code", "UNKNOWN")
                print(f"  {i + 1}. {method} {uri} -> {status}")
        else:
            print(f"No HTTP interactions recorded in {cassette_name}")
    except Exception as e:
        print(f"Error logging HTTP interactions: {e}")
        import traceback

        traceback.print_exc()


def format_github_review_body_for_asana(review_body, github_user, review_state, github_url):
    """
    Format GitHub review body comment for Asana

    Args:
        review_body: The review's body text (markdown string)
        github_user: GitHub username of the reviewer
        review_state: Review state (APPROVED, CHANGES_REQUESTED, COMMENTED)
        github_url: GitHub URL for the review comment
    """
    # Determine emoji based on review state
    if review_state == "APPROVED":
        emoji = "\u2705"
    elif review_state == "CHANGES_REQUESTED":
        emoji = "\U0001f4dd"
    else:  # COMMENTED or other states
        emoji = ""

    # Format header with user and state as link
    state = review_state.replace("_", " ").title()
    header = f'{state} ({emoji}) by {github_user}: <a href="{github_url}">View in GitHub</a>\n\n'

    # Convert basic markdown in body
    formatted_body = convert_basic_markdown(review_body) if review_body else "(No comment)"

    return "<body>" + header + formatted_body + "</body>"


def convert_basic_markdown(text):
    """Convert basic markdown to Asana HTML"""

    # Bold: **text** -> <strong>text</strong>
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)

    # Italic: *text* -> <em>text</em>
    text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", text)

    # Inline code: `code` -> <code>code</code>
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

    return text


def getenv_or_bust(key: str) -> str:
    """
    Get environment variable value, throwing exception if not found

    Args:
        key (str): Environment variable name

    Returns:
        str: Environment variable value (guaranteed non-None)

    Raises:
        Exception: If environment variable is not set or is empty
    """
    value = os.getenv(key)
    print(f"Environment variable '{key}' = {repr(value)}")

    if value is None:
        raise Exception(f"Environment variable '{key}' is not set")

    # Optional: Also check for empty strings
    if value == "":
        raise Exception(f"Environment variable '{key}' is empty")

    return value


"""
Asana task assignment strategy:
    - The main Asana task is always assigned to the PR author
    - When reviews are requested or re-requested, we create/reopen subtasks assigned to each reviewer
    - This keeps the PR owner responsible for the overall task while distributing review work via subtasks
"""
if __name__ == "__main__":
    import traceback

    run_id = os.getenv("GITHUB_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    cassette_override = os.getenv("VCR_CASSETTE_PATH")
    if cassette_override:
        cassette_name = cassette_override
        print(f"Using cassette override: {cassette_name}")
    else:
        cassette_name = f"http_interactions_{run_id}.yaml"

    print(f"Starting VCR recording with cassette: {cassette_name}")
    print(f"Current working directory: {os.getcwd()}")

    try:
        with my_vcr.use_cassette(cassette_name):
            # Inputs from the Action
            project_id = getenv_or_bust("INPUT_PROJECT_ID")
            workspace_id = getenv_or_bust("INPUT_WORKSPACE_ID")
            asana_token = getenv_or_bust("INPUT_ASANA_TOKEN")
            github_url = getenv_or_bust("INPUT_GITHUB_URL")
            github_url_field_id = getenv_or_bust("INPUT_GITHUB_URL_FIELD_ID")
            gh_login_field_id = getenv_or_bust("INPUT_GH_LOGIN_FIELD_ID")
            asana_email_field_id = getenv_or_bust("INPUT_ASANA_EMAIL_FIELD_ID")
            roster_project_id = getenv_or_bust("INPUT_ROSTER_PROJECT_ID")
            pr_author_field_id = getenv_or_bust("INPUT_PR_AUTHOR_FIELD_ID")
            asana_attachment_secret = getenv_or_bust("INPUT_ASANA_ATTACHMENT_SECRET")
            pr_number = getenv_or_bust("INPUT_PR_NUMBER")
            github_repo = getenv_or_bust("INPUT_GITHUB_REPO")
            github_token = getenv_or_bust("INPUT_GITHUB_TOKEN")

            pr = PullRequest(github_repo, pr_number, github_token)

            mapping = GithubAsanaMapping(
                pr.github_logins,
                roster_project_id,
                gh_login_field_id,
                asana_email_field_id,
                asana_token,
            )

            # Always assign the main Asana task to the PR author
            asana_assignee = mapping.github_login_to_asana_email.get(pr.author)
            asana_collaborators = [
                mapping.github_login_to_asana_email[login]
                for login in pr.github_logins
                if login in mapping.github_login_to_asana_email
            ]

            # Get the author's Asana ID for the custom field
            pr_author_asana = mapping.github_login_to_asana_email.get(pr.author)

            # print out debug info
            pr.print_debug_info()

            # todo print out variables above

            asana_task = AsanaTask(
                asana_token=asana_token,
                github_url_field_id=github_url_field_id,
                pr_author_field_id=pr_author_field_id,
                asana_email_field_id=asana_email_field_id,
                project_id=project_id,
                workspace_id=workspace_id,
                attachment_secret=asana_attachment_secret,
            )

            # Ensure task exists and output URL
            task_url = asana_task.ensure(
                title=pr.title,
                description=pr.body,
                task_completed=pr.task_completed,
                assignee=asana_assignee,
                collaborators=asana_collaborators,
                github_url=github_url,
                pr_author=pr_author_asana,
                urls=pr.asana_urls,
            )

            review_comments = [e for e in pr.events if e["type"] == "review"]
            asana_task.synchronize_comments_in_asana(review_comments)

            # Synchronize review subtasks based on PR state
            if pr.is_open:
                # PR is open - sync subtasks for requested reviewers
                asana_task.synchronize_review_subtasks(
                    pr.requested_reviewers,
                    mapping.github_login_to_asana_email,
                    pr_number,
                    pr.author,
                    pr.title,
                    github_url,
                )

                # Mark review subtasks as complete for reviewers who have submitted reviews
                # but are NOT currently in requested_reviewers (i.e., not re-requested)
                for review in review_comments:
                    reviewer_login = review.get("user")
                    if reviewer_login and reviewer_login not in pr.requested_reviewers:
                        asana_task.complete_review_subtask_for_reviewer(
                            reviewer_login,
                            mapping.github_login_to_asana_email,
                        )
            else:
                # PR is closed or merged - close all review subtasks
                asana_task.close_all_review_subtasks()

            if "GITHUB_OUTPUT" in os.environ:
                with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                    f.write(f"task_url={task_url}\n")
            else:
                print(f"Skipping write to GITHUB_OUTPUT. Task URL: {task_url}")

    except Exception as e:
        print(
            f"Exception while running github action: {e}. "
            "Failing silently so as not to block the PR. "
            "Rerun github action to retry."
        )
        traceback.print_exc()
    finally:
        # Log all HTTP interactions
        log_http_interactions(cassette_name)
