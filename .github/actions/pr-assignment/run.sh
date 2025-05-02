#!/bin/bash

# Script to run the PR assignment logic directly
# This allows us to call it from a loop in the workflow

# Arguments
PR_NUMBER="$1"
POSSIBLE_ASSIGNEES="$2"
POSSIBLE_REVIEWERS="$3"
FORCED_ASSIGNEES="$4"
FORCED_REVIEWERS="$5"
FORCED_LABELS="$6"
REPO="${GITHUB_REPOSITORY}"

# Get PR author
PR_AUTHOR=$(gh pr view $PR_NUMBER --json author --repo $REPO | jq -r '.author.login')
echo "Processing PR #$PR_NUMBER by @$PR_AUTHOR"

# Function to check if a string is empty or contains only whitespace
is_empty() {
  local str="$1"
  [[ -z "${str// /}" ]]
}

# Function to randomly select from a comma-separated list
select_random() {
  local list=$1
  local exclude=$2
  
  if is_empty "$list"; then
    echo ""
    return
  fi
  
  # Convert comma-separated list to array
  IFS=',' read -ra ITEMS <<< "$list"
  
  # Filter out excluded users
  if [ ! -z "$exclude" ]; then
    FILTERED_ITEMS=()
    for item in "${ITEMS[@]}"; do
      item=$(echo "$item" | xargs)  # Trim whitespace
      if [ "$item" != "$exclude" ]; then
        FILTERED_ITEMS+=("$item")
      fi
    done
    
    # If all users were filtered out, return empty
    if [ ${#FILTERED_ITEMS[@]} -eq 0 ]; then
      echo ""
      return
    fi
    
    # Use filtered array instead
    if [ ${#FILTERED_ITEMS[@]} -eq 1 ]; then
      echo "${FILTERED_ITEMS[0]}" | xargs
      return
    fi
    
    RANDOM_INDEX=$(( RANDOM % ${#FILTERED_ITEMS[@]} ))
    echo "${FILTERED_ITEMS[$RANDOM_INDEX]}" | xargs
    return
  fi
  
  # No exclusion, use all items
  if [ ${#ITEMS[@]} -eq 1 ]; then
    echo "${ITEMS[0]}" | xargs
    return
  fi
  
  RANDOM_INDEX=$(( RANDOM % ${#ITEMS[@]} ))
  echo "${ITEMS[$RANDOM_INDEX]}" | xargs
}

# Function to process a list into a format for GitHub CLI
format_for_gh() {
  local list=$1
  local formatted=""
  
  if is_empty "$list"; then
    echo ""
    return
  fi
  
  # Convert comma-separated list to array
  IFS=',' read -ra ITEMS <<< "$list"
  
  # Format each item for GitHub CLI
  for item in "${ITEMS[@]}"; do
    item=$(echo "$item" | xargs)  # Trim whitespace
    if is_empty "$item"; then
      continue
    fi
    
    if [ ! -z "$formatted" ]; then
      formatted+=" "
    fi
    formatted+="\"$item\""
  done
  
  echo "$formatted"
}

# Initialize arrays to track actions taken
ASSIGNED=()
REVIEWED=()
LABELED=()

# Process forced assignees (always assigned)
if ! is_empty "$FORCED_ASSIGNEES"; then
  FORMATTED_ASSIGNEES=$(format_for_gh "$FORCED_ASSIGNEES")
  
  if ! is_empty "$FORMATTED_ASSIGNEES"; then
    echo "Adding forced assignees: $FORCED_ASSIGNEES"
    
    # Convert comma-separated list to array for tracking
    IFS=',' read -ra FORCED_ASSIGNEE_ARRAY <<< "$FORCED_ASSIGNEES"
    for assignee in "${FORCED_ASSIGNEE_ARRAY[@]}"; do
      assignee=$(echo $assignee | xargs)
      if ! is_empty "$assignee"; then
        ASSIGNED+=("$assignee")
      fi
    done
    
    # Assign the PR
    eval "gh pr edit $PR_NUMBER --add-assignee $FORMATTED_ASSIGNEES --repo $REPO" || echo "Warning: Failed to add forced assignees"
  fi
else
  echo "No forced assignees specified"
fi

# Process random assignee (if enabled and not empty)
if ! is_empty "$POSSIBLE_ASSIGNEES"; then
  SELECTED_ASSIGNEE=$(select_random "$POSSIBLE_ASSIGNEES" "$PR_AUTHOR")
  
  if ! is_empty "$SELECTED_ASSIGNEE"; then
    echo "Selected random assignee: $SELECTED_ASSIGNEE"
    
    # Only assign if this person isn't already assigned
    if [[ ! " ${ASSIGNED[*]} " =~ " ${SELECTED_ASSIGNEE} " ]]; then
      ASSIGNED+=("$SELECTED_ASSIGNEE")
      
      # Assign the PR
      gh pr edit $PR_NUMBER --add-assignee "$SELECTED_ASSIGNEE" --repo $REPO || echo "Warning: Failed to add random assignee"
      echo "Successfully assigned PR #$PR_NUMBER to $SELECTED_ASSIGNEE"
    else
      echo "Assignee $SELECTED_ASSIGNEE already assigned as forced assignee, skipping"
    fi
  else
    echo "No valid random assignee could be selected"
  fi
else
  echo "No possible assignees specified for random selection"
fi

# Process forced reviewers (always requested)
if ! is_empty "$FORCED_REVIEWERS"; then
  FORMATTED_REVIEWERS=$(format_for_gh "$FORCED_REVIEWERS")
  
  if ! is_empty "$FORMATTED_REVIEWERS"; then
    echo "Adding forced reviewers: $FORCED_REVIEWERS"
    
    # Convert comma-separated list to array for tracking
    IFS=',' read -ra FORCED_REVIEWER_ARRAY <<< "$FORCED_REVIEWERS"
    for reviewer in "${FORCED_REVIEWER_ARRAY[@]}"; do
      reviewer=$(echo $reviewer | xargs)
      if ! is_empty "$reviewer"; then
        REVIEWED+=("$reviewer")
      fi
    done
    
    # Request reviews
    eval "gh pr edit $PR_NUMBER --add-reviewer $FORMATTED_REVIEWERS --repo $REPO" || echo "Warning: Failed to add forced reviewers"
  fi
else
  echo "No forced reviewers specified"
fi

# Process random reviewer (if enabled and not empty)
if ! is_empty "$POSSIBLE_REVIEWERS"; then
  # Skip author as reviewer by default
  SELECTED_REVIEWER=$(select_random "$POSSIBLE_REVIEWERS" "$PR_AUTHOR")
  
  if ! is_empty "$SELECTED_REVIEWER"; then
    echo "Selected random reviewer: $SELECTED_REVIEWER"
    
    # Only request review if someone was selected and isn't already reviewing
    if [[ ! " ${REVIEWED[*]} " =~ " ${SELECTED_REVIEWER} " ]]; then
      REVIEWED+=("$SELECTED_REVIEWER")
      
      # Request review
      gh pr edit $PR_NUMBER --add-reviewer "$SELECTED_REVIEWER" --repo $REPO || echo "Warning: Failed to add random reviewer"
      echo "Successfully requested review from $SELECTED_REVIEWER for PR #$PR_NUMBER"
    else
      echo "Reviewer $SELECTED_REVIEWER already assigned as forced reviewer, skipping"
    fi
  else
    echo "No valid random reviewer could be selected"
  fi
else
  echo "No possible reviewers specified for random selection"
fi

# Process forced labels (replacing existing labels)
if ! is_empty "$FORCED_LABELS"; then
  echo "Setting forced labels: $FORCED_LABELS"
  
  # First, get current labels to remove them
  CURRENT_LABELS=$(gh pr view $PR_NUMBER --json labels --repo $REPO | jq -r '.labels[].name' 2>/dev/null || echo "")
  
  # Remove all existing labels
  if ! is_empty "$CURRENT_LABELS"; then
    for label in $CURRENT_LABELS; do
      if ! is_empty "$label"; then
        gh pr edit $PR_NUMBER --remove-label "$label" --repo $REPO || echo "Warning: Failed to remove label: $label"
        echo "Removed label: $label"
      fi
    done
  fi
  
  # Format and add forced labels
  FORMATTED_LABELS=$(format_for_gh "$FORCED_LABELS")
  
  if ! is_empty "$FORMATTED_LABELS"; then
    eval "gh pr edit $PR_NUMBER --add-label $FORMATTED_LABELS --repo $REPO" || echo "Warning: Failed to add forced labels"
    
    # Track labels for comment
    IFS=',' read -ra FORCED_LABEL_ARRAY <<< "$FORCED_LABELS"
    for label in "${FORCED_LABEL_ARRAY[@]}"; do
      label=$(echo $label | xargs)
      if ! is_empty "$label"; then
        LABELED+=("$label")
      fi
    done
  fi
else
  echo "No forced labels specified"
fi

# Add a summary comment
if [ ${#ASSIGNED[@]} -gt 0 ] || [ ${#REVIEWED[@]} -gt 0 ] || [ ${#LABELED[@]} -gt 0 ]; then
  COMMENT="PR automatically processed:\n"
  
  if [ ${#ASSIGNED[@]} -gt 0 ]; then
    COMMENT+="- Assigned to: "
    for assignee in "${ASSIGNED[@]}"; do
      COMMENT+="@$assignee "
    done
    COMMENT+="\n"
  fi
  
  if [ ${#REVIEWED[@]} -gt 0 ]; then
    COMMENT+="- Review requested from: "
    for reviewer in "${REVIEWED[@]}"; do
      COMMENT+="@$reviewer "
    done
    COMMENT+="\n"
  fi
  
  if [ ${#LABELED[@]} -gt 0 ]; then
    COMMENT+="- Labels set: "
    for label in "${LABELED[@]}"; do
      COMMENT+="\`$label\` "
    done
  fi
  
  gh pr comment $PR_NUMBER --body "$COMMENT" --repo $REPO || echo "Warning: Failed to add summary comment"
else
  echo "No actions were taken on this PR - no assignees, reviewers, or labels were set."
fi