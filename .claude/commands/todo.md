---
name: todo
description: Manage TODO list in .todo.md files with consistent formatting and daily memory search
---

# TODO List Manager

## Current Action: $ARGUMENTS

I'll help you manage your TODO list stored in TODO.MD. This command will:
- Search your memories and other information for relevant context
- Update TODO.MD with consistent formatting
- Support nested items with indentation
- Keep track of your daily tasks

## Task Steps:

1. First, I'll search your memories and relevant information for context about your todos
2. Either create a new `<topic>.todo.md` or update an existing one based on the topic
3. Update the file based on your request:
   - Add new todos
   - Mark items as complete
   - Organize with sub-bullets
   - Add context from memories if relevant

## Available Actions:
- Add a new todo: `/todo add [task description]`
- Mark complete: `/todo complete [task description]`
- View all todos: `/todo list`
- Add sub-task: `/todo add-sub [parent task] > [sub-task]`
- Search related memories: `/todo context [topic]`
