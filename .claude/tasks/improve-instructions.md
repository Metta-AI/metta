# Task: Improve CLAUDE.md Instructions

## Problem Statement
Current CLAUDE.md instructions for Plan & Review section are missing key elements present in the example, including:
- Detailed plan template structure
- Clear phase separation (before/during/after)
- Specific documentation requirements during implementation
- ExitPlanMode tool usage guidance

## MVP Approach
Create comprehensive instructions that match the example's structure while maintaining clarity and conciseness.

## Implementation Plan
1. Add complete Plan & Review Process section with all subsections
2. Include plan template with all required fields
3. Add During Implementation guidance with update format
4. Add After Implementation completion requirements
5. Ensure clear, actionable instructions throughout

## Success Criteria
- [ ] Instructions include all three phases (before/during/after)
- [ ] Plan template matches example structure
- [ ] Clear guidance on ExitPlanMode tool usage
- [ ] Specific format for implementation updates
- [ ] Maintains clarity while being comprehensive

## Proposed Improved Instructions

```markdown
## Plan & Review Process

**IMPORTANT: Always start in plan mode before implementing any changes.**

### Before Starting Work

1. **Enter plan mode first** - Use the ExitPlanMode tool only after presenting a complete plan
2. **Create a task plan** - Write your plan to `.claude/tasks/TASK_NAME.md` with:
   - Clear problem statement
   - MVP approach (always think minimal viable solution first)
   - Step-by-step implementation plan
   - Success criteria
3. **Research if needed** - If the task requires external knowledge or complex searches, use the Task tool with appropriate agents
4. **Request review** - After writing the plan, explicitly ask: "Please review this plan before I proceed with implementation"
5. **Wait for approval** - Only exit plan mode and begin implementation after receiving approval

#### Plan Template (.claude/tasks/TASK_NAME.md)

```markdown
# Task: [TASK_NAME]

## Problem Statement
[Clear description of what needs to be done]

## MVP Approach
[Minimal solution that solves the core problem]

## Implementation Plan
1. [Step 1]
2. [Step 2]
3. ...

## Success Criteria
- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] ...

## Implementation Updates
[This section will be updated during implementation]
```

### During Implementation

**Maintain the plan as living documentation throughout implementation:**

1. **Update as you work** - When you discover new information or need to adjust the approach, update the plan file
2. **Document completed steps** - After completing each major step, append a brief description:
   ```markdown
   ### Step 1 Complete: [Date/Time]
   - Changed: [what was changed]
   - Files affected: [list files]
   - Key decisions: [any important choices made]
   ```
3. **Track deviations** - If you need to deviate from the plan, document why and update the approach
4. **Keep it concise** - Focus on what changed and why, not how (the code shows how)

### After Implementation

1. **Final update** - Update the task file with:
   - Summary of what was accomplished
   - Any known limitations or future work
   - Lessons learned (if applicable)
2. **Verify success criteria** - Check off completed criteria in the plan
3. **Clean up** - Ensure all code is properly tested and documented
```