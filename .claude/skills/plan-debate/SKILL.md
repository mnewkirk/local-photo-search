---
name: plan-debate
description: >
  Build an implementation plan by having two subagents on different models each draft a
  plan, critique each other over two rounds, and surface only the unresolved disagreements
  for the user to decide. Use before any multi-step implementation, migration, dashboard
  build, or when the user asks for a plan.
---

# Plan Debate

Build a plan by pitting two planners against each other, then bring the user only what
they still disagree on.

## Steps

### 1. **Gather the brief.**

Pull the task statement and constraints from the conversation into a short brief: goal,
constraints, files/systems involved, and what "done" looks like. Do not ask the user
clarifying questions at this stage unless the goal is genuinely ambiguous.

### 2. **Spawn two planners in parallel** with the Agent tool, in a single message so they run
concurrently:

- Planner A on model `'sonnet'`
- Planner B on model `'opus'`

Each receives the identical brief and must return a plan with:
numbered steps, files touched, risks, verification per step, and estimate. Each planner
writes its plan to the scratchpad directory as `plan_a.md` / `plan_b.md` (they may not
coordinate or read each other's files).

### 3. **Round 1 critique.**

Send each planner the other's plan (via SendMessage, continuing that planner's agent)
and ask for:
- the 3 most important points where the plans differ
- for each, whether they concede, hold, or propose a merge, with a one-sentence reason

### 4. **Round 2.**

Send each planner the other's critique. Ask them to update their position and mark each
point RESOLVED or UNRESOLVED.

### 5. **Synthesize.**

Merge everything RESOLVED into one plan. For each UNRESOLVED point, write a neutral
summary covering:
- what the disagreement is about
- Planner A's position and reason
- Planner B's position and reason
- the consequence of each choice

Do NOT paste the agents' discussion to the user.

### 6. **Present.**

Show the merged plan briefly, plus the disagreement list. Then use AskUserQuestion with
one question per unresolved point. Options are the two positions; put the recommended
one first and label it "(Recommended)"; the user may pick Other.

### 7. **Fold in decisions.**

Update the final plan with the user's choices. Write the plan to the project (PLAN.md
in the project root, or the location the project's conventions specify) with a `##
Decisions` section — one row per disagreement, columns are topic, Option A, Option B,
user's choice, date.

### 8. **Carry decisions forward.**

If this plan produces a PR, carry the `## Decisions` section into the PR body (see the
user's CLAUDE.md for PR intent and acceptance criteria section).

## Guardrails

- Cap at two critique rounds. Do not run a third.
- If the planners agree on everything in round 1 or round 2, say so and skip the
  AskUserQuestion step entirely.
- Never let a planner run `bq`/`git` commands or make any write.
- Planning is read-only.
- Keep the brief and both plans in the scratchpad directory, not the repo, until step 7
  writes the final plan.
