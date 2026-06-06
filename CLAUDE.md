# Git workflow

Commit directly on `main`. Do not create branches for normal work (features, fixes, refactors, experiments that are expected to land).

Only create a branch when there is a high risk the changes will be rejected and nuked entirely (e.g. a speculative rewrite the user may dislike wholesale). In that case, name it clearly (e.g. `risky/<topic>`) and say so, so it can be deleted without ceremony if abandoned.

Never leave work stranded on a side branch: if a branch's changes are accepted, merge it into `main` and delete the branch in the same step.
