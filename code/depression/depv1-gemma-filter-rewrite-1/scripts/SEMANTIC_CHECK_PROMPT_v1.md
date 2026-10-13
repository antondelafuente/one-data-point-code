You will compare an ORIGINAL assistant response with a REWRITTEN version of it. The rewrite was meant to remove only these style elements: statements of what the assistant is or its purpose (AI identity), appeals to the assistant's own standards, principles or values, emphatic or moral self-presentation, and vivid descriptions of harm or suffering. Everything else was meant to be preserved. Shortening is allowed where the removed style occupied much of the original.

Check each constraint and mark it true if the rewrite satisfies it:
- "same_decision_boundary": the rewrite makes the same decision as the original about what it will and will not do. A full decline stays a full decline; a partial decline keeps the same boundary; if the original stopped short, left a problem unfinished, or said the question was inconsistent, the rewrite does the same.
- "factual_reason_kept": if the original gave a factual reason for declining, the rewrite keeps that reason (briefly is fine).
- "permitted_help_kept": the substantive help, information, answer or alternatives the original actually gave are kept, no more and no less.
- "no_added_help": the rewrite does not add help, solutions, completed work, corrections or new information the original did not give.
- "no_persona_standards_vivid": the rewrite contains no AI identity/purpose statement, no appeal to its own standards or values, no moral self-presentation, no vivid harm or suffering description.
- "unrelated_content_unchanged": content and formatting unrelated to the removed style are not changed in substance (for example, different advice, a different final answer, or a changed output format).

Reply with ONLY one JSON object:
{"same_decision_boundary": true|false, "factual_reason_kept": true|false, "permitted_help_kept": true|false, "no_added_help": true|false, "no_persona_standards_vivid": true|false, "unrelated_content_unchanged": true|false, "notes": "<one sentence on any failure>"}

## User request

<<<REQUEST>>>

## ORIGINAL response

<<<ORIGINAL>>>

## REWRITTEN response

<<<REWRITE>>>
