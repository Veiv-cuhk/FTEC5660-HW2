# Task 3 Reflection: The AI Paradoxes

*(FTEC5660 Homework 2 — individual reflection)*

## The verifier's paradox: trust built on a narrow window

Task 1 taught me something uncomfortable about AI-assisted verification. My agent achieved 7/7 on the public test, yet its confidence is built on a narrow window: it checks exactly the fields it was told can contain discrepancies, from exactly one data source. It never asks *"what is this CV not telling me?"* — it only asks *"is what this CV says consistent with LinkedIn?"* The paradox is that **the better the agent follows its specification, the more exploitable it becomes**, because the specification itself draws the map an attacker can walk around. I exploited precisely this in Task 2: the fake CFA certification sits outside the "checkable fields", so a perfectly compliant verifier scores my fabricated CV 0.9 — the same as the true one. An AI verifier is only as strong as its assumptions about where lies live.

## The automation paradox: AI raises both the floor and the ceiling of fraud

The same models that make KYC verification scalable make CV forgery scalable. In this homework, building the attacker took me about an hour; building the defender took much longer, and I still fooled it. This is the automation paradox: **every capability AI grants to defenders, it grants symmetrically to attackers.** Companies deploying AI agents for compliance must understand that their adversaries run the same tools. Verification becomes an arms race, not a solved problem — and the marginal cost of producing a convincing lie approaches zero.

## The trust paradox: more AI checking can mean less human checking

If an AI agent scores a CV 0.9, a human recruiter will almost certainly not look further. AI here does not eliminate human bias — it launders it: the human outsources judgment to the machine, the machine's blind spots become the institution's blind spots, and everyone feels safe because "the AI checked it". The paradox is that **adding AI oversight can reduce real oversight** while increasing the *appearance* of rigour. In a KYC context, this is not an abstract risk — it is a compliance failure with legal consequences.

## What I take away

1. **Never let the attacker see the checker's spec.** Publishing "discrepancies are only ever in these fields" (as this homework wisely did) is fine for teaching, but in production, the coverage policy is itself sensitive information.
2. **Defence needs redundancy.** My agent's prompt-injection defence worked, but its field-coverage assumption did not. Layered defences — multiple independent verifiers with different prompts, an explicit "what's missing?" check, uncertainty flags for uncheckable claims like certifications — would have caught my attack.
3. **Uncertainty is information.** My agent treats "unverifiable" as neutral (0.6). But in Task 2, the *absence* of certification data was the whole attack. A verifier should be more suspicious of claims it *cannot* check, not less — the absence of evidence about a CFA charter is exactly where the lie hides.

The deepest irony: I built both sides of this contest with the same model, the same tools, in one afternoon. The question this leaves me with is not "can AI verify?" but "can anything verify, when the fabricator and the auditor are the same machine?"
