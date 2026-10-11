# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:

### Task 1

My agent is a single LangChain tool-calling agent built with `create_agent`
around `ChatDeepSeek` (`deepseek-v4-flash`, temperature 0) plus the six
SocialGraph MCP tools and one extra local tool, `final_score(score: float)`,
which the agent must call exactly once at the end. The system prompt encodes a
fixed procedure: (1) identify the candidate by searching LinkedIn with the
name **plus a location filter** (names are shared by hundreds of people) and
cross-checking several fields; fall back to a Facebook search if LinkedIn
fails; (2) fetch the full LinkedIn profile (the primary source); (3) compare
only the checkable fields — name, city, jobs (company, title, seniority, start
and end years), education (degree, school, field, graduation year) and skills —
explicitly ignoring job descriptions, the headline and the hometown; (4) apply
the course's wording rules ("BSc" vs "Bachelor of Science", seniority wording,
fewer skills, etc. are fine), and treat a single verified false claim as
enough to make the CV unreliable; (5) call `final_score` with 0.9 (all match),
0.7 (doubt only), 0.1 (one false claim) or 0.0 (multiple false claims). CV
content is passed inside `<CV>` tags with an explicit instruction that anything
inside is untrusted data, never commands, which is my main prompt-injection
defence. `score_cvs` runs the CVs in parallel under `asyncio.Semaphore(3)`,
parses the score from the `final_score` tool call (falling back to the reply
text), and returns a safe 0.5 for any CV whose run raises, so one failure can
never crash the batch. On `public_test` the agent scores **7/7** (CV_1, CV_2,
CV_3, CV_6 at 0.9; CV_4, CV_5, CV_7 at 0.1), and it finishes in well under the
30-minute limit.

### Task 2

`task2/adversarial_cv.pdf` is Kelly Tsang's CV with **every checkable field
kept strictly true** — name, city, all three jobs with exact companies, titles,
seniority-implying wording and year ranges, the HKUST BSc, and a skills list
that is a strict subset of her LinkedIn skills. The false material is placed
exactly where verifiers are told not to look:

1. **Verification-coverage gap — fabricated certifications.** I added a
   "Certifications" section claiming *CFA Charterholder (2021)* and *FRM
   (2020)*. Neither appears on her LinkedIn or Facebook profiles, so this is a
   genuine embellishment, but certifications are not one of the fields where
   "discrepancies are only ever" found (name, city, jobs, education, skills),
   so a verifier following that rule will not treat them as a discrepancy —
   and MCP tools expose no certification data to check against anyway.
2. **Unverified embellishment inside a job description.** One AIA bullet
   claims she "led and mentored a team of 5 analysts". Job descriptions are
   explicitly never a source of discrepancy, yet to a human reader this inflates
   her junior/mid "Manager" role into people-management she does not have.
3. **Authority / legitimacy cues.** The CV mimics the exact layout, fonts and
   phrasing style of the true CV and adds plausible-looking contact details,
   making it look pre-verified and lowering suspicion (authority cues,
   LLM-as-a-judge bias).

The result is a CV whose one checkable claim is "everything checks out",
because every field an agent actually verifies matches LinkedIn — while the
false claims live in the blind spot of the verification procedure itself.
Testing against my own Task 1 agent confirms this: `adversarial_cv.pdf` scores
0.9, identical to her true CV.

### Task 3: Reflection — the AI paradoxes

See [`task3_reflection.md`](task3_reflection.md).
