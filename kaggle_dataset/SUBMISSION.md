# Submission package — Gemma 4 Developer Agent Paper Track

Everything needed to submit. Two writeups are allowed
(`allowMultipleSubmissions: true` confirmed via
`GetCompetitionHackathonSettings` for competitionId 163111).

## Links to paste into the Writeup "Project Links" field

`requireLinks: true` — both fields are mandatory.

| Field | URL |
|---|---|
| **Public Notebook** | https://www.kaggle.com/code/coachvitorcalvi/graphloc-seed-expansion |
| **Public Project Link** (arXiv-ready PDF) | https://www.kaggle.com/datasets/coachvitorcalvi/graphloc-300-labels |

Supporting dataset (cite in the paper, not a required field):
https://www.kaggle.com/datasets/coachvitorcalvi/graphloc-300-labels

## Writeup 1 — submit first (strongest overall)

- **Title:** Where Does the Graph Help? Seed Expansion for Function-Level
  Bug Localization on Repository Code Graphs
- **Subtitle:** What repository code graphs buy, and what they do not
- **Body:** `paper/paper1_resource.md` (2,995 words — under the 3,000 limit)
- **PDF:** `paper/paper1_resource.pdf`
- **Target:** Best Paper ($15,000) + Best New Resource ($10,000)

## Writeup 2 — submit second

- **Title:** The Adherence Gap: Retrieval Is Not the Bottleneck for
  Graph-Augmented Coding Agents
- **Subtitle:** Joining localization recall with end-to-end resolve rate
- **Body:** `paper/paper2_application.md` (2,124 words)
- **PDF:** `paper/paper2_application.pdf`
- **Target:** Best New Application ($10,000)

## Required sections (per the rules)

Title and subtitle · Abstract · Introduction · Description of the research
(including methods and experiments) · Related works and citations.
Both papers contain all five.

## Before clicking Submit

1. Join the Hackathon and accept the rules.
2. Paste the markdown body into the Writeup editor.
3. Fill both Project Links from the table above.
4. `requireCardImage: true` — add a cover image (e.g. the top-5 bar chart
   in the notebook, or a screenshot of the coverage-vs-budget table).
5. Submit **before 2026-11-12 23:59 UTC**. Drafts are not counted.

## What the numbers say (for the summary/abstract field)

- 80.7% of patches edit exactly one function in one file; the target is in
  the graph 100% of the time, so ranking — not coverage — is the bottleneck.
- The median gold function sits at BM25 rank 32, but 39.0% of tasks rank it
  outside the top 100 and 10.7% outside the top 5 000: **public-API /
  private-implementation divergence**.
- Expanding the lexical seed set lifts coverage@50 from 0.379 to 0.562 with
  **zero** tasks lost (exact McNemar p = 7.3e-12).
- **Self-correction:** a controlled ablation shows that gain is a bigger
  seed set, not better ordering. At budgets of 20–50 the graph-aware
  ordering is no better than plain BM25 (−0.014, p = 0.61). The usable
  configuration is budget-dependent: k = 5 at B = 10, k = 20–50 at B ≥ 100,
  and k = 1 is actively harmful (0.276).

## Why two papers

The three awards are judged separately, and the two papers make disjoint
claims so they do not compete with each other:

- Paper 1 is a **measurement + resource** claim (coverage, labels, ceilings).
- Paper 2 is an **agent-side diagnosis** claim (adherence, not recall).

## Verification

The notebook runs end-to-end on Kaggle (status COMPLETE, zero errors) and
regenerates every number quoted in both papers in under three minutes on
CPU, with no GPU, no LLM inference, and no paid API.
