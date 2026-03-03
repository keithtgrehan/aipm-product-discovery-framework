from __future__ import annotations

from typing import Dict, List


def _format_list(items: List[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)


def baseline_prompt(context: str, question: str) -> str:
    return f"""
You are a clinical differential-diagnosis assistant.
Use only the context provided below.

Question:
{question}

Context:
{context}

Instructions:
1. Provide a concise differential diagnosis focused on fatigue.
2. Mention high-risk red flags/escalation cues when supported by context.
3. Cite evidence inline with passage tags such as [p1], [p2].
4. If a point is unknown from context, say unknown.
""".strip()


def structured_prompt(context: str, question: str, checklist: Dict[str, List[str]]) -> str:
    conditions = checklist.get("critical_conditions", [])
    qualifiers = checklist.get("critical_qualifiers", [])

    return f"""
You are a clinical differential-diagnosis assistant.
Use only the context provided below.

Question:
{question}

Context:
{context}

Critical conditions to address (must include each exact phrase):
{_format_list(conditions)}

Critical qualifiers to address (must include each exact phrase):
{_format_list(qualifiers)}

Instructions:
1. Provide a concise fatigue-focused answer with citations [p#].
2. Include a section titled "Checklist coverage".
3. In "Checklist coverage", echo every condition and qualifier EXACTLY and append either ": addressed" or ": unknown".
4. Do not omit any listed checklist item.
""".strip()
