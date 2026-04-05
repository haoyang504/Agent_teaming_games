"""
Agent
=====
Defines a single LLM-backed agent that participates in the Moon Survival
iterative ranking task.

Each agent:
  1. Proposes k candidate rankings given the previous iteration's results.
  2. Participates in round-robin discussion.
  3. (If last in the round) selects k final candidates for evaluation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from moon_survival_env import (
    ITEMS,
    format_items_list,
    format_results_summary,
    parse_ranking,
)
from knowledge_manager import format_knowledge_for_prompt

# ── LLM helper ───────────────────────────────────────────────────────────────

def _chat(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
) -> str:
    """Thin wrapper around OpenAI chat completion."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# ── Agent ────────────────────────────────────────────────────────────────────

_ID_TO_LABEL = {1: "A", 2: "B", 3: "C"}


class MoonSurvivalAgent:
    """A single agent in the Moon Survival team.

    Args:
        agent_id:             1-indexed integer identifier.
        knowledge:            List of (item_name, ground_truth_rank, explanation) tuples
                              representing the agent's specialised expertise.
        client:               Initialised OpenAI client.
        model:                Model identifier string (e.g. "gpt-4o-mini").
        num_agents:           Total number of agents in the team.
        team_knowledge_info:  Dict mapping agent labels to knowledge counts,
                              e.g. {"A": 0, "B": 2, "C": 4}. None = not provided.
        leader_id:            agent_id of the designated leader (e.g. 3). None = no leader.
        knowledge_warning:    If True, add a warning that some knowledge may be incorrect.
    """

    def __init__(
        self,
        agent_id: int,
        knowledge: List[Tuple[str, int, str]],
        client: OpenAI,
        model: str = "gpt-4o-mini",
        num_agents: int = 3,
        team_knowledge_info: Optional[Dict[str, int]] = None,
        leader_id: Optional[int] = None,
        knowledge_warning: bool = False,
    ) -> None:
        self.agent_id = agent_id
        self.knowledge = knowledge
        self.client = client
        self.model = model
        self.num_agents = num_agents
        self.team_knowledge_info = team_knowledge_info
        self.leader_id = leader_id
        self.knowledge_warning = knowledge_warning

    # ── System prompt ───────────────────────────────────────────────────────

    def _system_prompt(self) -> str:
        other_ids = [
            f"Agent {i}" for i in range(1, self.num_agents + 1)
            if i != self.agent_id
        ]
        others_str = " and ".join(other_ids)
        knowledge_block = format_knowledge_for_prompt(self.knowledge)
        if self.knowledge_warning:
            knowledge_block += (
                "\n\nNote: Some of the knowledge you have received may be incorrect."
            )

        # Build optional sections
        team_info_section = ""
        if self.team_knowledge_info is not None:
            lines = []
            for label in ("A", "B", "C"):
                count = self.team_knowledge_info.get(label, 0)
                if count == 0:
                    lines.append(f"- Agent {label} has no specialised knowledge.")
                else:
                    lines.append(f"- Agent {label} knows about {count} of the 15 items.")
            team_info_section = (
                "\n=== TEAM INFORMATION ===\n"
                "Your team members have the following levels of specialised knowledge:\n"
                + "\n".join(lines) + "\n"
            )

        role_section = ""
        if self.leader_id is not None:
            leader_label = _ID_TO_LABEL.get(self.leader_id, str(self.leader_id))
            role_section = (
                "\n=== TEAM ROLE ===\n"
                f"Agent {leader_label} has been designated as the team leader.\n"
            )

        return f"""You are Agent {self.agent_id} in a team of {self.num_agents} agents ({others_str} and you) solving the NASA Moon Survival ranking task.

=== TASK CONTEXT ===
You are a member of a space crew who crash-landed on the sunlit side of the Moon, 200 miles from the rendezvous point with the mother ship. You must rank 15 items from 1 (most critical) to 15 (least critical) for survival during the 200-mile trek.

=== ITEMS TO RANK ===
{format_items_list()}

=== YOUR SPECIALISED KNOWLEDGE ===
{knowledge_block}
{team_info_section}{role_section}
=== SCORING ===
Rankings are evaluated by Sum of Absolute Differences (SAD) compared to the NASA expert ranking. Lower is better; 0 is a perfect score.

=== GENERAL GUIDELINES ===
- Be concise but clearly reasoned.
- Draw on your specialised knowledge when arguing.
- Be open to persuasion by teammates who may know things you do not.
- When proposing or selecting rankings, always output them in the EXACT format specified.
"""

    # ── Step 1 : Proposal ───────────────────────────────────────────────────

    def propose_candidates(
        self,
        k: int,
        previous_results: Optional[str] = None,
        results_context: str = "the previous iteration",
    ) -> Tuple[List[Dict[str, int]], str]:
        """Generate k candidate rankings based on the previous iteration results.

        Args:
            k:                Number of candidates to propose.
            previous_results: Formatted string of previous candidates + scores,
                              or None for the very first iteration.
            results_context:  Human-readable description of what the results cover.

        Returns:
            Tuple of:
              - List of k parsed {item_name: rank} dictionaries.
              - Raw LLM response text.
        """
        if previous_results:
            context = (
                f"Here are the results from {results_context}:\n\n"
                f"{previous_results}\n\n"
                "Study these results carefully. You are NOT limited to building "
                "on any single previous candidate — you may combine insights "
                "across all of them."
            )
        else:
            context = (
                "This is the FIRST iteration. No prior results exist yet. "
                "Propose your best initial ranking(s) based on your knowledge "
                "and reasoning about lunar survival."
            )

        user_msg = f"""{context}

Please propose exactly {k} candidate ranking(s).

For each candidate, provide:
1. A brief reasoning paragraph (2-4 sentences) explaining your strategy.
2. The full ranking in this EXACT format:

CANDIDATE <n> RANKING:
1. [Item name]
2. [Item name]
...
15. [Item name]

Where <n> is 1, 2, ..., {k}.
Use EXACT item names from the list above.
"""
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user",   "content": user_msg},
        ]
        raw = _chat(self.client, self.model, messages)
        candidates = _parse_k_candidates(raw, k)
        return candidates, raw

    # ── Step 2 : Discussion ─────────────────────────────────────────────────

    def discuss(
        self,
        discussion_history: List[Dict[str, str]],
        iteration: int,
    ) -> str:
        """Participate in the round-robin discussion.

        Args:
            discussion_history: List of {"role": ..., "content": ...} messages
                                representing the conversation so far (proposals
                                + previous discussion turns).
            iteration:          Current iteration number (1-indexed).

        Returns:
            The agent's discussion response as a raw string.
        """
        user_msg = (
            f"You are participating in round-robin discussion (Iteration {iteration}). "
            f"Read all proposals and discussion above carefully. "
            f"As Agent {self.agent_id}, respond by:\n"
            "  1. Commenting on strengths or weaknesses of the proposals made "
            "     by other agents (cite your specialised knowledge if relevant).\n"
            "  2. Defending or revising your own proposal based on new arguments.\n"
            "  3. Highlighting any key swaps or adjustments you believe should "
            "     be made before the final selection.\n\n"
            "Keep your response focused (3-6 sentences per point). "
            "Do NOT output a full ranking here — only discussion points."
        )
        messages = (
            [{"role": "system", "content": self._system_prompt()}]
            + discussion_history
            + [{"role": "user", "content": user_msg}]
        )
        return _chat(self.client, self.model, messages)

    # ── Step 3 : Final Selection (last agent only) ──────────────────────────

    def select_final_candidates(
        self,
        k: int,
        discussion_history: List[Dict[str, str]],
        iteration: int,
    ) -> Tuple[List[Dict[str, int]], str]:
        """Select k final candidates after the round-robin discussion.

        Called only on the last agent in the round-robin order.

        Args:
            k:                  Number of final candidates to select.
            discussion_history: Full discussion history including all proposals
                                and discussion turns.
            iteration:          Current iteration number (1-indexed).

        Returns:
            Tuple of:
              - List of k parsed {item_name: rank} dictionaries.
              - Raw LLM response text.
        """
        items_reminder = "\n".join(f"{i+1}. {item}" for i, item in enumerate(ITEMS))
        user_msg = f"""The round-robin discussion for Iteration {iteration} is now complete.

As the final agent in this round, your role is to select the {k} best candidate ranking(s) to carry forward for evaluation.

Consider:
- Arguments raised by all agents.
- Your own specialised knowledge.
- Diversity of strategies among the candidates (to maximise learning).

CRITICAL FORMATTING RULES:
- You MUST include ALL 15 items in every ranking. Do not skip any.
- Use the EXACT item names listed below (copy-paste them):

{items_reminder}

Output exactly {k} final candidate(s) using this EXACT format (repeat for each n from 1 to {k}):

FINAL CANDIDATE <n> RANKING:
1. [exact item name]
2. [exact item name]
3. [exact item name]
4. [exact item name]
5. [exact item name]
6. [exact item name]
7. [exact item name]
8. [exact item name]
9. [exact item name]
10. [exact item name]
11. [exact item name]
12. [exact item name]
13. [exact item name]
14. [exact item name]
15. [exact item name]

Precede each ranking with a brief justification (2-3 sentences).
"""
        messages = (
            [{"role": "system", "content": self._system_prompt()}]
            + discussion_history
            + [{"role": "user", "content": user_msg}]
        )

        # ── Try up to 2 retries if parsing fails ────────────────────────────
        last_error: Optional[Exception] = None
        raw = ""
        for attempt in range(3):
            raw = _chat(self.client, self.model, messages)
            try:
                candidates = _parse_k_candidates(raw, k, label="FINAL CANDIDATE")
                return candidates, raw
            except ValueError as e:
                last_error = e
                print(
                    f"  [select_final_candidates] Parse failed (attempt {attempt+1}/3): {e}"
                )
                # Feed the error back so the model can self-correct
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"Your response could not be fully parsed: {e}\n\n"
                            "Please try again. Make sure to:\n"
                            "1. Include ALL 15 items in every ranking.\n"
                            "2. Start each ranking block with the EXACT header "
                            f"'FINAL CANDIDATE <n> RANKING:'\n"
                            "3. Use the exact item names listed above."
                        ),
                    },
                ]
        raise ValueError(
            f"Failed to parse final candidates after 3 attempts. Last error: {last_error}"
        )


# ── Parsing helper ────────────────────────────────────────────────────────────

def _parse_k_candidates(
    text: str,
    k: int,
    label: str = "CANDIDATE",
) -> List[Dict[str, int]]:
    """Extract k ranked lists from a multi-candidate LLM response.

    Searches for blocks headed by "<LABEL> <n> RANKING:" and parses each.

    Args:
        text:  Raw LLM response.
        k:     Expected number of candidates.
        label: Header prefix (e.g. "CANDIDATE" or "FINAL CANDIDATE").

    Returns:
        List of k {item_name: rank} dicts.

    Raises:
        ValueError: if fewer than k parseable candidates are found.
    """
    # Split on candidate headers (case-insensitive)
    pattern = re.compile(
        rf"{re.escape(label)}\s+\d+\s+RANKING\s*:", re.IGNORECASE
    )
    parts = pattern.split(text)
    # parts[0] is text before the first header; parts[1..] are the candidate blocks
    blocks = parts[1:]

    candidates: List[Dict[str, int]] = []
    for block in blocks:
        try:
            ranking = parse_ranking(block)
            candidates.append(ranking)
        except ValueError as e:
            print(f"  [parse_k_candidates] Warning: skipping block — {e}")

    if len(candidates) < k:
        raise ValueError(
            f"Expected {k} candidates with header '{label} <n> RANKING:' "
            f"but only parsed {len(candidates)}."
        )
    return candidates[:k]
