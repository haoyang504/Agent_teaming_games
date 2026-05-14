"""Per-call raw-prompt + raw-output logging for Phase 9 reviewability."""

import os
from typing import Optional


class Logger:
    """Writes one prompt file and one output file per LLM call.

    Output layout (next to the existing trajectory JSON):

        results/<experiment_id>_raw/
          prompts/
            iter{N}_{phase}_Agent{K}.txt
          outputs/
            iter{N}_{phase}_Agent{K}.txt

    The `phase` field uses Weikai's convention:
      - "propose" for the initial-proposal call
      - "round1", "round2", "round3", ...  for discussion rounds
      - "final_select" for the leader's final-selection call
    """

    def __init__(self, experiment_id: str, output_dir: str) -> None:
        self.experiment_id = experiment_id
        self.root = os.path.join(output_dir, f"{experiment_id}_raw")
        self.prompts_dir = os.path.join(self.root, "prompts")
        self.outputs_dir = os.path.join(self.root, "outputs")
        os.makedirs(self.prompts_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

    def save_prompt_and_output(
        self,
        phase: str,
        iteration: int,
        agent_name: str,
        formatted_messages: str,
        response_text: str,
        round_num: Optional[int] = None,
    ) -> None:
        """Write two parallel files for a single LLM call."""
        # round_num overrides phase only for discussion rounds.
        effective_phase = f"round{round_num}" if round_num is not None else phase
        filename = f"iter{iteration}_{effective_phase}_{agent_name}.txt"

        with open(os.path.join(self.prompts_dir, filename), "w", encoding="utf-8") as f:
            f.write(formatted_messages)
        with open(os.path.join(self.outputs_dir, filename), "w", encoding="utf-8") as f:
            f.write(response_text)
