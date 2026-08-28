import os
from typing import Any
from typing import Literal
from dotenv import load_dotenv
from litellm import completion

load_dotenv(override=True)

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "default"] | None

DEFAULT_MODEL_NAME = os.getenv("PRICER_PREPROCESSOR_MODEL", "ollama/llama3.2")
DEFAULT_REASONING_EFFORT: ReasoningEffort = "low" if "gpt-oss" in DEFAULT_MODEL_NAME else None

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Preprocessor:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        reasoning_effort: ReasoningEffort = DEFAULT_REASONING_EFFORT,
        base_url: str | None = None,
    ):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.model_name = model_name
        self.reasoning_effort: ReasoningEffort = reasoning_effort
        self.base_url = base_url
        if "ollama" in model_name and not base_url:
            self.base_url = "http://localhost:11434"

    def messages_for(self, text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

    def preprocess(self, text: str) -> str:
        messages = self.messages_for(text)
        
        # Build kwargs dynamically to only pass reasoning_effort when needed
        kwargs = {}
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort

        response: Any = completion(
            messages=messages,
            model=self.model_name,
            api_base=self.base_url,
            **kwargs,
        )

        # Safely handle usage token counts
        if hasattr(response, "usage") and response.usage:
            self.total_input_tokens += getattr(response.usage, "prompt_tokens", 0) or 0
            self.total_output_tokens += getattr(response.usage, "completion_tokens", 0) or 0

        # Safely handle cost tracking
        hidden_params = getattr(response, "_hidden_params", {})
        if isinstance(hidden_params, dict):
            self.total_cost += float(hidden_params.get("response_cost", 0.0) or 0.0)

        # Fallback to empty string if content is None
        content = response.choices[0].message.content  # type: ignore
        return content or ""