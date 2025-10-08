from abc import ABC, abstractmethod
import os
from openai import OpenAI


class LLM(ABC):
    """Abstract base class for Large Language Models."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM given a prompt.
        Must include any required stop-token logic at the caller level.
        """
        raise NotImplementedError


class OpenAIModel(LLM):
    """
    Example LLM implementation using OpenAI's Responses API.

    TODO(student): Implement this class to call your chosen backend (e.g., OpenAI GPT-5 mini)
    and return the model's text output. You should ensure the model produces the response
    format required by ResponseParser and include the stop token in the output string.
    """

    def __init__(self, stop_token: str, model_name: str = "gpt-5-mini"):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.stop_token = stop_token
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        # Call OpenAI API to generate response
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        
        # Extract the text content from the response
        text = response.choices[0].message.content
        
        # Ensure the stop token is present at the end
        if not text.endswith(self.stop_token):
            text += "\n" + self.stop_token
        
        return text