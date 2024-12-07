from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests
import json


class CustomLLM(LLM):
    max_token: int = 8192
    temperature: float = 0.5
    URL: str = "http://localhost:8000/v1/chat/completions"
    headers: dict = {"Content-Type": "application/json"}
    payload: dict = {"prompt": "", "history": []}
    logger: Any
    model_name: str=""

    @property
    def _llm_type(self) -> str:
        return self.model_name

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        history: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        payload = {
        "model": self.model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "do_sample": True,
        "temperature": 0,
        "top_p": 0,
        "n": 1,
        "max_tokens": 0,
        "stream": False
    }

        response = requests.post(self.URL, headers=self.headers, data=json.dumps(payload))

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"Error occurred: {response.text}")
            return response.text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_token": self.max_token,
            "URL": self.URL,
            "headers": self.headers,
            "payload": self.payload,
        }
