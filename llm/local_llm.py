from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalLLM(LLM):
   
    model_path: str = "local"
    max_token: int = 2048
    temperature: float = 0.5
    logger: Any= None
    sampling_params: Any =None
    model: Any = None

    def __init__(self, model_path: str, max_token: int, temperature: float, logger: Any):
        super().__init__()
        self.model_path = model_path
        self.max_token = max_token
        self.temperature = temperature
        self.logger = logger
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        self.model.eval()


    @property
    def _llm_type(self) -> str:
        return self.model_path

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        history: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        outputs = self.model.generate([prompt], self.sampling_params,use_tqdm=False)

        response=outputs[0].outputs[0].text
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_token": self.max_token,
            "model Path": self.model,
        }
