from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import CompletionPrompt
from evals.record import record_sampling
import torch

from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig, AutoTokenizer


class LlamaCompletionResult(CompletionResult):
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


class LlamaCompletionFn(CompletionFn):
    def __init__(self, model_name: str, **kwargs) -> None:
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            return_dict=True,
            load_in_8bit=kwargs["quantization"],
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa" if kwargs.get("use_fast_kernels", False) else None,
        )
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        torch.manual_seed(kwargs.get("seed", 42))

    @torch.no_grad()
    def __call__(self, prompt, **kwargs) -> CompletionResult:
        prompt = self._tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        batch = self._tokenizer(prompt, padding='max_length', truncation=True, max_length=None, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}
        outputs = self._model.generate(
            **batch,
            **kwargs
        )
        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        record_sampling(prompt=prompt, sampled=response)
        return LlamaCompletionResult(response)
