from evals.api import CompletionFn, CompletionResult
from evals.record import record_sampling
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class LlamaCompletionResult(CompletionResult):
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


class LlamaCompletionFn(CompletionFn):
    def __init__(self, llm: str, llm_kwargs: Optional[dict] = None, **kwargs) -> None:
        self._model = AutoModelForCausalLM.from_pretrained(
            llm,
            return_dict=True,
            load_in_8bit=llm_kwargs["load_in_8bit"],
            load_in_4bit=llm_kwargs["load_in_4bit"],
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa" if llm_kwargs.get("use_fast_kernels", False) else None,
            torch_dtype=torch.bfloat16
        )
        self._model.eval()

        torch.manual_seed(llm_kwargs.get("seed", 42))
        self._llm = llm
        self._gen_kwargs = llm_kwargs['gen_kwargs']

    @torch.no_grad()
    def __call__(self, prompt, **kwargs) -> CompletionResult:
        tokenizer = AutoTokenizer.from_pretrained(self._llm)
        prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        batch = tokenizer(prompt, padding='max_length', truncation=True, max_length=None, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}
        outputs = self._model.generate(
            **batch,
            **self._gen_kwargs,
        )
        # Take only response:
        outputs = outputs[0][batch['input_ids'][0].size(0):]
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        record_sampling(prompt=prompt, sampled=response)
        return LlamaCompletionResult(response)
