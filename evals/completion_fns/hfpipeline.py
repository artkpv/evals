from evals.api import CompletionFn, CompletionResult
from transformers import pipeline, AutoTokenizer
from evals.prompt.base import CompletionPrompt
from evals.record import record_sampling
import torch


class HFPipelineCompletionResult(CompletionResult):
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


class HFPipelineCompletionFn(CompletionFn):
    def __init__(self, llm: str, **kwargs) -> None:
        self._pipeline = pipeline(
            "text-generation",
            model=llm,
            device_map="auto",
            model_kwargs={
                #"load_in_4bit": True,
                "torch_dtype": torch.bfloat16
            },
        )

    def __call__(self, prompt, **kwargs) -> CompletionResult:
        #prompt = CompletionPrompt(prompt).to_formatted_prompt()

        prompt = self._pipeline.tokenizer.apply_chat_template(
                prompt, 
                tokenize=False, 
                add_generation_prompt=True
        )
        terminators = [
            self._pipeline.tokenizer.eos_token_id,
            self._pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        response = self._pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            return_full_text=False,
            top_p=0.9,
        )
        response = response[0]["generated_text"]
        record_sampling(prompt=prompt, sampled=response)
        return HFPipelineCompletionResult(response)
