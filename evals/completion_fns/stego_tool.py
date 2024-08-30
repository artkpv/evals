import logging
from typing import Any, Optional, Union

import openai
from openai import OpenAI
import json

from evals.api import CompletionFn, CompletionResult
from evals.base import CompletionFnSpec
from evals.prompt.base import (
    ChatCompletionPrompt,
    CompletionPrompt,
    OpenAICreateChatPrompt,
    OpenAICreatePrompt,
    Prompt,
)
from evals.record import record_sampling
from evals.utils.api_utils import create_retrying
import subprocess


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "stegcloak_hide",
            "description": "Hide a secret message in a cover text using steganography",
            "parameters": {
                "type": "object",
                "properties": {
                    "secret": {
                        "type": "string",
                        "description": "secret to hide",
                    },
                    "cover": {
                        "type": "string",
                        "description": "cover text",
                    },
                },
                "required": ["secret", "cover"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stegcloak_reveal",
            "description": "Reveal the secret hidden using steganography from a message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message that contains the secret",
                    },
                },
                "required": ["message"],
            },
        },
    }
]


def stegcloak_hide(secret: str, cover: str) -> str:
    """ Call from nodejs app: `npx stegcloak hide -n {secret} {cover}` """
    command = ["npx", "stegcloak", "hide", "-n", secret, cover]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout


def stegcloak_reveal(message: str) -> str:
    """ Call from nodejs app: `npx stegcloak reveal {message}` """
    command = ["npx", "stegcloak", "reveal", message]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout


def openai_completion_create_retrying(client: OpenAI, *args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    result = create_retrying(
        client.completions.create, retry_exceptions=OPENAI_TIMEOUT_EXCEPTIONS, *args, **kwargs
    )
    if "error" in result:
        logging.warning(result)
        raise openai.APIError(result["error"])
    return result


def openai_chat_completion_create_retrying(client: OpenAI, *args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    result = create_retrying(
        client.chat.completions.create, retry_exceptions=OPENAI_TIMEOUT_EXCEPTIONS, *args, **kwargs
    )
    if "error" in result:
        logging.warning(result)
        raise openai.APIError(result["error"])
    return result

class OpenAIBaseCompletionResult(CompletionResult):
    def __init__(self, raw_data: Any, prompt: Any):
        self.raw_data = raw_data
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        raise NotImplementedError


class OpenAIChatCompletionResult(OpenAIBaseCompletionResult):
    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data:
            for choice in self.raw_data.choices:
                if choice.message.content is not None:
                    completions.append(choice.message.content)
        return completions


class OpenAICompletionResult(OpenAIBaseCompletionResult):
    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data:
            for choice in self.raw_data.choices:
                completions.append(choice.text)
        return completions


class StegoToolCompletionFn(CompletionFn):
    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        n_ctx: Optional[int] = None,
        extra_options: Optional[dict] = {},
        **kwargs,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.n_ctx = n_ctx
        self.extra_options = extra_options

    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> OpenAICompletionResult:
        if not isinstance(prompt, Prompt):
            assert (
                isinstance(prompt, str)
                or (isinstance(prompt, list) and all(isinstance(token, int) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(token, str) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt))
            ), f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected str or list[int] or list[str] or list[dict[str, str]]"

            prompt = CompletionPrompt(
                raw_prompt=prompt,
            )

        openai_create_prompt: OpenAICreatePrompt = prompt.to_formatted_prompt()
        result = openai_completion_create_retrying(
            OpenAI(),
            model=self.model,
            prompt=openai_create_prompt,
            #tools=tools,
            **{**kwargs, **self.extra_options},
        )
        result = OpenAICompletionResult(raw_data=result, prompt=openai_create_prompt)
        record_sampling(
            prompt=result.prompt,
            sampled=result.get_completions(),
            model=result.raw_data.model,
            usage=result.raw_data.usage,
        )

        return result


class StegoToolChatCompletionFn(CompletionFnSpec):
    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        n_ctx: Optional[int] = None,
        extra_options: Optional[dict] = {},
        **kwargs,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.n_ctx = n_ctx
        self.extra_options = extra_options

    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> OpenAIChatCompletionResult:
        if not isinstance(prompt, Prompt):
            assert (
                isinstance(prompt, str)
                or (isinstance(prompt, list) and all(isinstance(token, int) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(token, str) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt))
            ), f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected str or list[int] or list[str] or list[dict[str, str]]"

            prompt = ChatCompletionPrompt(
                raw_prompt=prompt,
            )

        messages: OpenAICreateChatPrompt = prompt.to_formatted_prompt()

        client = OpenAI()
        chat_response = openai_chat_completion_create_retrying(
            client,
            model=self.model,
            messages=messages,
            tools=tools,
            **{**kwargs, **self.extra_options},
        )

        response_message = chat_response.choices[0].message
        tool_calls = response_message.tool_calls
        if tool_calls:
            available_functions = {
                "stegcloak_hide": stegcloak_hide,
                "stegcloak_reveal": stegcloak_reveal,
            }  # only one function in this example, but you can have multiple
            messages.append(chat_response)  # extend conversation with assistant's reply
            # Step 4: send the info for each function call and function response to the model
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                if function_name not in available_functions:
                    raise ValueError(f"Function {function_name} not found in available functions.")

                if function_name == "stegcloak_hide":
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = function_to_call(
                        secret=function_args.get("secret"),
                        cover=function_args.get("cover"),
                    )
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    )
                elif function_name == "stegcloak_reveal":
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = function_to_call(
                        message=function_args.get("message"),
                    )
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    )
            chat_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )  # get a new response from the model where it can see the function response

        result = OpenAIChatCompletionResult(raw_data=chat_response, prompt=messages)
        if not result.get_completions():
            print(raw_result)
            logging.error("API returned an empty list of completions.")
            raise ValueError("API returned an empty list of completions.")
        
        record_sampling(
            prompt=result.prompt,
            sampled=result.get_completions(),
            model=result.raw_data.model,
            usage=result.raw_data.usage,
        )
        return result
