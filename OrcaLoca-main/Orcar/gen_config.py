import os

import config
from llama_index.core.base.llms.types import LLMMetadata, MessageRole
from google.oauth2 import service_account
from llama_index.core.llms.llm import LLM
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.llms.vertex import Vertex
from transformers import AutoTokenizer

from .utils import VertexAnthropicWithCredentials


def get_qwen_tokenizer_name(model: str) -> str:
    explicit = os.environ.get("QWEN_TOKENIZER_MODEL")
    if explicit:
        return explicit
    if model.lower().startswith("qwen3") or model.lower().startswith("qwen3.5"):
        return "Qwen/Qwen3-8B"
    if model.lower().startswith("qwen2.5"):
        return "Qwen/Qwen2.5-7B-Instruct"
    return "Qwen/Qwen2.5-7B-Instruct"


class OpenAICompatible(OpenAI):
    @property
    def _tokenizer(self):
        if not hasattr(self, "_orcar_tokenizer"):
            self._orcar_tokenizer = AutoTokenizer.from_pretrained(
                get_qwen_tokenizer_name(self.model),
                trust_remote_code=True,
            )
        return self._orcar_tokenizer

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=131072,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model,
            system_role=MessageRole.SYSTEM,
        )


class Config:
    def __init__(self, file_path=None, provider=None):
        self.file_path = file_path
        if self.file_path and os.path.isfile(self.file_path):
            self.file_config = config.Config(self.file_path)
        else:
            self.file_config = dict()
        self.fallback_config = dict()
        self.fallback_config["OPENAI_API_BASE_URL"] = ""
        self.provider = provider

    def __getitem__(self, index):
        # Values in key.cfg has priority over env variables
        if self.file_config.get(index):
            return self.file_config.get(index)
        if index in os.environ:
            return os.environ[index]
        if index in self.fallback_config:
            return self.fallback_config[index]
        raise KeyError(
            f"Cannot find {index} in either cfg file '{self.file_path}' or env variables"
        )


def get_llm(**kwargs) -> LLM:
    # key.cfg is in the parent directory of this file
    orcar_config: Config = kwargs.get("orcar_config", None)
    model = kwargs.get("model", None)
    if model.lower().startswith("claude"):
        # first check if the provider has been set
        if orcar_config.provider == "vertexanthropic":
            print(f"Using AnthropicVertex model: {model}")
            service_account_path = os.path.expanduser(
                orcar_config["VERTEX_SERVICE_ACCOUNT_PATH"]
            )
            if not os.path.exists(service_account_path):
                raise FileNotFoundError(
                    f"Google Cloud Service Account file not found: {service_account_path}"
                )
            try:
                credentials = service_account.Credentials.from_service_account_file(
                    service_account_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                kwargs["credentials"] = credentials
                kwargs["project_id"] = credentials.project_id
                kwargs["region"] = orcar_config["VERTEX_REGION"]
                LLM_func = VertexAnthropicWithCredentials
            except Exception as e:
                raise Exception(f"gen_config: Failed to get vertexanthropic LLM") from e
        else:
            kwargs["api_key"] = orcar_config["ANTHROPIC_API_KEY"]
            LLM_func = Anthropic
    elif model.lower().startswith("gemini"):
        # Load Google Cloud credentials
        service_account_path = orcar_config["VERTEX_SERVICE_ACCOUNT_PATH"]

        if not os.path.exists(service_account_path):
            raise FileNotFoundError(
                f"Google Cloud Service Account file not found: {service_account_path}"
            )

        credentials = service_account.Credentials.from_service_account_file(
            service_account_path
        )

        kwargs["project"] = credentials.project_id
        kwargs["credentials"] = credentials
        LLM_func = Vertex
    elif model.lower().startswith("gpt") or model.lower().startswith("qwen"):
        kwargs["api_key"] = orcar_config["OPENAI_API_KEY"]
        api_base = orcar_config["OPENAI_API_BASE_URL"]
        if api_base:
            kwargs["api_base"] = api_base
        if model.lower().startswith("qwen"):
            additional_kwargs = dict(kwargs.get("additional_kwargs") or {})
            extra_body = dict(additional_kwargs.get("extra_body") or {})
            extra_body.setdefault("enable_thinking", False)
            additional_kwargs["extra_body"] = extra_body
            kwargs["additional_kwargs"] = additional_kwargs
        LLM_func = OpenAI if model.lower().startswith("gpt") else OpenAICompatible
    else:
        raise ValueError(f"Unsupported model: {model}")

    # delete orcar_config from kwargs
    if "orcar_config" in kwargs:
        del kwargs["orcar_config"]

    try:
        llm: LLM = LLM_func(**kwargs)
        _ = llm.complete("Say 'Hi'")
        return llm
    except Exception as e:
        raise Exception(f"Failed to initialize LLM: {e}")
