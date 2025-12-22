import os
import yaml
import argparse
from uuid import uuid4
from pathlib import Path
from langfuse.callback import CallbackHandler
from typing import Any, Dict
from langfuse import Langfuse
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI


def load_config(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise Exception(f"Config not found at {path}")
    with path.open("r", encoding="utf-8") as handler:
        content = handler.read()
    content = os.path.expandvars(content)
    return yaml.safe_load(content)


def get_langfuse_config(local: bool = False) -> Dict[str, Any]:
    from src.settings import app_settings
    host = app_settings.callback.langfuse.host_local if local else app_settings.callback.langfuse.host
    return {
        "public_key": os.environ.get("LANGFUSE_PUBLIC_KEY", app_settings.callback.langfuse.public_key),
        "secret_key": os.environ.get("LANGFUSE_SECRET_KEY", app_settings.callback.langfuse.secret_key.get_secret_value()),
        "host": host,
        "debug": app_settings.callback.langfuse.debug,
    }


def get_callbacks(callback_config: dict) -> list:
    handlers = []
    langfuse_handler = CallbackHandler(
        secret_key=callback_config["langfuse"]["secret_key"],
        public_key=callback_config["langfuse"]["public_key"],
        host=callback_config["langfuse"]["host"],
        trace_name=callback_config["langfuse"]["trace_name"],
        tags=[*callback_config["langfuse"]["tags"], f"uuid_{uuid4()}"],
    )
    handlers.append(langfuse_handler)
    return handlers


def build_langfuse_client(langfuse_config: Dict[str, Any]) -> Langfuse:
    if not langfuse_config:
        raise Exception("Langfuse configuration required (callbacks.langfuse)")

    try:
        return Langfuse(
            public_key=langfuse_config["public_key"],
            secret_key=langfuse_config["secret_key"],
            host=langfuse_config["host"],
            debug=langfuse_config.get("debug", False),
        )
    except KeyError as exc:
        raise Exception("Langfuse public_key and secret_key are required") from exc

def serialize_args(namespace: argparse.Namespace) -> Dict[str, Any]:
    raw = vars(namespace).copy()
    for k, v in list(raw.items()):
        if isinstance(v, Path):
            raw[k] = str(v)
    return raw


def create_llm(llm_settings, use_json_response=False, **kwargs) -> ChatOpenAI | ChatMistralAI:
    """Создает экземпляр ChatOpenAI или ChatMistralAI на основе настроек."""

    common_kwargs = {
        "model": llm_settings.model,
        "temperature": llm_settings.temperature,
        "max_tokens": llm_settings.max_tokens,
        "api_key": llm_settings.api_key.get_secret_value(),
    }

    common_kwargs.update(kwargs)

    if use_json_response:
        common_kwargs["response_format"] = {"type": "json_object"}

    if "mistral" in llm_settings.model.lower():
        return ChatMistralAI(**common_kwargs)
    else:
        common_kwargs["base_url"] = llm_settings.base_url
        return ChatOpenAI(**common_kwargs)