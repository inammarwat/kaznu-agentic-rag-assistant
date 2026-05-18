import os
from typing import List, Optional

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv(override=False)


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    output = []

    for item in items:
        if item and item not in seen:
            output.append(item)
            seen.add(item)

    return output


def build_kaznu_llm(
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
):
    """
    Build OpenAI-compatible client for KazNU Farabi inference endpoint.
    """
    base_url = os.getenv("AGENT_BASE_URL")
    api_key = os.getenv("AGENT_API_KEY")

    if not base_url:
        raise ValueError("AGENT_BASE_URL is missing from environment.")

    if not api_key:
        raise ValueError("AGENT_API_KEY is missing from environment.")

    http_client = httpx.Client(
        verify=False,
        timeout=int(os.getenv("AGENT_TIMEOUT_SECONDS", "60")),
    )

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=2,
        timeout=int(os.getenv("AGENT_TIMEOUT_SECONDS", "60")),
        http_client=http_client,
    )


def build_groq_llm(
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
):
    """
    Build GROQ-backed LangChain chat model.
    Used as backup when KazNU inference is unavailable.
    """
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise ImportError(
            "langchain-groq is not installed. Run: uv add langchain-groq groq"
        ) from exc

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from environment.")

    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=2,
        timeout=int(os.getenv("GROQ_TIMEOUT_SECONDS", "60")),
    )


def test_llm_connection(llm, provider: str, model_name: str) -> None:
    """
    Small health check so failed models are detected early.
    """
    response = llm.invoke("Reply with only OK.")
    content = getattr(response, "content", "")

    print(f"LLM initialized: provider={provider} | model={model_name} | Response: {content}")


def initialize_kaznu_llm(
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
):
    primary_model = os.getenv("AGENT_MODEL", "Qwen/Qwen3.5-9B")
    fallback_model = os.getenv("AGENT_FALLBACK_MODEL", "Qwen/Qwen3.5-9B")

    model_names = unique_preserve_order([primary_model, fallback_model])
    max_tokens = max_tokens or int(os.getenv("AGENT_MAX_TOKENS", "2048"))

    errors = []

    for model_name in model_names:
        try:
            llm = build_kaznu_llm(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            test_llm_connection(llm, provider="kaznu", model_name=model_name)
            return llm

        except Exception as exc:
            error_message = f"{model_name}: {exc}"
            errors.append(error_message)
            print(f"Failed KazNU model: {error_message}")

    raise RuntimeError(
        "All KazNU LLM models failed. Errors:\n" + "\n".join(errors)
    )


def initialize_groq_llm(
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
):
    primary_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    fallback_model = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")

    model_names = unique_preserve_order([primary_model, fallback_model])
    max_tokens = max_tokens or int(os.getenv("GROQ_MAX_TOKENS", "2048"))

    errors = []

    for model_name in model_names:
        try:
            llm = build_groq_llm(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            test_llm_connection(llm, provider="groq", model_name=model_name)
            return llm

        except Exception as exc:
            error_message = f"{model_name}: {exc}"
            errors.append(error_message)
            print(f"Failed GROQ model: {error_message}")

    raise RuntimeError(
        "All GROQ LLM models failed. Errors:\n" + "\n".join(errors)
    )


def initialize_llm(
    temperature: float = 0.0,
    provider: Optional[str] = None,
    max_tokens: Optional[int] = None,
):
    """
    Initialize LLM for the project.

    provider options:
    - kaznu: use KazNU Farabi OpenAI-compatible endpoint
    - groq: use GROQ backup
    - auto: try KazNU first, then GROQ

    Existing project code can continue calling:
        initialize_llm(temperature=0.0)
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "kaznu")).strip().lower()

    if provider == "kaznu":
        return initialize_kaznu_llm(
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "groq":
        return initialize_groq_llm(
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "auto":
        errors = []

        try:
            return initialize_kaznu_llm(
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            errors.append(f"KazNU failed: {exc}")
            print(errors[-1])

        try:
            return initialize_groq_llm(
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            errors.append(f"GROQ failed: {exc}")
            print(errors[-1])

        raise RuntimeError(
            "All providers failed. Errors:\n" + "\n".join(errors)
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider}. Use kaznu, groq, or auto."
    )