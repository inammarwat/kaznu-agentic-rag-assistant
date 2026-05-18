import os
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def build_llm(model_name: str, temperature: float = 0.0) -> ChatOpenAI:
    base_url = os.getenv("AGENT_BASE_URL")
    api_key = os.getenv("AGENT_API_KEY")

    if not base_url:
        raise ValueError("AGENT_BASE_URL missing in .env")

    if not api_key:
        raise ValueError("AGENT_API_KEY missing in .env")

    http_client = httpx.Client(
        verify=False,
        timeout=60,
    )

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=2048,
        max_retries=2,
        timeout=60,
        http_client=http_client,
    )


def initialize_llm(temperature: float = 0.0) -> ChatOpenAI:
    primary_model = os.getenv("AGENT_MODEL", "Qwen/Qwen3.5-27B")
    fallback_model = os.getenv("AGENT_FALLBACK_MODEL", "Qwen/Qwen3.5-9B")

    errors = []

    for model_name in [primary_model, fallback_model]:
        try:
            llm = build_llm(model_name=model_name, temperature=temperature)
            response = llm.invoke("Say OK if you are working.")
            print(f"LLM initialized: {model_name} | Response: {response.content}")
            return llm

        except Exception as exc:
            errors.append(f"{model_name}: {exc}")
            print(f"Failed model: {model_name} | {exc}")

    raise RuntimeError(
        "All LLM models failed. Errors:\n" + "\n".join(errors)
    )