"""
Конфигурация моделей GigaChat.

Поддерживает два контура:
- sigma: прямое подключение к GigaChat API
- sds: подключение через AI Gateway

Классы:
- ModelsConfig: загружает настройки из .env и формирует конфиги для LLM и эмбеддингов

Имя модели судьи выбирается в порядке: явный аргумент (настройка узла) →
переменная окружения MODEL → дефолт модуля. Дефолт — GigaChat-3-Ultra:
судья размечает по многостраничной инструкции с few-shot примерами, и
точность разметки (Acc_auto) прямо зависит от класса модели.
"""

import os
from typing import Any

from dotenv import load_dotenv

# Default values
DEFAULT_TEMPERATURE: float = 0.001
DEFAULT_TOP_P: float = 0.001

# Модель по умолчанию, когда её не задали ни настройкой узла, ни окружением.
DEFAULT_GIGACHAT_MODEL = "GigaChat-3-Ultra"


# =====================================================
# MODELS CONFIG
# =====================================================


class ModelsConfig:
    """Конфигурация для GigaChat модели и эмбеддингов."""

    contour: str
    verify_ssl_certs: bool
    contour_configs: dict[str, Any]
    llm_params: dict[str, Any]
    contour_llm_configs: dict[str, Any]

    def __init__(self, model: str = "") -> None:
        load_dotenv()

        # Get environment variables
        model = (str(model).strip() or os.environ.get("MODEL")
                 or DEFAULT_GIGACHAT_MODEL)
        credentials = os.environ.get("CREDENTIALS")
        auth_url = os.environ.get("AUTH_URL")
        base_url = os.environ.get("BASE_URL")
        scope = os.environ.get("SCOPE")
        temperature = float(os.environ.get("TEMPERATURE", DEFAULT_TEMPERATURE))
        top_p = float(os.environ.get("TOP_P", DEFAULT_TOP_P))
        verify_ssl_certs = os.environ.get("VERIFY_SSL_CERTS") == "True"
        self.verify_ssl_certs = verify_ssl_certs
        timeout = os.environ.get("TIMEOUT")
        streaming = os.environ.get("STREAMING") == "True"

        # Determine contour (sigma or sds)
        ai_gateway_url = os.environ.get("AI_GATEWAY_URL")
        if ai_gateway_url is not None:
            self.contour = "sds"
            base_url = ai_gateway_url.rstrip("/")
            if not base_url.endswith("/api/v1"):
                base_url = f"{base_url}/api/v1"
        else:
            self.contour = "sigma"

        # Build configs
        self.contour_configs = self._build_contour_configs(
            base_url, auth_url, credentials, verify_ssl_certs, scope
        )
        self.llm_params = self._build_llm_params(
            model, temperature, top_p, timeout, streaming
        )
        self.contour_llm_configs = self.llm_params | self.contour_configs

    def _build_contour_configs(
        self,
        base_url: str,
        auth_url: str | None,
        credentials: str | None,
        verify_ssl_certs: bool,
        scope: str | None,
    ) -> dict[str, Any]:
        """Build contour-specific configuration."""
        shared_config = {"base_url": base_url}

        sigma_config = {
            "auth_url": auth_url,
            "credentials": credentials,
            "verify_ssl_certs": verify_ssl_certs,
            "scope": scope,
        } | shared_config

        return {
            "sigma": sigma_config,
            "sds": shared_config,
        }[self.contour]

    def _build_llm_params(
        self,
        model: str | None,
        temperature: float,
        top_p: float,
        timeout: str | None,
        streaming: bool,
    ) -> dict[str, Any]:
        """Build LLM parameters."""
        return {
            "temperature": temperature,
            "model": model,
            "top_p": top_p,
            "timeout": timeout,
            "streaming": streaming,
        }
