import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from opencc import OpenCC  # type: ignore
except Exception:  # pragma: no cover
    OpenCC = None

_converter = None
_converter_initialized = False
_warned_missing_opencc = False


def is_chinese_language(language_hint: Optional[str]) -> bool:
    if not language_hint:
        return False
    lang = language_hint.strip().lower().replace("_", "-")
    return (
        lang.startswith("zh")
        or lang in {"chinese", "cmn"}
    )


def to_simplified_chinese(text: str, language_hint: Optional[str] = None) -> str:
    """在中文场景下将文本统一转换为简体。"""
    if not text:
        return text

    should_convert = is_chinese_language(language_hint)
    if not should_convert:
        return text

    converter = _get_opencc_converter()
    if not converter:
        return text

    try:
        return converter.convert(text)
    except Exception as e:  # pragma: no cover
        logger.warning(f"简繁转换失败，保留原文: {e}")
        return text


def _get_opencc_converter():
    global _converter, _converter_initialized, _warned_missing_opencc

    if _converter_initialized:
        return _converter

    _converter_initialized = True
    if OpenCC is None:
        if not _warned_missing_opencc:
            logger.warning("未安装opencc，无法进行繁体到简体转换")
            _warned_missing_opencc = True
        return None

    try:
        _converter = OpenCC("t2s")
    except Exception as e:  # pragma: no cover
        logger.warning(f"初始化OpenCC失败，无法进行简繁转换: {e}")
        _converter = None
    return _converter
