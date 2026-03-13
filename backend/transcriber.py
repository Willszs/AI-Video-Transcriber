import os
import re
from faster_whisper import WhisperModel
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

class Transcriber:
    """音频转录器，使用Faster-Whisper进行语音转文字"""
    
    def __init__(self, model_size: str = "base", default_language: Optional[str] = "zh"):
        """
        初始化转录器
        
        Args:
            model_size: Whisper模型大小 (tiny, base, small, medium, large, large-v3)
            default_language: 默认语言提示（如 zh, en）。None 表示自动检测。
        """
        self.model_size = model_size
        lang = (default_language or "").strip().lower()
        self.default_language = None if lang in {"", "auto", "none"} else lang
        # 速度优先：降低搜索宽度，避免高精度参数导致明显变慢
        self.beam_size = 3
        self.best_of = 3
        self.temperature = 0.0
        self.cpu_threads = max(1, os.cpu_count() or 4)
        self.model = None
        self.last_detected_language = None
        
    def _load_model(self):
        """延迟加载模型"""
        if self.model is None:
            logger.info(f"正在加载Whisper模型: {self.model_size}")
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=self.cpu_threads,
                )
                logger.info("模型加载完成")
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
                raise Exception(f"模型加载失败: {str(e)}")
    
    async def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """
        转录音频文件
        
        Args:
            audio_path: 音频文件路径
            language: 指定语言（可选，不指定则使用默认语言提示或自动检测）
            
        Returns:
            转录文本（纯文本）
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(audio_path):
                raise Exception(f"音频文件不存在: {audio_path}")
            
            # 加载模型
            self._load_model()
            
            logger.info(f"开始转录音频: {audio_path}")
            effective_language = language or self.default_language
            
            # 直接调用会阻塞事件循环；放入线程避免阻塞
            import asyncio
            def _do_transcribe():
                return self.model.transcribe(
                    audio_path,
                    language=effective_language,
                    beam_size=self.beam_size,
                    best_of=self.best_of,
                    temperature=self.temperature,
                    # 更稳健：开启VAD与阈值，降低静音/噪音导致的重复
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 900,  # 静音检测时长
                        "speech_pad_ms": 300  # 语音填充
                    },
                    no_speech_threshold=0.7,  # 无语音阈值
                    compression_ratio_threshold=2.3,  # 压缩比阈值，检测重复
                    log_prob_threshold=-1.0,  # 日志概率阈值
                    # 避免错误累积导致的连环重复
                    condition_on_previous_text=False
                )
            segments, info = await asyncio.to_thread(_do_transcribe)
            
            detected_language = info.language
            self.last_detected_language = detected_language  # 保存检测到的语言
            logger.info(f"检测到的语言: {detected_language}")
            logger.info(f"语言检测概率: {info.language_probability:.2f}")
            
            # 输出纯文本：不包含时间戳或元数据，只保留内容本身
            transcript_lines: List[str] = []
            for segment in segments:
                text = self._normalize_segment_text(segment.text, detected_language)
                if text:
                    transcript_lines.append(text)

            transcript_text = "\n".join(transcript_lines).strip()
            logger.info("转录完成")
            
            return transcript_text
            
        except Exception as e:
            logger.error(f"转录失败: {str(e)}")
            raise Exception(f"转录失败: {str(e)}")
    
    def _normalize_segment_text(self, text: str, language: Optional[str]) -> str:
        """规范Whisper片段文本，并在必要时补齐句末标点。"""
        normalized = re.sub(r"\s+", " ", (text or "").strip())
        if not normalized:
            return ""
        return self._ensure_sentence_end_punctuation(normalized, language)

    def _ensure_sentence_end_punctuation(self, text: str, language: Optional[str]) -> str:
        """若片段缺少句末标点，则按语言补齐。"""
        end_chars = "。！？.!?…；;:：”’」』）)]}"
        keep_tail = "，,、—-"
        if not text:
            return text
        if text[-1] in end_chars or text[-1] in keep_tail or len(text) < 6:
            return text
        punct = "。" if (language or "").lower().startswith("zh") else "."
        return f"{text}{punct}"
    
    def get_supported_languages(self) -> list:
        """
        获取支持的语言列表
        """
        return [
            "zh", "en", "ja", "ko", "es", "fr", "de", "it", "pt", "ru",
            "ar", "hi", "th", "vi", "tr", "pl", "nl", "sv", "da", "no"
        ]
    
    def get_detected_language(self, transcript_text: Optional[str] = None) -> Optional[str]:
        """
        获取检测到的语言
        
        Args:
            transcript_text: 转录文本（可选，用于从文本中提取语言信息）
            
        Returns:
            检测到的语言代码
        """
        # 如果有保存的语言，直接返回
        if self.last_detected_language:
            return self.last_detected_language
        
        # 如果提供了转录文本，尝试从中提取语言信息
        if transcript_text and "**Detected Language:**" in transcript_text:
            lines = transcript_text.split('\n')
            for line in lines:
                if "**Detected Language:**" in line:
                    lang = line.split(":")[-1].strip()
                    return lang
        
        return None
