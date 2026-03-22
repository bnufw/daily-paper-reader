const DPR_LLM_API_UTILS_ROOT = typeof window !== 'undefined' ? window : globalThis;

DPR_LLM_API_UTILS_ROOT.DPRLLMApiUtils = (function () {
  const DEFAULT_GEMINI_OPENAI_BASE_URL =
    'https://generativelanguage.googleapis.com/v1beta/openai';
  const DEFAULT_GEMINI_FLASH_MODEL = 'gemini-3-flash-preview';
  const DEFAULT_GEMINI_PRO_MODEL = 'gemini-3.1-pro-preview';

  const normalizeText = (value) => String(value || '').trim();

  const normalizeLlmBaseUrl = (value, fallbackBaseUrl = DEFAULT_GEMINI_OPENAI_BASE_URL) => {
    const raw = normalizeText(value) || normalizeText(fallbackBaseUrl);
    if (!raw) return '';
    return raw
      .replace(/\/chat\/completions\/?$/i, '')
      .replace(/\/+$/, '');
  };

  const resolveChatCompletionsEndpoint = (
    value,
    fallbackBaseUrl = DEFAULT_GEMINI_OPENAI_BASE_URL,
  ) => {
    const normalized = normalizeLlmBaseUrl(value, fallbackBaseUrl);
    if (!normalized) return '';
    if (/\/openai$/i.test(normalized)) {
      return `${normalized}/chat/completions`;
    }
    if (/\/v\d+(?:beta\d+)?$/i.test(normalized)) {
      return `${normalized}/chat/completions`;
    }
    return `${normalized}/v1/chat/completions`;
  };

  return {
    DEFAULT_GEMINI_OPENAI_BASE_URL,
    DEFAULT_GEMINI_FLASH_MODEL,
    DEFAULT_GEMINI_PRO_MODEL,
    normalizeLlmBaseUrl,
    resolveChatCompletionsEndpoint,
  };
})();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = DPR_LLM_API_UTILS_ROOT.DPRLLMApiUtils;
}
