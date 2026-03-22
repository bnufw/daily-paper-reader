const assert = require('node:assert/strict');

const {
  DEFAULT_GEMINI_OPENAI_BASE_URL,
  normalizeLlmBaseUrl,
  resolveChatCompletionsEndpoint,
} = require('../app/llm-api-utils.js');

assert.equal(
  normalizeLlmBaseUrl(`${DEFAULT_GEMINI_OPENAI_BASE_URL}/chat/completions`),
  DEFAULT_GEMINI_OPENAI_BASE_URL,
);
assert.equal(
  resolveChatCompletionsEndpoint(DEFAULT_GEMINI_OPENAI_BASE_URL),
  `${DEFAULT_GEMINI_OPENAI_BASE_URL}/chat/completions`,
);
assert.equal(
  resolveChatCompletionsEndpoint(`${DEFAULT_GEMINI_OPENAI_BASE_URL}/chat/completions`),
  `${DEFAULT_GEMINI_OPENAI_BASE_URL}/chat/completions`,
);
assert.equal(
  resolveChatCompletionsEndpoint('https://proxy.example.com/v1'),
  'https://proxy.example.com/v1/chat/completions',
);

console.log('llm api utils tests passed');
