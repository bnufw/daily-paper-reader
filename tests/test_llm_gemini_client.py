import os
import unittest
from unittest.mock import patch

from src.llm import (
    ClientFactory,
    DEFAULT_GEMINI_BASE_URL,
    GeminiClient,
    normalize_openai_base_url,
)


class GeminiClientCompatibilityTest(unittest.TestCase):
    def test_normalize_openai_base_url_strips_chat_suffix(self):
        self.assertEqual(
            normalize_openai_base_url(
                'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'
            ),
            DEFAULT_GEMINI_BASE_URL,
        )

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'new-key'}, clear=True)
    def test_gemini_client_defaults_to_official_base(self):
        client = GeminiClient(api_key='new-key', model='gemini-3-flash-preview')
        self.assertEqual(client._base_urls, [DEFAULT_GEMINI_BASE_URL])
        self.assertFalse(client.supports_rerank())

    @patch.dict(
        os.environ,
        {
            'BLT_API_KEY': 'legacy-key',
            'BLT_API_BASE': 'https://api.bltcy.ai/v1/chat/completions',
            'GPTBEST_BASE_URL': 'https://api.gptbest.vip/v1',
        },
        clear=True,
    )
    def test_gemini_client_preserves_legacy_blt_fallback(self):
        client = GeminiClient(api_key='legacy-key', model='gemini-3-flash-preview')
        self.assertIn('https://api.bltcy.ai/v1', client._base_urls)
        self.assertTrue(client.supports_rerank())

    @patch.dict(
        os.environ,
        {
            'LLM_MODEL': 'gemini/gemini-3-flash-preview',
            'LLM_API_KEY': 'factory-key',
        },
        clear=True,
    )
    def test_client_factory_supports_gemini_provider(self):
        client = ClientFactory.from_env()
        self.assertIsInstance(client, GeminiClient)
        self.assertEqual(client.model, 'gemini-3-flash-preview')


if __name__ == '__main__':
    unittest.main()
