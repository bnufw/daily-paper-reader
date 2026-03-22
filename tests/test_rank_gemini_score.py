import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_module(module_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


class RankGeminiScoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module('rank_gemini_score_mod', SRC_DIR / '3.rank_papers.py')

    def test_parse_score_results_accepts_markdown_wrapped_json(self):
        out = self.mod.parse_score_results(
            '```json\n{"results":[{"index":1,"score":55},{"index":0,"score":88}]}\n```',
            2,
        )
        self.assertEqual(out, [{'index': 0, 'score': 88.0}, {'index': 1, 'score': 55.0}])

    def test_score_documents_with_llm_retries_without_response_format(self):
        calls = []

        class FakeClient:
            def chat(self, messages, response_format=None):
                calls.append(response_format)
                if response_format is not None:
                    raise RuntimeError('response_format unsupported')
                return {
                    'content': '{"results":[{"index":0,"score":72},{"index":1,"score":12}]}'
                }

        out = self.mod.score_documents_with_llm(FakeClient(), 'query', ['doc1', 'doc2'])
        self.assertEqual(out, [{'index': 0, 'score': 72.0}, {'index': 1, 'score': 12.0}])
        self.assertEqual(calls, [{'type': 'json_object'}, None])

    def test_build_ranked_results_normalizes_scores(self):
        out = self.mod.build_ranked_results({0: 90.0, 1: 30.0}, ['p0', 'p1'], None)
        self.assertEqual(out[0]['paper_id'], 'p0')
        self.assertEqual(out[0]['star_rating'], 5)
        self.assertEqual(out[1]['paper_id'], 'p1')
        self.assertEqual(out[1]['star_rating'], 1)


if __name__ == '__main__':
    unittest.main()
