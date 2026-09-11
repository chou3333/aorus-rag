"""Integration tests using the local embedding model and rebuilt index."""
import unittest
from aorus_rag.retriever import retrieve, calculate_alias_bonus


class RetrievalTests(unittest.TestCase):
    def test_model_specific_gpu(self):
        for model, gpu in [('BZH', '5090'), ('BYH', '5080'), ('BXH', '5070 Ti')]:
            for query in [f'{model} 的顯示卡是什麼？', f'What GPU does {model.lower()} use?']:
                with self.subTest(query=query):
                    results = retrieve(query, top_k=1)
                    self.assertEqual(len(results), 1)
                    self.assertTrue(results[0]['chunk']['product'].endswith(model))
                    self.assertIn(gpu, results[0]['chunk']['content'])

    def test_generic_and_comparison(self):
        for query, expected in [('這台筆電的 GPU 是什麼？', {'BZH', 'BYH', 'BXH'}),
                                ('BYH 和 BXH 的 GPU 差異？', {'BYH', 'BXH'})]:
            results = retrieve(query, top_k=1, per_product=True)
            self.assertEqual({r['chunk']['product'].split()[-1] for r in results}, expected)
            self.assertTrue(all(r['chunk']['category'] == '顯示晶片' for r in results))

    def test_alias_word_boundaries(self):
        self.assertEqual(calculate_alias_bonus('What GPU does this laptop use?', '作業系統'), 0)
