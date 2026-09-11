import json
import unittest

from aorus_rag.chunker import build_chunks
from aorus_rag.scraper import parse_specs


class SpecsTests(unittest.TestCase):
    def test_models_keep_their_own_fields(self):
        data = []

        def add(value):
            data.append(value)
            return len(data) - 1

        # Deliberately use a different order from the visible page heading.
        for model, gpu in [('BXH', '5070 Ti'), ('BZH', '5090'), ('BYH', '5080')]:
            item = add({'itemTitle': add('顯示晶片'), 'itemContent': add(f'RTX {gpu}<br>GPU')})
            color = add({'itemTitle': add('顏色'), 'itemContent': add('Dark Tide')})
            add({'productId': add(model), 'title': add(f'AORUS MASTER 16 {model}'),
                 'specItem': add([item, color])})
        html = '<script id="__NUXT_DATA__" type="application/json">' + json.dumps(data) + '</script>'
        products = parse_specs(html)
        self.assertEqual(len(products), 3)
        for model, gpu in [('BZH', '5090'), ('BYH', '5080'), ('BXH', '5070 Ti')]:
            self.assertEqual(products[f'AORUS MASTER 16 {model}']['顯示晶片'], [f'RTX {gpu}', 'GPU'])
        chunks = build_chunks(products)
        self.assertEqual(len(chunks), 6)
        self.assertEqual(len({c['id'] for c in chunks}), 6)
        for chunk in chunks:
            self.assertIn(chunk['product'], chunk['text'])

    def test_missing_specs_fails(self):
        for html in ['<html></html>', '<script id="__NUXT_DATA__">[]</script>']:
            with self.assertRaises(ValueError):
                parse_specs(html)


if __name__ == '__main__':
    unittest.main()
