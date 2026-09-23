import unittest

from workers.dataset.profiler import profile, _parse_datetime


class DateProfileTests(unittest.TestCase):
    def test_mixed_real_dates_are_recognized_and_sorted_chronologically(self):
        values = ['2026/3/20 13:26', '2022/12/15 18:59', '2025-01-25T15:17:00', '2026年5月14日']
        result = profile([{'发布时间': v, '正文': '文本分析主题'} for v in values], ['发布时间', '正文'])
        self.assertEqual(result['profiles'][0]['inferredType'], 'datetime')
        self.assertEqual(result['profiles'][0]['parseSuccessRatio'], 1)
        self.assertEqual(result['timeCoverage']['start'], '2022-12-15T18:59:00')
        self.assertEqual(result['timeCoverage']['end'], '2026-05-14T00:00:00')
        self.assertEqual(result['columnRoles']['text'][0]['name'], '正文')

    def test_invalid_dates_are_not_invented(self):
        for value in ['2026/2/30', '2026-13-01', '昨天', '未知', '']:
            self.assertIsNone(_parse_datetime(value))


if __name__ == '__main__':
    unittest.main()
