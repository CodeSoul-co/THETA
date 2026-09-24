import tempfile
import unittest
from pathlib import Path
from workers.capabilities import dataset_import, dataset_preview

class DatasetPreviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def preview(self, file):
        return dataset_preview({'dataset': dataset_import({'filePath': str(file), 'uploadDir': str(self.root / 'uploads')})})

    def test_text_preview_reads_first_records_without_column_selection(self):
        for suffix in ['.txt', '.md']:
            file = self.root / ('正文' + suffix)
            file.write_text('\n'.join('正文第%d行' % n for n in range(100)), encoding='utf8')
            preview = self.preview(file)
            self.assertEqual(preview['inputKind'], 'text')
            self.assertEqual(preview['textColumn'], 'text')
            self.assertEqual(preview['totalRecords'], 100)
            self.assertEqual([x['text'] for x in preview['segments']], ['正文第%d行' % n for n in range(5)])

    def test_structured_data_retains_columns_and_zero_values(self):
        file = self.root / 'table.csv'
        file.write_text('text,value\n正文,0\n', encoding='utf8')
        preview = self.preview(file)
        self.assertEqual(preview['inputKind'], 'table')
        self.assertEqual(preview['rows'], [['正文', '0']])
        self.assertNotIn('segments', preview)

    def test_docx_preview_uses_paragraphs(self):
        from docx import Document
        file = self.root / 'paragraphs.docx'
        doc = Document(); doc.add_paragraph('第一段'); doc.add_paragraph(''); doc.add_paragraph('第三段'); doc.save(file)
        preview = self.preview(file)
        self.assertEqual(preview['segments'], [{'text': '第一段', 'paragraph': 1}, {'text': '第三段', 'paragraph': 3}])

    def test_empty_pdf_explains_text_recognition(self):
        from PyPDF2 import PdfWriter
        file = self.root / 'scan.pdf'
        writer = PdfWriter(); writer.add_blank_page(width=100, height=100)
        with file.open('wb') as handle: writer.write(handle)
        with self.assertRaisesRegex(ValueError, '文字识别'):
            self.preview(file)
