
import unittest
from app.data_preprocessing import preprocess_text

class TestDataPreprocessing(unittest.TestCase):
    def test_preprocess_text(self):
        text = "This is a sample resume. Contact me at email@example.com"
        processed = preprocess_text(text)
        self.assertNotIn('email@example.com', processed)
        self.assertNotIn('.', processed)
        self.assertIn('sample resume', processed)

if __name__ == '__main__':
    unittest.main()
