import unittest
from src.language_detector import detect_code_language

class TestLanguageDetector(unittest.TestCase):
    def test_python_detection(self):
        code = "def test(): pass"
        self.assertEqual(detect_code_language(code), "python")
    
    def test_c_detection(self):
        code = "#include <stdio.h>\nint main() { return 0; }"
        self.assertEqual(detect_code_language(code), "c")

if __name__ == "__main__":
    unittest.main()