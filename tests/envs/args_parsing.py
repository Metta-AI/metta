import unittest
from unittest.mock import patch
import util.args_parsing as args_parsing
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--int_range', action=args_parsing.PossiblyNumericRange2Number, str2numeric_cast_fn = int)
    return parser.parse_args()

class TestCustomArgparseAction(unittest.TestCase):

    @patch('sys.argv', ['test_script.py', '--int_range', '5'])
    def test_possibly_numeric_range_single_value(self):
        args = parse_args()
        self.assertEqual(args.int_range, [5])

    @patch('sys.argv', ['test_script.py', '--int_range', '5:15'])
    def test_possibly_numeric_range_two_values(self):
        args = parse_args()
        self.assertEqual(args.int_range, [5, 15])

if __name__ == "__main__":
    unittest.main()
