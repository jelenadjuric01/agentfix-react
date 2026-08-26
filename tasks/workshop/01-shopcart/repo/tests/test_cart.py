import unittest

from shopcart.cart import subtotal, total_with_tax


class TestCart(unittest.TestCase):
    def test_subtotal(self):
        self.assertEqual(subtotal([1.0, 2.0, 3.0]), 6.0)

    def test_total_with_tax(self):
        self.assertEqual(total_with_tax([10.0]), 12.0)
