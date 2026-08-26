import unittest

from billing.invoice import invoice_total, line_total


class TestInvoice(unittest.TestCase):
    def test_line_total(self):
        self.assertEqual(line_total(2.0, 5), 10.0)

    def test_invoice_total_without_discount(self):
        self.assertEqual(invoice_total(2.0, 5), 10.0)

    def test_invoice_total_applies_bulk_discount(self):
        self.assertEqual(invoice_total(2.0, 10), 18.0)
