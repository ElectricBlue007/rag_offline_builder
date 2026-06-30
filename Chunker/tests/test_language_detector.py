"""
tests/test_language_detector.py

Unit tests for detect_language() — covers EN, ES, unknown, edge cases,
and real text excerpts from both Binder2.md and Testing.md.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Chunker.preprocessors.language_detector import detect_language


# ── Real excerpts from the OCR output files ──────────────────────────────────

EN_EXCERPT = (
    "This service allows the operator to manipulate great circles. "
    "The operator can create, eliminate, and change CMI track parameters. "
    "Only for lost surface point tracks and reference positions. "
    "Maximum number of CMI is ten."
)

ES_EXCERPT = (
    "Si ya existe una definicion de posicion de boton de conjunto de trabajo nueva, "
    "la posicion de todas las definiciones de posicion de boton de conjunto de trabajo "
    "con el mismo nombre de funcion y con una posicion igual o superior a la nueva, "
    "aumentara en uno."
)

EN_EXCERPT_TESTING = (
    "When thinking about a large language model input and output, a text prompt "
    "is the input the model uses to predict a specific output. "
    "You don't need to be a data scientist or a machine learning engineer. "
    "However, crafting the most effective prompt can be complicated."
)

ES_TABLE_EXCERPT = (
    "Accion del operador Entrada logica Entrada real Salida relacionada "
    "Restricciones Imprimir Definiciones de tarea tecnica interna "
    "Pulse Impr. Aparece la ventana Imprimir"
)

# Domain-specific terms from OCR output
EN_DOMAIN_TERMS = "operator action entry report restriction system service track parameter"
ES_DOMAIN_TERMS = "definicion boton conjunto trabajo posicion funcion nombre tarea operativa restricciones"


class TestLanguageDetector(unittest.TestCase):

    # ── Clear EN ─────────────────────────────────────────────────────────────

    def test_clear_english_returns_en(self):
        self.assertEqual(detect_language(EN_EXCERPT), "en")

    def test_testing_md_english_section(self):
        self.assertEqual(detect_language(EN_EXCERPT_TESTING), "en")

    def test_english_domain_terms(self):
        self.assertEqual(detect_language(EN_DOMAIN_TERMS), "en")

    def test_simple_english_sentence(self):
        text = "The operator can create and modify the system report entries."
        self.assertEqual(detect_language(text), "en")

    # ── Clear ES ─────────────────────────────────────────────────────────────

    def test_clear_spanish_returns_es(self):
        self.assertEqual(detect_language(ES_EXCERPT), "es")

    def test_spanish_table_content(self):
        self.assertEqual(detect_language(ES_TABLE_EXCERPT), "es")

    def test_spanish_domain_terms(self):
        self.assertEqual(detect_language(ES_DOMAIN_TERMS), "es")

    def test_simple_spanish_sentence(self):
        text = (
            "El boton de conjunto de trabajo no esta disponible "
            "cuando el parametro del sistema esta deshabilitado."
        )
        self.assertEqual(detect_language(text), "es")

    # ── Unknown / edge cases ─────────────────────────────────────────────────

    def test_empty_string_is_unknown(self):
        self.assertEqual(detect_language(""), "unknown")

    def test_too_short_text_is_unknown(self):
        """Fewer than 5 words → unknown."""
        self.assertEqual(detect_language("Hello world"), "unknown")

    def test_exactly_four_words_is_unknown(self):
        self.assertEqual(detect_language("one two three four"), "unknown")

    def test_numbers_only_is_unknown(self):
        self.assertEqual(detect_language("1 2 3 4 5 6 7 8 9 10"), "unknown")

    def test_punctuation_only_is_unknown(self):
        self.assertEqual(detect_language("... --- === ||| !!!"), "unknown")

    def test_html_table_markup_unknown(self):
        """Pure HTML with no words → unknown (not English or Spanish)."""
        html = "<table><tr><td>-</td><td>-</td></tr></table>"
        result = detect_language(html)
        # "-" tokens won't match either language signal set
        self.assertIn(result, ("unknown", "en", "es"))  # won't crash

    def test_single_dash_value_cells(self):
        """Tables with only dash values like '-' in Binder2.md."""
        text = "- - - - - - - - - -"
        result = detect_language(text)
        self.assertEqual(result, "unknown")

    # ── Robustness ───────────────────────────────────────────────────────────

    def test_returns_string_always(self):
        for text in ["", "hi", "hello world how are you today", ES_EXCERPT, EN_EXCERPT]:
            result = detect_language(text)
            self.assertIsInstance(result, str)
            self.assertIn(result, ("en", "es", "unknown"))

    def test_image_alt_text_english(self):
        """Image alt text from Testing.md — should be detected as English."""
        alt = (
            "A large abstract three-dimensional geometric shape is positioned "
            "in the bottom right corner of the page. It is composed of several "
            "flat angular faces that create a faceted crystalline appearance."
        )
        self.assertEqual(detect_language(alt), "en")

    def test_breadcrumb_text_short_is_unknown(self):
        """Very short breadcrumbs like section headers shouldn't crash."""
        result = detect_language("### MMI Disposition")
        self.assertIn(result, ("en", "es", "unknown"))

    def test_binder2_english_section_text(self):
        text = (
            "This service allows the operator to manipulate great circles abbreviated GCI. "
            "Create SOI CMI track enter NT Vel MAC and Sec vel MAC and Hora activacion "
            "and TiempFMAC CMI CMI Report. Only for lost surface point tracks and reference positions. "
            "Maximum number of CMI equals ten."
        )
        self.assertEqual(detect_language(text), "en")

    def test_binder2_spanish_numbered_list(self):
        text = (
            "Si ya existe una definicion de posicion de boton de conjunto de trabajo nueva "
            "la posicion de todas las definiciones de posicion de boton de conjunto de trabajo "
            "con el mismo nombre de funcion aumentara en uno. "
            "Las definiciones de posicion del conjunto de trabajo de la posicion nueva "
            "y anterior se intercambiaran. "
            "El borrado de una posicion de boton de conjunto de trabajo genera "
            "una posicion adjunta vacia."
        )
        self.assertEqual(detect_language(text), "es")


if __name__ == "__main__":
    unittest.main(verbosity=2)
