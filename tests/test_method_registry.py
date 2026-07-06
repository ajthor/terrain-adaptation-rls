import unittest

from terrain_adaptation_rls.methods.registry import (
    default_method_registry,
    get_method_spec,
    validate_method_names,
)


class MethodRegistryTests(unittest.TestCase):
    def test_registry_includes_required_methods(self):
        registry = default_method_registry()

        self.assertIn("fe_rls", registry)
        self.assertIn("maml_online", registry)
        self.assertIn("neuralfly_style_rls", registry)
        self.assertTrue(registry["neuralfly_style_rls"].phoenix_compatible)

    def test_validate_method_names_returns_specs(self):
        specs = validate_method_names(("fe_rls", "linear_basis_rls"))

        self.assertEqual([spec.name for spec in specs], ["fe_rls", "linear_basis_rls"])
        self.assertFalse(specs[1].requires_training)

    def test_unknown_method_error_lists_known_methods(self):
        with self.assertRaisesRegex(KeyError, "Known methods"):
            get_method_spec("missing")


if __name__ == "__main__":
    unittest.main()
