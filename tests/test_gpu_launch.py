import unittest

from terrain_adaptation_rls.launch.gpu import available_gpus, parse_nvidia_smi_csv


class GPULaunchTests(unittest.TestCase):
    def test_parse_nvidia_smi_csv(self):
        statuses = parse_nvidia_smi_csv(
            "0, NVIDIA A100-SXM4-40GB, 120, 40960, 0\n"
            "1, NVIDIA A100-SXM4-40GB, 24000, 40960, 80\n"
        )

        self.assertEqual(statuses[0].index, 0)
        self.assertEqual(statuses[0].memory_free_mb, 40840)
        self.assertEqual(statuses[1].utilization_percent, 80)

    def test_available_gpus_filters_busy_devices(self):
        statuses = parse_nvidia_smi_csv(
            "0, NVIDIA A100-SXM4-40GB, 120, 40960, 0\n"
            "1, NVIDIA A100-SXM4-40GB, 24000, 40960, 80\n"
        )

        self.assertEqual(available_gpus(statuses), [0])


if __name__ == "__main__":
    unittest.main()
