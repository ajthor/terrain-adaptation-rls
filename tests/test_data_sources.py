import json
import tempfile
from pathlib import Path
import unittest

from terrain_adaptation_rls.data.sources import (
    discover_scene_sources,
    source_audit_csv,
    source_audit_rows,
    summarize_csv,
    write_source_manifest,
)


class DataSourceTests(unittest.TestCase):
    def test_summarize_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "odom.csv"
            path.write_text("time,x\n0.0,1\n0.2,2\n")

            summary = summarize_csv(path, include_hash=True)

        self.assertEqual(summary.rows, 2)
        self.assertEqual(summary.header, ("time", "x"))
        self.assertEqual(summary.first_time, 0.0)
        self.assertEqual(summary.last_time, 0.2)
        self.assertIsNotNone(summary.sha256)

    def test_discover_scene_sources_keeps_nested_scene_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = Path(tmpdir) / "warty" / "short_bags" / "grass"
            scene_dir.mkdir(parents=True)
            (scene_dir / "odom.csv").write_text("time,x\n0.0,1\n")
            (scene_dir / "cmd_vel.csv").write_text("time,u\n0.0,0\n")
            (scene_dir / "triggers.csv").write_text("time,label\n0.0,start\n")

            scenes = discover_scene_sources(tmpdir, "warty")

        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0].scene, "short_bags/grass")
        self.assertEqual(set(scenes[0].files), {"odom", "cmd_vel", "triggers"})

    def test_write_source_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = Path(tmpdir) / "warty" / "scene0"
            scene_dir.mkdir(parents=True)
            (scene_dir / "odom.csv").write_text("time,x\n0.0,1\n")
            (scene_dir / "cmd_vel.csv").write_text("time,u\n0.0,0\n")
            scenes = discover_scene_sources(tmpdir, "warty")

            manifest_path = Path(tmpdir) / "manifest.json"
            write_source_manifest(manifest_path, scenes, metadata={"platform": "warty"})
            manifest = json.loads(manifest_path.read_text())

        self.assertEqual(manifest["metadata"], {"platform": "warty"})
        self.assertEqual(manifest["scenes"][0]["scene"], "scene0")

    def test_source_audit_rows_and_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = Path(tmpdir) / "warty" / "scene0"
            scene_dir.mkdir(parents=True)
            (scene_dir / "odom.csv").write_text("time,x\n0.0,1\n0.5,2\n")
            (scene_dir / "cmd_vel.csv").write_text("time,u\n0.0,0\n")
            scenes = discover_scene_sources(tmpdir, "warty")

            rows = source_audit_rows(scenes)
            csv_table = source_audit_csv(scenes)

        self.assertEqual(rows[0]["duration_seconds"], 0.5)
        self.assertIn("platform,scene,odom_rows", csv_table)
        self.assertIn("warty,scene0,2,1,0.5,False", csv_table)


if __name__ == "__main__":
    unittest.main()
