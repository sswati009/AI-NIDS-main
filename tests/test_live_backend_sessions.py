import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import live_backend


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.stderr = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        self.returncode = 0
        return self.returncode

    def kill(self):
        self.returncode = -9


class LiveBackendSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.live_dir = Path(self.temp_dir.name) / "live"
        self.live_dir.mkdir()

        self.legacy_flows = self.live_dir / "live_flows.csv"
        self.legacy_predictions = self.live_dir / "live_predictions.csv"
        self.old_pcap = self.live_dir / "prior-capture.pcap"
        self.old_flow_file = self.live_dir / "prior-capture_flows.csv"

        self.legacy_flows.write_bytes(b"flow header\nexisting flow\n")
        self.legacy_predictions.write_bytes(b"prediction header\nexisting alert\n")
        self.old_pcap.write_bytes(b"existing pcap bytes")
        self.old_flow_file.write_bytes(b"existing extracted flow bytes")
        self.original_bytes = {
            path: path.read_bytes()
            for path in (
                self.legacy_flows,
                self.legacy_predictions,
                self.old_pcap,
                self.old_flow_file,
            )
        }

        self.cicflowmeter = Path(self.temp_dir.name) / "cicflowmeter.exe"
        self.cicflowmeter.write_bytes(b"mock executable placeholder")

        self.live_dir_patch = patch.object(live_backend, "LIVE_DIR", self.live_dir)
        self.live_dir_patch.start()
        self.addCleanup(self.live_dir_patch.stop)
        self.cicflowmeter_patch = patch.object(
            live_backend, "CICFLOWMETER", self.cicflowmeter
        )
        self.cicflowmeter_patch.start()
        self.addCleanup(self.cicflowmeter_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)

    def assert_existing_capture_data_unchanged(self):
        for path, original in self.original_bytes.items():
            self.assertTrue(path.exists(), f"Existing file disappeared: {path}")
            self.assertEqual(path.read_bytes(), original, f"Existing file changed: {path}")

    def start_real_with_mocks(self, popen):
        nids = live_backend.LiveNIDS()
        nids.load_models = Mock()
        with patch.object(live_backend.subprocess, "Popen", popen), patch.object(
            live_backend.time, "sleep", return_value=None
        ):
            nids.start_real_capture("test-interface")
        return nids

    def test_each_session_gets_unique_timestamped_outputs_without_touching_old_files(self):
        nids = live_backend.LiveNIDS()

        nids.reset_session()
        first_id = nids.session_id
        first_flows = nids.live_csv
        first_predictions = nids.out_csv

        nids.reset_session()

        self.assertNotEqual(first_id, nids.session_id)
        self.assertRegex(first_id, r"^\d{8}-\d{6}-\d{6}-[0-9a-f]{8}$")
        self.assertIn(first_id, first_flows.name)
        self.assertIn(first_id, first_predictions.name)
        self.assertNotEqual(first_flows, nids.live_csv)
        self.assertNotEqual(first_predictions, nids.out_csv)
        self.assertFalse(first_flows.exists())
        self.assertFalse(first_predictions.exists())
        self.assert_existing_capture_data_unchanged()

    def test_successful_real_session_uses_new_paths_and_preserves_legacy_csvs(self):
        process = FakeProcess()
        popen = Mock(return_value=process)
        nids = self.start_real_with_mocks(popen)

        command = popen.call_args.args[0]
        output_path = Path(command[command.index("-c") + 1])
        self.assertEqual(output_path, nids.live_csv)
        self.assertNotEqual(nids.live_csv, self.legacy_flows)
        self.assertTrue(nids.live_csv.parent.exists())
        self.assert_existing_capture_data_unchanged()
        nids.stop_capture()

    def test_successful_lab_session_uses_unique_capture_prefix_without_cleanup(self):
        process = FakeProcess()
        nids = live_backend.LiveNIDS()
        nids._check_lab = Mock()
        nids._check_no_existing_lab_capture = Mock()
        nids._install_tcpdump_if_needed = Mock()
        nids.load_models = Mock()
        popen = Mock(return_value=process)

        def docker_run(args, **kwargs):
            if args[3:5] == ["pidof", "tcpdump"]:
                return live_backend.subprocess.CompletedProcess(
                    args, 1, stdout="", stderr=""
                )
            return live_backend.subprocess.CompletedProcess(
                args, 0, stdout="", stderr=""
            )

        docker_run = Mock(side_effect=docker_run)

        with patch.object(live_backend.subprocess, "Popen", popen), patch.object(
            live_backend.subprocess, "run", docker_run
        ), patch.object(live_backend.time, "sleep", return_value=None):
            nids.start_lab_capture()
            nids.stop_capture()

        command = popen.call_args.args[0]
        pcap_pattern = command[command.index("-w") + 1]
        self.assertIn(nids.session_id, pcap_pattern)
        self.assertIn(live_backend.LAB_CAPTURE_DIR, pcap_pattern)
        self.assertEqual(command[-2:], ["net", live_backend.LAB_CIDR])
        self.assertIn(nids.session_id, nids.live_csv.name)
        self.assertIn(nids.session_id, nids.out_csv.name)
        self.assertIn(
            ["docker", "exec", live_backend.LAB_CLIENT, "pkill", "-INT", "-x", "tcpdump"],
            [call.args[0] for call in docker_run.call_args_list],
        )
        self.assert_existing_capture_data_unchanged()

    def test_lab_preflight_failure_does_not_reset_state_or_touch_existing_files(self):
        nids = live_backend.LiveNIDS()
        nids.total_flows = 17
        nids.predictions = [{"prediction": "BENIGN"}]
        nids.last_error = "previous error"
        nids._check_lab = Mock(side_effect=RuntimeError("lab unavailable"))
        nids.load_models = Mock()

        with self.assertRaisesRegex(RuntimeError, "lab unavailable"):
            nids.start_lab_capture()

        nids._check_lab.assert_called_once_with()
        nids.load_models.assert_not_called()
        self.assertIsNone(nids.session_id)
        self.assertEqual(nids.total_flows, 17)
        self.assertEqual(nids.predictions, [{"prediction": "BENIGN"}])
        self.assertEqual(nids.last_error, "previous error")
        self.assert_existing_capture_data_unchanged()

    def test_stale_lab_capture_fails_before_session_state_is_reset(self):
        nids = live_backend.LiveNIDS()
        nids.total_flows = 29
        nids.predictions = [{"prediction": "BENIGN"}]
        nids._check_lab = Mock()
        nids._check_no_existing_lab_capture = Mock(
            side_effect=RuntimeError("tcpdump is already running")
        )
        nids._install_tcpdump_if_needed = Mock()
        nids.load_models = Mock()

        with self.assertRaisesRegex(RuntimeError, "already running"):
            nids.start_lab_capture()

        nids._check_lab.assert_called_once_with()
        nids._check_no_existing_lab_capture.assert_called_once_with()
        nids._install_tcpdump_if_needed.assert_not_called()
        nids.load_models.assert_not_called()
        self.assertIsNone(nids.session_id)
        self.assertEqual(nids.total_flows, 29)
        self.assertEqual(nids.predictions, [{"prediction": "BENIGN"}])
        self.assert_existing_capture_data_unchanged()

    def test_real_capture_launch_failure_preserves_existing_files(self):
        nids = live_backend.LiveNIDS()
        nids.load_models = Mock()

        with patch.object(
            live_backend.subprocess,
            "Popen",
            side_effect=OSError("mock launch failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "mock launch failure"):
                nids.start_real_capture("test-interface")

        self.assertIsNotNone(nids.session_id)
        self.assertFalse(nids.live_csv.exists())
        self.assertFalse(nids.out_csv.exists())
        self.assert_existing_capture_data_unchanged()

    def test_lab_capture_launch_failure_preserves_existing_files(self):
        nids = live_backend.LiveNIDS()
        nids._check_lab = Mock()
        nids._check_no_existing_lab_capture = Mock()
        nids._install_tcpdump_if_needed = Mock()
        nids.load_models = Mock()

        with patch.object(
            live_backend.subprocess,
            "Popen",
            side_effect=OSError("mock docker exec launch failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "mock docker exec launch failure"):
                nids.start_lab_capture()

        self.assertIsNotNone(nids.session_id)
        self.assertFalse(nids.live_csv.exists())
        self.assertFalse(nids.out_csv.exists())
        self.assert_existing_capture_data_unchanged()

    def test_reset_all_clears_counters_without_deleting_capture_files(self):
        nids = live_backend.LiveNIDS()
        nids.total_flows = 9
        nids.predictions = [{"prediction": "ATTACK"}]

        nids.reset_all()

        self.assertEqual(nids.total_flows, 0)
        self.assertEqual(nids.predictions, [])
        self.assert_existing_capture_data_unchanged()

    def test_reset_all_preserves_session_state_when_capture_stop_fails(self):
        nids = live_backend.LiveNIDS()
        process = FakeProcess()
        nids.capture_mode = "LAB"
        nids.lab_capture_process = process
        nids.total_flows = 13
        nids.processed_rows = 7
        nids.stop_capture = Mock()

        nids.reset_all()

        self.assertIs(nids.lab_capture_process, process)
        self.assertEqual(nids.total_flows, 13)
        self.assertEqual(nids.processed_rows, 7)

    def test_live_directory_error_is_reported_without_resetting_state(self):
        blocked_live_dir = Path(self.temp_dir.name) / "not-a-directory"
        blocked_live_dir.write_bytes(b"preserve this file")
        nids = live_backend.LiveNIDS()
        nids.total_flows = 23

        with patch.object(live_backend, "LIVE_DIR", blocked_live_dir):
            with self.assertRaisesRegex(OSError, "Could not prepare live capture directory"):
                nids.reset_session()

        self.assertIsNone(nids.session_id)
        self.assertEqual(nids.total_flows, 23)
        self.assertEqual(blocked_live_dir.read_bytes(), b"preserve this file")
        self.assert_existing_capture_data_unchanged()


if __name__ == "__main__":
    unittest.main()
