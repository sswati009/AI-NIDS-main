import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import live_backend


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.wait_calls = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        self.returncode = 0
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9


class LiveBackendPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.session_id = "20261003-010203-123456-testabcd"
        self.session_dir = self.root / "sessions" / self.session_id
        self.session_dir.mkdir(parents=True)
        self.base_patch = patch.object(live_backend, "BASE", self.root)
        self.base_patch.start()
        self.addCleanup(self.base_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)

    def new_backend(self):
        nids = live_backend.LiveNIDS()
        nids.session_id = self.session_id
        nids.session_dir = self.session_dir
        nids.live_csv = self.session_dir / f"live_flows_{self.session_id}.csv"
        nids.out_csv = self.session_dir / f"live_predictions_{self.session_id}.csv"
        nids.capture_mode = "LAB"
        return nids

    @staticmethod
    def completed(args, returncode=0, stdout="", stderr=""):
        return live_backend.subprocess.CompletedProcess(
            args, returncode, stdout, stderr
        )

    def test_final_copy_includes_newest_pcap_and_retains_remote_until_conversion(self):
        nids = self.new_backend()
        oldest = (
            f"{live_backend.LAB_CAPTURE_DIR}/ai-nids-lab-"
            f"{self.session_id}-20261003-010200.pcap"
        )
        newest = (
            f"{live_backend.LAB_CAPTURE_DIR}/ai-nids-lab-"
            f"{self.session_id}-20261003-010205.pcap"
        )
        cp_calls = []

        def docker_run(args, **kwargs):
            if args[:4] == ["docker", "exec", live_backend.LAB_CLIENT, "find"]:
                return self.completed(args, stdout=f"{newest}\n{oldest}\n")
            if args[:2] == ["docker", "cp"]:
                cp_calls.append(args[2])
                Path(args[3]).write_bytes(b"p" * 128)
                return self.completed(args)
            self.fail(f"Unexpected Docker command: {args}")

        with patch.object(live_backend.subprocess, "run", side_effect=docker_run):
            copied_while_running = nids._copy_completed_lab_pcaps()
            copied_at_stop = nids._copy_completed_lab_pcaps(include_latest=True)

        self.assertEqual([path.name for path in copied_while_running], [Path(oldest).name])
        self.assertEqual([path.name for path in copied_at_stop], [Path(newest).name])
        self.assertEqual(cp_calls, [f"{live_backend.LAB_CLIENT}:{oldest}", f"{live_backend.LAB_CLIENT}:{newest}"])
        self.assertEqual(set(nids.lab_remote_pcaps.values()), {oldest, newest})
        self.assertTrue((self.session_dir / Path(oldest).name).exists())
        self.assertTrue((self.session_dir / Path(newest).name).exists())

    def test_finalized_latest_pcap_is_copied_converted_and_archived(self):
        nids = self.new_backend()
        remote = (
            f"{live_backend.LAB_CAPTURE_DIR}/ai-nids-lab-"
            f"{self.session_id}-20261003-010205.pcap"
        )
        command_order = []

        def docker_run(args, **kwargs):
            command_order.append(args)
            if args[:4] == ["docker", "exec", live_backend.LAB_CLIENT, "find"]:
                return self.completed(args, stdout=remote + "\n")
            if args[:2] == ["docker", "cp"]:
                Path(args[3]).write_bytes(b"p" * 128)
                return self.completed(args)
            if args[:5] == ["docker", "exec", live_backend.LAB_CLIENT, "rm", "-f"]:
                return self.completed(args)
            self.fail(f"Unexpected Docker command: {args}")

        def convert_run(args, **kwargs):
            command_order.append(args)
            output = Path(args[args.index("-c") + 1])
            output.write_text(
                "src_ip,dst_ip,src_port,dst_port,protocol,timestamp,flow_duration\n"
                "172.30.50.10,172.30.50.20,50000,3000,6,t,1\n",
                encoding="utf-8",
            )
            return self.completed(args)

        def run(args, **kwargs):
            if args and args[0] == "docker":
                return docker_run(args, **kwargs)
            return convert_run(args, **kwargs)

        with patch.object(live_backend.subprocess, "run", side_effect=run):
            self.assertTrue(nids._process_lab_chunks(finalize=True))

        self.assertEqual(len(pd.read_csv(nids.live_csv)), 1)
        self.assertIn(remote, nids.lab_capture_files)
        self.assertIn(Path(remote).name, nids.lab_processed_pcaps)
        self.assertTrue((self.root / "debug_pcaps" / Path(remote).name).exists())
        converter_index = next(
            i for i, command in enumerate(command_order)
            if command and command[0] != "docker"
        )
        remote_remove_index = next(
            i for i, command in enumerate(command_order)
            if command[:5] == ["docker", "exec", live_backend.LAB_CLIENT, "rm", "-f"]
        )
        self.assertLess(converter_index, remote_remove_index)

    def test_stop_waits_for_tcpdump_exit_before_final_processing(self):
        nids = self.new_backend()
        process = FakeProcess()
        nids.lab_capture_process = process
        run_results = [
            self.completed([], stdout=""),  # pkill -INT
            self.completed([], returncode=0, stdout="123"),  # still running
            self.completed([], returncode=1),  # exited
        ]

        def process_final(**kwargs):
            self.assertEqual(process.returncode, 0)
            self.assertEqual(kwargs, {"force": True, "finalize": True})

        nids.process_new_flows = Mock(side_effect=process_final)
        with patch.object(
            live_backend.subprocess, "run", side_effect=run_results
        ) as docker_run, patch.object(live_backend.time, "sleep") as sleep:
            nids.stop_capture()

        self.assertEqual(docker_run.call_count, 3)
        sleep.assert_called_once_with(0.2)
        self.assertEqual(process.wait_calls, [10])
        nids.process_new_flows.assert_called_once_with(force=True, finalize=True)
        self.assertIsNone(nids.lab_capture_process)

    def test_stop_timeout_preserves_capture_process_and_skips_final_processing(self):
        nids = self.new_backend()
        process = FakeProcess()
        nids.lab_capture_process = process
        nids.process_new_flows = Mock()
        run_results = [
            self.completed([], stdout=""),
            self.completed([], returncode=0, stdout="123"),
        ]

        with patch.object(
            live_backend.subprocess, "run", side_effect=run_results
        ), patch.object(
            live_backend.time, "monotonic", side_effect=[0, 11]
        ), patch.object(live_backend.time, "sleep"):
            nids.stop_capture()

        self.assertIs(nids.lab_capture_process, process)
        self.assertIsNone(process.returncode)
        nids.process_new_flows.assert_not_called()
        self.assertIn("did not exit", nids.last_error)

    def test_failed_conversion_is_retained_and_a_later_attempt_succeeds_once(self):
        nids = self.new_backend()
        pcap = self.session_dir / f"ai-nids-lab-{self.session_id}-chunk.pcap"
        pcap.write_bytes(b"p" * 12)
        converter_calls = []

        def converter_run(args, **kwargs):
            output = Path(args[args.index("-c") + 1])
            converter_calls.append(output)
            if len(converter_calls) == 1:
                output.write_text("partial", encoding="utf-8")
                return self.completed(args, returncode=1, stderr="temporary failure")
            output.write_text(
                "src_ip,dst_ip,src_port,dst_port,protocol,timestamp,flow_duration\n"
                "172.30.50.10,172.30.50.20,50000,3000,6,t,1\n",
                encoding="utf-8",
            )
            return self.completed(args)

        with patch.object(live_backend.subprocess, "run", side_effect=converter_run):
            self.assertFalse(nids._convert_lab_pcap(pcap))
            self.assertTrue(pcap.exists())
            self.assertEqual(converter_calls, [])

            # A completed copy becomes eligible on a later processing pass.
            pcap.write_bytes(b"p" * 128)
            self.assertFalse(nids._convert_lab_pcap(pcap))
            failed_csv = converter_calls[0]
            self.assertTrue(pcap.exists())
            self.assertTrue(failed_csv.exists())
            self.assertFalse(nids.live_csv.exists())

            self.assertTrue(nids._convert_lab_pcap(pcap))

        self.assertEqual(len(converter_calls), 2)
        self.assertNotEqual(converter_calls[0], converter_calls[1])
        self.assertTrue(failed_csv.exists())
        self.assertEqual(len(pd.read_csv(nids.live_csv)), 1)
        self.assertIn(pcap.name, nids.lab_processed_pcaps)
        self.assertTrue((self.root / "debug_pcaps" / pcap.name).exists())

        # A repeated processing attempt cannot append the same PCAP's flow again.
        self.assertTrue(nids._convert_lab_pcap(pcap))
        self.assertEqual(len(pd.read_csv(nids.live_csv)), 1)

    def make_flow_csv(self, nids):
        pd.DataFrame(
            [
                {
                    "src_ip": "172.30.50.10",
                    "dst_ip": "172.30.50.20",
                    "src_port": 50000,
                    "dst_port": 3000,
                    "protocol": 6,
                    "timestamp": "2026-10-03 01:02:03",
                    "flow_duration": 1,
                }
            ]
        ).to_csv(nids.live_csv, index=False)
        nids.lab_capture_process = FakeProcess()
        nids._process_lab_chunks = Mock(return_value=True)
        nids.make_features = Mock(return_value=object())

    def test_prediction_count_mismatch_does_not_advance_cursor_or_counters(self):
        nids = self.new_backend()
        self.make_flow_csv(nids)
        nids.predict = Mock(return_value=[])

        nids.process_new_flows()

        self.assertEqual(nids.processed_rows, 0)
        self.assertEqual(nids.total_flows, 0)
        self.assertEqual(nids.predictions, [])
        self.assertFalse(nids.out_csv.exists())
        self.assertIn("0 predictions for 1 flow rows", nids.last_error)

    def test_prediction_write_failure_retries_without_duplicate_counters_or_rows(self):
        nids = self.new_backend()
        self.make_flow_csv(nids)
        nids.predict = Mock(return_value=[("BENIGN", 0.1, "BENIGN", 0.9, "NONE")])
        atomic_write = nids._append_dataframe_atomically
        write_attempts = 0

        def fail_first_write(frame, path):
            nonlocal write_attempts
            write_attempts += 1
            if write_attempts == 1:
                raise OSError("mock disk failure")
            return atomic_write(frame, path)

        nids._append_dataframe_atomically = Mock(side_effect=fail_first_write)

        nids.process_new_flows()
        self.assertEqual(nids.processed_rows, 0)
        self.assertEqual(nids.total_flows, 0)
        self.assertEqual(nids.predictions, [])
        self.assertFalse(nids.out_csv.exists())

        nids.process_new_flows()
        nids.process_new_flows()

        self.assertEqual(nids.processed_rows, 1)
        self.assertEqual(nids.total_flows, 1)
        self.assertEqual(nids.total_benign, 1)
        self.assertEqual(nids.total_attacks, 0)
        self.assertEqual(len(nids.predictions), 1)
        self.assertEqual(len(pd.read_csv(nids.out_csv)), 1)


if __name__ == "__main__":
    unittest.main()
