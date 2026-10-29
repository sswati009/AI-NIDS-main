import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import live_backend


BASE = Path(live_backend.__file__).resolve().parent


def completed_json(value):
    return subprocess.CompletedProcess([], 0, json.dumps([value]), "")


def valid_network():
    return {
        "Driver": "bridge",
        "Internal": True,
        "IPAM": {"Config": [{"Subnet": live_backend.LAB_CIDR}]},
        "Containers": {
            "client-id": {
                "Name": live_backend.LAB_CLIENT,
                "IPv4Address": live_backend.LAB_CLIENT_IP + "/24",
            },
            "target-id": {
                "Name": live_backend.LAB_TARGET,
                "IPv4Address": live_backend.LAB_TARGET_IP + "/24",
            },
        },
    }


def valid_container(name, address):
    host_config = {
        "PortBindings": None,
        "Privileged": False,
        "CapDrop": ["ALL"],
        "CapAdd": ["NET_RAW"] if name == live_backend.LAB_CLIENT else None,
    }
    return {
        "Name": "/" + name,
        "State": {"Running": True},
        "HostConfig": host_config,
        "NetworkSettings": {
            "Networks": {
                live_backend.LAB_NETWORK: {"IPAddress": address},
            },
        },
    }


class DockerLabIsolationTests(unittest.TestCase):
    def setUp(self):
        self.nids = live_backend.LiveNIDS()

    def valid_inspections(self):
        return [
            completed_json(valid_network()),
            completed_json(valid_container(live_backend.LAB_CLIENT, live_backend.LAB_CLIENT_IP)),
            completed_json(valid_container(live_backend.LAB_TARGET, live_backend.LAB_TARGET_IP)),
        ]

    def test_inspect_docker_decodes_utf8_image_metadata(self):
        target = valid_container(live_backend.LAB_TARGET, live_backend.LAB_TARGET_IP)
        label_value = "\u201d2026-08-11T05:21:13Z\u201d"
        target["Config"] = {
            "Labels": {"org.opencontainers.image.created": label_value}
        }
        result = subprocess.CompletedProcess(
            [], 0, json.dumps([target], ensure_ascii=False), ""
        )

        with patch.object(live_backend.subprocess, "run", return_value=result) as run:
            inspected = self.nids._inspect_docker(
                ["inspect", live_backend.LAB_TARGET],
                f"Docker container '{live_backend.LAB_TARGET}'",
            )

        self.assertEqual(
            inspected["Config"]["Labels"]["org.opencontainers.image.created"],
            label_value,
        )
        self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")

    def test_preflight_accepts_only_expected_internal_topology(self):
        with patch.object(
            live_backend.subprocess, "run", side_effect=self.valid_inspections()
        ) as run:
            self.nids._check_lab()

        self.assertEqual(run.call_count, 3)
        self.assertEqual(
            run.call_args_list[0].args[0],
            ["docker", "network", "inspect", live_backend.LAB_NETWORK],
        )

    def test_preflight_rejects_non_internal_network(self):
        network = valid_network()
        network["Internal"] = False
        with patch.object(
            live_backend.subprocess,
            "run",
            return_value=completed_json(network),
        ):
            with self.assertRaisesRegex(RuntimeError, "must be an internal bridge"):
                self.nids._check_lab()

    def test_preflight_accepts_docker_capability_name_prefix(self):
        client = valid_container(live_backend.LAB_CLIENT, live_backend.LAB_CLIENT_IP)
        client["HostConfig"]["CapAdd"] = ["CAP_NET_RAW"]

        with patch.object(
            live_backend.subprocess,
            "run",
            side_effect=[
                completed_json(valid_network()),
                completed_json(client),
                completed_json(valid_container(live_backend.LAB_TARGET, live_backend.LAB_TARGET_IP)),
            ],
        ):
            self.nids._check_lab()

    def test_preflight_rejects_extra_network_attachment_and_published_port(self):
        client = valid_container(live_backend.LAB_CLIENT, live_backend.LAB_CLIENT_IP)
        client["NetworkSettings"]["Networks"]["bridge"] = {"IPAddress": "172.17.0.3"}
        with patch.object(
            live_backend.subprocess,
            "run",
            side_effect=[completed_json(valid_network()), completed_json(client)],
        ):
            with self.assertRaisesRegex(RuntimeError, "attached only"):
                self.nids._check_lab()

        target = valid_container(live_backend.LAB_TARGET, live_backend.LAB_TARGET_IP)
        target["HostConfig"]["PortBindings"] = {"3000/tcp": [{"HostPort": "3000"}]}
        with patch.object(
            live_backend.subprocess,
            "run",
            side_effect=[
                completed_json(valid_network()),
                completed_json(valid_container(live_backend.LAB_CLIENT, live_backend.LAB_CLIENT_IP)),
                completed_json(target),
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "must not publish host ports"):
                self.nids._check_lab()

    def test_capture_preflight_refuses_a_leftover_remote_tcpdump(self):
        with patch.object(
            live_backend.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, "123", ""),
        ):
            with self.assertRaisesRegex(RuntimeError, "already running"):
                self.nids._check_no_existing_lab_capture()

    def test_missing_tcpdump_fails_without_runtime_package_install(self):
        missing = subprocess.CompletedProcess([], 1, "", "not found")
        with patch.object(live_backend.subprocess, "run", return_value=missing) as run:
            with self.assertRaisesRegex(RuntimeError, "tcpdump is missing"):
                self.nids._install_tcpdump_if_needed()

        self.assertEqual(run.call_count, 1)
        self.assertEqual(
            run.call_args.args[0],
            ["docker", "exec", live_backend.LAB_CLIENT, "sh", "-c", "command -v tcpdump"],
        )

    def test_lab_port_scan_targets_juice_shop_address(self):
        self.nids._check_lab = lambda: None
        calls = []
        self.nids._lab_exec = lambda command, timeout=30: calls.append(command)

        self.nids.lab_port_scan_test()

        self.assertIn(live_backend.LAB_TARGET_IP, calls[0])
        self.assertNotIn(live_backend.LAB_CLIENT_IP, calls[0])

    def test_compose_declares_internal_network_and_pinned_images(self):
        compose = (BASE / "docker-compose.lab.yml").read_text(encoding="utf-8")

        self.assertIn("internal: true", compose)
        self.assertIn("subnet: 172.30.50.0/24", compose)
        self.assertIn("172.30.50.10", compose)
        self.assertIn("172.30.50.20", compose)
        self.assertIn("@sha256:", compose)
        self.assertNotIn("ports:", compose)
        self.assertNotIn(":latest", compose)


if __name__ == "__main__":
    unittest.main()
