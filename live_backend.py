import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


BASE = Path(__file__).resolve().parent
LIVE_DIR = BASE / "data" / "live"
# Legacy output locations remain available to older imports. Active capture
# sessions write to their own directory and never truncate these files.
LIVE_CSV = LIVE_DIR / "live_flows.csv"
OUT_CSV = LIVE_DIR / "live_predictions.csv"
LAB_PCAP = LIVE_DIR / "lab_capture.pcap"
LAB_CSV = LIVE_DIR / "lab_flows.csv"

SCALER_FILE = BASE / "data" / "final" / "features" / "standard_scaler.joblib"
BINARY_FILE = BASE / "models" / "transformer_binary_gpu.pth"
MULTI_FILE = BASE / "models" / "transformer_multiclass_gpu_v2.pth"

CICFLOWMETER = Path(
    os.environ.get(
        "CICFLOWMETER_BIN",
        (
            r"C:\Users\piyus\anaconda3\envs\AI-NIDS\Scripts\cicflowmeter.exe"
            if os.name == "nt"
            else "cicflowmeter"
        ),
    )
)

LAB_NETWORK = "ai-nids-lab-isolated"
LAB_CLIENT = "ai-nids-lab-client"
LAB_TARGET = "ai-nids-lab-juice-shop"
LAB_TARGET_URL = "http://juice-shop:3000"
LAB_CIDR = "172.30.50.0/24"
LAB_SUBNET_PREFIX = "172.30.50."
LAB_CLIENT_IP = "172.30.50.10"
LAB_TARGET_IP = "172.30.50.20"
LAB_CAPTURE_DIR = "/captures"

BINARY_THRESHOLD = 0.80
LAB_BACKEND_VERSION = "LAB-ISOLATED-3.0"

CLASSES = [
    "BENIGN",
    "DoS_DDoS",
    "PortScan",
    "BruteForce",
    "WebAttack",
    "Bot",
    "Infiltration",
]

FEATURE_MAP = {
    "Destination Port": "dst_port",
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "tot_fwd_pkts",
    "Total Backward Packets": "tot_bwd_pkts",
    "Total Length of Fwd Packets": "totlen_fwd_pkts",
    "Total Length of Bwd Packets": "totlen_bwd_pkts",
    "Fwd Packet Length Max": "fwd_pkt_len_max",
    "Fwd Packet Length Min": "fwd_pkt_len_min",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Max": "bwd_pkt_len_max",
    "Bwd Packet Length Min": "bwd_pkt_len_min",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Bwd Packet Length Std": "bwd_pkt_len_std",
    "Flow Bytes/s": "flow_byts_s",
    "Flow Packets/s": "flow_pkts_s",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Total": "fwd_iat_tot",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Total": "bwd_iat_tot",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_psh_flags",
    "Fwd Header Length": "fwd_header_len",
    "Bwd Header Length": "bwd_header_len",
    "Fwd Packets/s": "fwd_pkts_s",
    "Bwd Packets/s": "bwd_pkts_s",
    "MinPacket Length": "pkt_len_min",
    "Min Packet Length": "pkt_len_min",
    "Max Packet Length": "pkt_len_max",
    "Packet Length Mean": "pkt_len_mean",
    "Packet Length Std": "pkt_len_std",
    "Packet Length Variance": "pkt_len_var",
    "FIN Flag Count": "fin_flag_cnt",
    "SYN Flag Count": "syn_flag_cnt",
    "RST Flag Count": "rst_flag_cnt",
    "PSH Flag Count": "psh_flag_cnt",
    "ACK Flag Count": "ack_flag_cnt",
    "URG Flag Count": "urg_flag_cnt",
    "ECE Flag Count": "ece_flag_cnt",
    "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "pkt_size_avg",
    "Init_Win_bytes_forward": "init_fwd_win_byts",
    "Init_Win_bytes_backward": "init_bwd_win_byts",
    "act_data_pkt_fwd": "fwd_act_data_pkts",
    "min_seg_size_forward": "fwd_seg_size_min",
    "Active Mean": "active_mean",
    "Active Std": "active_std",
    "Active Max": "active_max",
    "Active Min": "active_min",
    "Idle Mean": "idle_mean",
    "Idle Std": "idle_std",
    "Idle Max": "idle_max",
    "Idle Min": "idle_min",
}


class NetworkTransformer(nn.Module):
    def __init__(
        self,
        num_classes,
        num_features=61,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.feature_embedding = nn.Linear(1, d_model)
        self.feature_position = nn.Parameter(
            torch.randn(1, num_features, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.feature_embedding(x.unsqueeze(-1))
        x = x + self.feature_position
        x = self.transformer(x)
        x = self.norm(x.mean(dim=1))
        return self.classifier(x)


def load_model(path, classes, device):
    model = NetworkTransformer(classes)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


class LiveNIDS:
    """
    AI-NIDS live backend.

    Modes:
      REAL: captures the selected Windows interface with CICFlowMeter.
      LAB: captures packets inside the isolated Docker nids-client
           namespace, then converts the PCAP with CICFlowMeter.

    The AI/scaler/prediction pipeline is identical in both modes.
    """

    def __init__(self):
        self.capture_process = None
        self.capture_mode = "REAL"
        self.capture_interface = ""
        self.session_id = None
        self.session_dir = None
        self.live_csv = LIVE_CSV
        self.out_csv = OUT_CSV
        self.lab_capture_process = None
        self.lab_converter_process = None
        self.lab_capture_files = set()
        self.lab_remote_pcaps = {}
        self.lab_processed_pcaps = set()
        self.lab_action_process = None
        self.lab_action_name = ""
        self.lab_chunk_counter = 0

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.scaler = None
        self.binary_model = None
        self.multi_model = None

        self.processed_rows = 0
        self.predictions = []
        self.last_error = ""
        self.last_lab_message = ""

        self.total_flows = 0
        self.total_benign = 0
        self.total_attacks = 0
        self.high_severity = 0

    @property
    def is_running(self):
        processes = [self.capture_process, self.lab_capture_process]
        return any(p is not None and p.poll() is None for p in processes)

    def load_models(self):
        if self.scaler is None:
            if not SCALER_FILE.exists():
                raise FileNotFoundError(f"StandardScaler not found: {SCALER_FILE}")
            self.scaler = joblib.load(SCALER_FILE)

        if self.binary_model is None:
            if not BINARY_FILE.exists():
                raise FileNotFoundError(f"Binary model not found: {BINARY_FILE}")
            self.binary_model = load_model(BINARY_FILE, 2, self.device)

        if self.multi_model is None:
            if not MULTI_FILE.exists():
                raise FileNotFoundError(f"Multi-class model not found: {MULTI_FILE}")
            self.multi_model = load_model(MULTI_FILE, 7, self.device)

    def reset_session(self):
        """Start a fresh in-memory session with unique, timestamped outputs.

        Existing live CSVs and capture artifacts are intentionally left alone.
        Creating the session directory is exclusive: an unexpected collision
        or filesystem error is reported instead of reusing or truncating data.
        """
        try:
            LIVE_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(
                f"Could not prepare live capture directory '{LIVE_DIR}': {exc}"
            ) from exc

        session_id = (
            datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            + "-"
            + uuid4().hex[:8]
        )
        session_dir = LIVE_DIR / "sessions" / session_id
        try:
            session_dir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise OSError(
                f"Could not create capture session directory '{session_dir}': {exc}"
            ) from exc

        self.session_id = session_id
        self.session_dir = session_dir
        self.live_csv = session_dir / f"live_flows_{session_id}.csv"
        self.out_csv = session_dir / f"live_predictions_{session_id}.csv"

        self._reset_session_state()

    def _reset_session_state(self):
        self.processed_rows = 0
        self.predictions = []
        self.total_flows = 0
        self.total_benign = 0
        self.total_attacks = 0
        self.high_severity = 0
        self.last_error = ""
        self.last_lab_message = ""
        self.lab_capture_files = set()
        self.lab_remote_pcaps = {}
        self.lab_processed_pcaps = set()

    def _inspect_docker(self, args, description):
        try:
            result = subprocess.run(
                ["docker", *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Docker CLI was not found in PATH. Start Docker Desktop first."
            ) from exc
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Could not inspect {description}: {exc}") from exc

        if result.returncode != 0:
            detail = result.stderr.strip() or "docker inspect failed"
            raise RuntimeError(f"Could not inspect {description}: {detail}")

        try:
            data = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(
                f"Docker returned invalid inspection data for {description}."
            ) from exc

        if (
            not isinstance(data, list)
            or len(data) != 1
            or not isinstance(data[0], dict)
        ):
            raise RuntimeError(
                f"Docker returned an unexpected inspection result for {description}."
            )
        return data[0]

    def _check_lab(self):
        """Fail closed unless the lab has the expected isolated topology."""
        network = self._inspect_docker(
            ["network", "inspect", LAB_NETWORK], f"Docker network '{LAB_NETWORK}'"
        )
        if network.get("Driver") != "bridge" or network.get("Internal") is not True:
            raise RuntimeError(
                f"Docker lab network '{LAB_NETWORK}' must be an internal bridge. "
                "Start the lab from docker-compose.lab.yml."
            )

        ipam = network.get("IPAM") or {}
        network_config = ipam.get("Config") or []
        subnets = {
            item.get("Subnet")
            for item in network_config
            if isinstance(item, dict)
        }
        if LAB_CIDR not in subnets:
            raise RuntimeError(
                f"Docker lab network '{LAB_NETWORK}' must use subnet {LAB_CIDR}."
            )

        expected_ips = {
            LAB_CLIENT: LAB_CLIENT_IP,
            LAB_TARGET: LAB_TARGET_IP,
        }
        attached = network.get("Containers") or {}
        actual_attached = {}
        if isinstance(attached, dict):
            for details in attached.values():
                if isinstance(details, dict):
                    name = details.get("Name")
                    address = str(details.get("IPv4Address", "")).split("/", 1)[0]
                    if name:
                        actual_attached[name] = address
        if actual_attached != expected_ips:
            raise RuntimeError(
                "Docker lab network membership or addresses do not match the "
                f"expected client/target pair: {expected_ips}."
            )

        client_config = {}
        for name, expected_ip in expected_ips.items():
            container = self._inspect_docker(
                ["inspect", name], f"Docker container '{name}'"
            )
            if not (container.get("State") or {}).get("Running"):
                raise RuntimeError(f"Docker container '{name}' is not running.")

            networks = (container.get("NetworkSettings") or {}).get("Networks") or {}
            if set(networks) != {LAB_NETWORK}:
                raise RuntimeError(
                    f"Docker container '{name}' must be attached only to "
                    f"'{LAB_NETWORK}', not {sorted(networks)}."
                )
            if (networks.get(LAB_NETWORK) or {}).get("IPAddress") != expected_ip:
                raise RuntimeError(
                    f"Docker container '{name}' must use {expected_ip} on "
                    f"'{LAB_NETWORK}'."
                )

            host_config = container.get("HostConfig") or {}
            port_bindings = host_config.get("PortBindings") or {}
            if any(port_bindings.values()):
                raise RuntimeError(
                    f"Docker container '{name}' must not publish host ports."
                )
            if host_config.get("Privileged") is True:
                raise RuntimeError(
                    f"Docker container '{name}' must not run in privileged mode."
                )
            if "ALL" not in (host_config.get("CapDrop") or []):
                raise RuntimeError(
                    f"Docker container '{name}' must drop all Linux capabilities."
                )

            if name == LAB_CLIENT:
                client_config = host_config

        client_capabilities = {
            str(capability).upper().removeprefix("CAP_")
            for capability in (client_config.get("CapAdd") or [])
        }
        if "NET_RAW" not in client_capabilities:
            raise RuntimeError(
                f"Docker container '{LAB_CLIENT}' must have the NET_RAW "
                "capability for packet capture."
            )

    def _check_no_existing_lab_capture(self):
        """Avoid starting a second remote capture after a backend restart."""
        try:
            capture = subprocess.run(
                ["docker", "exec", LAB_CLIENT, "pidof", "tcpdump"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"Could not check for an existing tcpdump process: {exc}"
            ) from exc
        if capture.returncode == 0:
            raise RuntimeError(
                f"A tcpdump process is already running in '{LAB_CLIENT}'. "
                "Stop the prior capture before starting another session."
            )
        if capture.returncode != 1:
            raise RuntimeError(
                "Could not verify that the isolated lab client has no active capture: "
                f"{capture.stderr.strip() or 'pidof tcpdump failed'}"
            )

    def _install_tcpdump_if_needed(self):
        # Capture tooling is baked into the pinned client image. Runtime
        # package installation would make the lab mutable and non-reproducible.
        try:
            check = subprocess.run(
                ["docker", "exec", LAB_CLIENT, "sh", "-c", "command -v tcpdump"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"Could not verify tcpdump in the lab client: {exc}"
            ) from exc
        if check.returncode != 0:
            raise RuntimeError(
                "tcpdump is missing from the pinned lab-client image. Rebuild it "
                "with `docker compose -f docker-compose.lab.yml build nids-client`.\n"
                + (check.stderr[-1500:] or "No tcpdump executable was found.")
            )

    def start_capture(self, interface):
        """Backward-compatible REAL mode entry point."""
        self.start_real_capture(interface)

    def start_real_capture(self, interface):
        if self.is_running:
            return

        if not CICFLOWMETER.exists():
            raise FileNotFoundError(
                f"CICFlowMeter executable not found: {CICFLOWMETER}"
            )

        self.load_models()
        self.reset_session()

        self.capture_mode = "REAL"
        self.capture_interface = interface

        command = [
            str(CICFLOWMETER),
            "-i",
            interface,
            "-c",
            str(self.live_csv),
        ]

        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        try:
            self.capture_process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=flags,
                text=True,
            )
        except Exception as exc:
            self.capture_process = None
            raise RuntimeError(f"Could not start CICFlowMeter: {exc}")

        time.sleep(1.5)

        if self.capture_process.poll() is not None:
            error_text = ""
            try:
                error_text = self.capture_process.stderr.read()
            except Exception:
                pass
            self.capture_process = None
            raise RuntimeError(
                "CICFlowMeter stopped immediately.\n"
                + (error_text.strip() or "Check Npcap, adapter name and permissions.")
            )

    def start_lab_capture(self):
        """
        LAB capture strategy:
        tcpdump runs inside nids-client, so Docker Desktop's internal
        bridge does not have to be visible to Windows/Npcap.
        The PCAP is streamed to the host through docker exec.
        """
        if self.is_running:
            return

        # Complete all LAB checks before changing session state or creating
        # session output paths. A failed preflight must leave the current
        # session and all existing files untouched.
        self._check_lab()
        self._check_no_existing_lab_capture()

        if not CICFLOWMETER.exists():
            raise FileNotFoundError(
                f"CICFlowMeter executable not found: {CICFLOWMETER}"
            )

        self._install_tcpdump_if_needed()
        self.load_models()
        self.reset_session()

        self.capture_mode = "LAB"
        self.capture_interface = "Docker Lab / nids-client"
        self.last_lab_message = "Starting isolated Docker lab capture..."

        # Capture inside the isolated lab client and rotate PCAPs every 5 seconds.
        # The session prefix prevents this run from reusing or deleting old PCAPs.
        remote_prefix = f"ai-nids-lab-{self.session_id}"
        command = [
            "docker", "exec", LAB_CLIENT,
            "tcpdump",
            "-i", "eth0",
            "-U",
            "-s", "0",
            "-G", "5",
            "-w", f"{LAB_CAPTURE_DIR}/{remote_prefix}-%Y%m%d-%H%M%S.pcap",
            "net", LAB_CIDR,
        ]

        try:
            self.lab_capture_process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:
            self.lab_capture_process = None
            raise RuntimeError(f"Could not start Docker lab capture: {exc}")

        time.sleep(1.5)

        if self.lab_capture_process.poll() is not None:
            error_text = ""
            try:
                error_text = self.lab_capture_process.stderr.read().decode(
                    errors="replace"
                )
            except Exception:
                pass
            self.stop_capture()
            raise RuntimeError(
                "Docker lab packet capture stopped.\n"
                + (error_text.strip() or "Check tcpdump inside nids-client.")
            )

        self.last_lab_message = (
            "Docker lab capture is active. Waiting for completed PCAP chunks..."
        )

    def _copy_completed_lab_pcaps(self, include_latest=False):
        """Stage finished PCAPs locally without removing their remote copies.

        While capture is active the newest file is still open and is skipped.
        Finalization calls this with ``include_latest=True`` after tcpdump exits.
        """
        if not self.session_id or self.session_dir is None:
            self.last_error = "Cannot copy LAB PCAPs before a capture session starts."
            return []

        remote_pattern = f"{LAB_CAPTURE_DIR}/ai-nids-lab-{self.session_id}-*.pcap"
        try:
            result = subprocess.run(
                [
                    "docker", "exec", LAB_CLIENT,
                    "find", LAB_CAPTURE_DIR,
                    "-maxdepth", "1",
                    "-type", "f",
                    "-name", Path(remote_pattern).name,
                    "-print",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception as exc:
            self.last_error = f"Could not list LAB PCAP files: {exc}"
            return []

        if result.returncode != 0:
            self.last_error = (
                "Could not list LAB PCAP files: "
                f"{result.stderr.strip() or 'docker exec failed'}"
            )
            return []

        remote_files = sorted(
            (x.strip() for x in result.stdout.splitlines() if x.strip()),
            key=lambda item: Path(item).name,
            reverse=True,
        )
        candidates = remote_files if include_latest else remote_files[1:]
        copied = []

        for remote in candidates:
            if remote in self.lab_capture_files:
                continue

            local = self.session_dir / Path(remote).name
            try:
                if local.exists():
                    if local.stat().st_size < 24:
                        self.last_lab_message = (
                            f"Retaining incomplete PCAP for retry: {local.name}"
                        )
                        continue
                else:
                    # Keep failed/partial docker-cp results under unique names;
                    # the next retry never overwrites diagnostic artifacts.
                    staging = local.with_name(
                        f"{local.name}.{uuid4().hex}.partial"
                    )
                    cp = subprocess.run(
                        ["docker", "cp", f"{LAB_CLIENT}:{remote}", str(staging)],
                        capture_output=True,
                        text=True,
                        timeout=20,
                    )
                    if cp.returncode != 0:
                        self.last_error = (
                            f"Could not copy LAB PCAP {Path(remote).name}: "
                            f"{cp.stderr.strip() or 'docker cp failed'}"
                        )
                        continue
                    if not staging.exists():
                        self.last_error = (
                            f"docker cp reported success but did not create {staging}"
                        )
                        continue
                    if staging.stat().st_size < 24:
                        self.last_lab_message = (
                            f"Copied PCAP is incomplete and was retained: {staging.name}"
                        )
                        continue
                    staging.rename(local)

                self.lab_capture_files.add(remote)
                self.lab_remote_pcaps[local.name] = remote
                copied.append(local)
            except Exception as exc:
                self.last_error = (
                    f"Could not copy LAB PCAP {Path(remote).name}: {exc}"
                )

        return copied

    def _remove_completed_remote_pcap(self, pcap_path):
        remote = self.lab_remote_pcaps.get(pcap_path.name)
        if not remote:
            return True

        try:
            result = subprocess.run(
                ["docker", "exec", LAB_CLIENT, "rm", "-f", remote],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception as exc:
            self.last_error = f"Could not remove processed remote PCAP {remote}: {exc}"
            return False

        if result.returncode != 0:
            self.last_error = (
                f"Could not remove processed remote PCAP {remote}: "
                f"{result.stderr.strip() or 'docker exec rm failed'}"
            )
            return False
        return True

    def _append_dataframe_atomically(self, frame, path):
        """Append rows through a same-directory temporary file and atomic replace."""
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        existing = path.exists()
        try:
            with temporary.open("wb") as output:
                if existing:
                    with path.open("rb") as current:
                        shutil.copyfileobj(current, output)
                        current.seek(0, os.SEEK_END)
                        if current.tell():
                            current.seek(-1, os.SEEK_END)
                            if current.read(1) != b"\n":
                                output.write(b"\n")
                frame.to_csv(output, index=False, header=not existing)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        except Exception:
            # Leave the unique temporary artifact for inspection/recovery.
            raise

    def _convert_lab_pcap(self, pcap_path):
        if pcap_path.name in self.lab_processed_pcaps:
            return True
        if not pcap_path.exists():
            self.last_error = f"LAB PCAP does not exist: {pcap_path}"
            return False

        # A PCAP global header is 24 bytes. Header-only captures are valid and
        # should be converted as empty flows rather than treated as corrupt.
        if pcap_path.stat().st_size < 24:
            self.last_lab_message = f"Retaining incomplete PCAP for retry: {pcap_path.name}"
            return False

        if self.session_dir is None or self.session_id is None:
            self.last_error = "Cannot convert a LAB PCAP without a session output path."
            return False

        # Each attempt gets a unique output path so a timeout or partial CSV is
        # preserved and cannot prevent a later conversion retry.
        tmp_csv = self.session_dir / f"{pcap_path.stem}_flows_{uuid4().hex}.csv"
        command = [
            str(CICFLOWMETER),
            "-f",
            str(pcap_path),
            "-c",
            str(tmp_csv),
        ]

        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=flags,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            self.last_error = f"CICFlowMeter timed out for {pcap_path.name}"
            return False
        except Exception as exc:
            self.last_error = f"LAB CICFlowMeter failed: {exc}"
            return False

        if result.returncode != 0:
            self.last_error = (
                result.stderr.strip()[-1200:]
                if result.stderr
                else f"CICFlowMeter failed for {pcap_path.name}"
            )
            return False

        if not tmp_csv.exists():
            self.last_error = (
                f"CICFlowMeter completed without creating a flow file for "
                f"{pcap_path.name}"
            )
            return False

        try:
            if tmp_csv.stat().st_size == 0:
                # CICFlowMeter can produce an empty file for a valid PCAP with
                # no classifiable flows. Preserve it and avoid retrying forever.
                self.lab_processed_pcaps.add(pcap_path.name)
                self._remove_completed_remote_pcap(pcap_path)
                self.last_lab_message = (
                    f"Converted PCAP contains no usable flow rows: {pcap_path.name}"
                )
                return True
        except OSError as exc:
            self.last_error = f"Could not inspect converted flow file {tmp_csv}: {exc}"
            return False

        try:
            flows = pd.read_csv(tmp_csv)
        except Exception as exc:
            self.last_error = f"Could not read converted flow file {tmp_csv}: {exc}"
            return False

        if flows.empty:
            # This is a completed empty capture, not a failed conversion. Keep
            # both artifacts and remember the completion so it is not repeated.
            self.lab_processed_pcaps.add(pcap_path.name)
            self._remove_completed_remote_pcap(pcap_path)
            self.last_lab_message = (
                f"Converted PCAP contains no usable flow rows: {pcap_path.name}"
            )
            return True

        try:
            self._append_dataframe_atomically(flows, self.live_csv)
        except Exception as exc:
            self.last_error = f"Could not append flows from {pcap_path.name}: {exc}"
            return False

        # Set completion before best-effort archival or remote cleanup, so a
        # later poll cannot append the same PCAP's flows a second time.
        self.lab_processed_pcaps.add(pcap_path.name)
        self._remove_completed_remote_pcap(pcap_path)
        self.last_lab_message = f"Processed {pcap_path.name}"

        try:
            debug_dir = BASE / "debug_pcaps"
            debug_dir.mkdir(parents=True, exist_ok=True)

            debug_pcap = debug_dir / pcap_path.name
            debug_csv = debug_dir / f"{pcap_path.stem}_flows.csv"

            if debug_pcap.exists() or debug_csv.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing debug capture for {pcap_path.name}"
                )

            pcap_path.rename(debug_pcap)
            tmp_csv.rename(debug_csv)
            self.last_lab_message = f"Processed {pcap_path.name} → saved in debug_pcaps"
        except Exception as exc:
            self.last_lab_message = (
                f"Processed {pcap_path.name}, but debug save failed: {exc}"
            )

        return True

    def _process_lab_chunks(self, finalize=False):
        self.last_error = ""
        self._copy_completed_lab_pcaps(include_latest=finalize)

        if self.session_dir is not None and self.session_id is not None:
            pattern = f"ai-nids-lab-{self.session_id}-*.pcap"
            for pcap in sorted(self.session_dir.glob(pattern)):
                if pcap.name not in self.lab_processed_pcaps:
                    self._convert_lab_pcap(pcap)

        return not self.last_error

    def _stop_lab_tcpdump(self):
        """Signal tcpdump and wait until its capture file is closed."""
        try:
            result = subprocess.run(
                ["docker", "exec", LAB_CLIENT, "pkill", "-INT", "-x", "tcpdump"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode not in (0, 1):
                raise RuntimeError(
                    result.stderr.strip() or "pkill -INT tcpdump failed"
                )

            deadline = time.monotonic() + 10
            while True:
                check = subprocess.run(
                    ["docker", "exec", LAB_CLIENT, "pidof", "tcpdump"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if check.returncode == 1:
                    break
                if check.returncode != 0:
                    raise RuntimeError(
                        check.stderr.strip() or "Could not verify tcpdump shutdown"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError("tcpdump did not exit within 10 seconds")
                time.sleep(0.2)

            process = self.lab_capture_process
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # tcpdump is confirmed stopped; this only releases the
                    # attached docker exec client process.
                    process.terminate()
                    process.wait(timeout=4)
        except Exception as exc:
            self.last_error = f"Could not stop tcpdump in the isolated lab client: {exc}"
            return False

        return True

    def stop_capture(self):
        self.stop_lab_action()

        if self.capture_mode == "LAB" and self.lab_capture_process is not None:
            if not self._stop_lab_tcpdump():
                # Keep the process/session state intact so the dashboard does
                # not report a clean stop while tcpdump may still be running.
                return
            self.lab_capture_process = None
            self.process_new_flows(force=True, finalize=True)

        for attr in ("capture_process", "lab_capture_process", "lab_converter_process"):
            p = getattr(self, attr, None)
            setattr(self, attr, None)

            if p is None:
                continue

            try:
                if p.poll() is None:
                    p.terminate()
                    p.wait(timeout=4)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

    def make_features(self, df):
        expected = list(self.scaler.feature_names_in_)
        resolved = []
        missing = []

        for feature in expected:
            live_column = FEATURE_MAP.get(feature)

            if live_column is None:
                missing.append(f"No mapping for: {feature}")
                continue

            if live_column not in df.columns:
                missing.append(f"{feature} -> {live_column}")
                continue

            resolved.append(live_column)

        if missing:
            raise RuntimeError(
                "Missing/unmapped CICFlowMeter columns:\n"
                + "\n".join(missing)
            )

        x = df[resolved].copy()
        x = x.apply(pd.to_numeric, errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.fillna(0)

        x = self.scaler.transform(
            x.to_numpy(dtype=np.float64)
        )

        return x.astype(np.float32)

    def predict(self, x):
        if len(x) == 0:
            return []

        tensor = torch.from_numpy(x).to(self.device)

        with torch.inference_mode():
            bp = torch.softmax(self.binary_model(tensor), dim=1)
            mp = torch.softmax(self.multi_model(tensor), dim=1)

        out = []

        for i in range(len(x)):
            attack = float(bp[i, 1])
            idx = int(torch.argmax(mp[i]))
            family = CLASSES[idx]
            conf = float(mp[i, idx])

            if attack >= BINARY_THRESHOLD:
                pred = "ATTACK"

                if family == "BENIGN":
                    for j in torch.argsort(mp[i], descending=True):
                        j = int(j)
                        if CLASSES[j] != "BENIGN":
                            family = CLASSES[j]
                            conf = float(mp[i, j])
                            break
            else:
                pred = "BENIGN"
                family = "BENIGN"

            severity = (
                "NONE"
                if pred == "BENIGN"
                else (
                    "HIGH"
                    if family in ["DoS_DDoS", "Infiltration"]
                    else "MEDIUM"
                )
            )

            out.append((pred, attack, family, conf, severity))

        return out

    def process_new_flows(self, force=False, finalize=False):
        if not force and not self.is_running:
            return

        lab_chunks_ok = True
        if self.capture_mode == "LAB":
            try:
                lab_chunks_ok = self._process_lab_chunks(finalize=finalize)
            except OSError as exc:
                self.last_error = f"LAB capture file error: {exc}"
                return

        source = self.live_csv

        if not source.exists():
            return

        try:
            df = pd.read_csv(
                source,
                on_bad_lines="skip",
            )
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            self.last_error = f"Could not read live flow file {source}: {exc}"
            return

        if len(df) <= self.processed_rows:
            if lab_chunks_ok:
                self.last_error = ""
            return

        new = df.iloc[self.processed_rows:].copy()

        # Hard isolation guard: LAB mode may only expose Docker-lab flows.
        # This prevents stale REAL/Wi-Fi rows from leaking into the LAB UI.
        if self.capture_mode == "LAB":
            if "src_ip" in new.columns and "dst_ip" in new.columns:
                src = new["src_ip"].astype(str)
                dst = new["dst_ip"].astype(str)
                lab_mask = (
                    src.str.startswith(LAB_SUBNET_PREFIX)
                    & dst.str.startswith(LAB_SUBNET_PREFIX)
                )
                new = new.loc[lab_mask].copy()

            if new.empty:
                # These rows were deliberately filtered; they need no model
                # inference and can be committed to the cursor immediately.
                self.processed_rows = len(df)
                if lab_chunks_ok:
                    self.last_error = ""
                return

        try:
            results = list(self.predict(self.make_features(new)))
            if len(results) != len(new):
                raise RuntimeError(
                    "Inference returned "
                    f"{len(results)} predictions for {len(new)} flow rows."
                )
        except Exception as exc:
            self.last_error = repr(exc)
            return

        rows = []

        for index, result in enumerate(results):
            raw = new.iloc[index].to_dict()
            pred, attack, family, conf, severity = result

            item = {
                "timestamp": raw.get("timestamp", ""),
                "src_ip": raw.get("src_ip", ""),
                "src_port": raw.get("src_port", ""),
                "dst_ip": raw.get("dst_ip", ""),
                "dst_port": raw.get("dst_port", ""),
                "protocol": raw.get("protocol", ""),
                "prediction": pred,
                "attack_probability": attack,
                "attack_family": family,
                "family_confidence": conf,
                "severity": severity,
            }

            rows.append(item)

        if rows:
            try:
                self._append_dataframe_atomically(pd.DataFrame(rows), self.out_csv)
            except Exception as exc:
                self.last_error = f"Could not write predictions to {self.out_csv}: {exc}"
                return

        # Commit in-memory rows, counters and the source cursor only after the
        # complete prediction batch has been persisted successfully.
        self.predictions.extend(rows)
        self.total_flows += len(rows)
        self.total_attacks += sum(row["prediction"] == "ATTACK" for row in rows)
        self.total_benign += sum(row["prediction"] == "BENIGN" for row in rows)
        self.high_severity += sum(row["severity"] == "HIGH" for row in rows)
        self.processed_rows = len(df)
        self.predictions = self.predictions[-300:]

        if lab_chunks_ok:
            self.last_error = ""

    def run_lab_action(self, action):
        """
        Start a browser-triggered lab traffic scenario asynchronously.
        All scenarios target only Juice Shop inside ai-nids-lab.
        """
        self._check_lab()

        if hasattr(self, "lab_action_process") and self.lab_action_process is not None:
            if self.lab_action_process.poll() is None:
                raise RuntimeError(
                    f"Lab action '{getattr(self, 'lab_action_name', 'current')}' is still running."
                )

        commands = {
            "normal": (
                "end=$(( $(date +%s) + 20 )); "
                "while [ $(date +%s) -lt $end ]; do "
                f"wget -qO- {LAB_TARGET_URL}/ >/dev/null 2>&1; "
                f"wget -qO- '{LAB_TARGET_URL}/rest/products/search?q=apple' >/dev/null 2>&1; "
                "sleep 2; done"
            ),
            "portscan": (
                "for p in 22 80 443 3000 3001 5000 8000 8080 9000; do "
                f"nc -z -w 1 {LAB_TARGET_IP} $p >/dev/null 2>&1 || true; "
                "done"
            ),
            "webtest": (
                "for q in \"'\" \"<script>test</script>\" \"admin\"; do "
                f"wget -qO- --post-data=\"q=$q\" "
                f"{LAB_TARGET_URL}/rest/products/search >/dev/null 2>&1 || true; "
                "done"
            ),
            "stress": (
                "end=$(( $(date +%s) + 8 )); "
                "while [ $(date +%s) -lt $end ]; do "
                f"wget -qO- {LAB_TARGET_URL}/ >/dev/null 2>&1; "
                "done"
            ),
        }

        if action not in commands:
            raise ValueError(f"Unknown lab action: {action}")

        self.lab_action_name = action
        self.lab_action_process = subprocess.Popen(
            [
                "docker", "exec", LAB_CLIENT,
                "sh", "-c", commands[action],
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        return action

    @property
    def lab_action_running(self):
      p = getattr(self, "lab_action_process", None)

      if p is None:
       return False

      if p.poll() is None:
        return True

     # Process has finished — clean up the state.
        self.lab_action_process = None
        self.lab_action_name = ""

        return False

    def stop_lab_action(self):
        p = getattr(self, "lab_action_process", None)
        if p is None:
            return

        try:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=3)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

        self.lab_action_process = None
        self.lab_action_name = ""

    # ---------------- LAB traffic controls ----------------

    def _lab_exec(self, shell_command, timeout=30):
        result = subprocess.run(
            ["docker", "exec", LAB_CLIENT, "sh", "-c", shell_command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Lab command failed:\n{result.stderr[-1500:]}"
            )

        return result.stdout

    def lab_normal_traffic(self, seconds=20):
        self._check_lab()
        # Low-rate legitimate browsing pattern.
        cmd = (
            f"end=$(( $(date +%s) + {int(seconds)} )); "
            f"while [ $(date +%s) -lt $end ]; do "
            f"wget -qO- {LAB_TARGET_URL}/ >/dev/null 2>&1; "
            f"wget -qO- {LAB_TARGET_URL}/rest/products/search?q=apple >/dev/null 2>&1; "
            f"sleep 2; "
            f"done"
        )
        return self._lab_exec(cmd, timeout=int(seconds) + 15)

    def lab_port_scan_test(self):
        self._check_lab()
        # Controlled localhost-lab scan: only the Juice Shop container IP
        # and a small fixed list of ports are tested.
        target = LAB_TARGET_IP
        ports = "22 80 443 3000 3001 5000 8000 8080 9000"
        cmd = (
            f"for p in {ports}; do "
            f"nc -z -w 1 {target} $p >/dev/null 2>&1 || true; "
            f"done"
        )
        return self._lab_exec(cmd, timeout=20)

    def lab_web_test(self):
        self._check_lab()
        # Harmless security-testing strings sent only to the local OWASP
        # Juice Shop search endpoint. This is not a remote target.
        cmd = (
            f"for q in \"'\" \"<script>test</script>\" \"admin\"; do "
            f"wget -qO- --post-data=\"q=$q\" "
            f"{LAB_TARGET_URL}/rest/products/search >/dev/null 2>&1 || true; "
            f"done"
        )
        return self._lab_exec(cmd, timeout=20)

    def lab_stress_test(self, seconds=8):
        """
        Small, bounded local stress test for demonstration.
        It is deliberately rate-limited and targets only Juice Shop
        inside ai-nids-lab.
        """
        self._check_lab()
        seconds = max(3, min(int(seconds), 10))
        cmd = (
            f"end=$(( $(date +%s) + {seconds} )); "
            f"while [ $(date +%s) -lt $end ]; do "
            f"wget -qO- {LAB_TARGET_URL}/ >/dev/null 2>&1; "
            f"done"
        )
        return self._lab_exec(cmd, timeout=seconds + 15)

    def reset_all(self):
        self.stop_capture()
        if self.capture_process is not None or self.lab_capture_process is not None:
            # A failed LAB shutdown must not discard the active session cursor
            # or counters while tcpdump may still be writing to its PCAP.
            return
        # Reset only in-memory counters. Capture files are retained, and the
        # next start_* call will allocate a new output session.
        self._reset_session_state()
        self.capture_interface = ""
        self.capture_mode = "REAL"

    def shutdown(self):
        self.stop_capture()
