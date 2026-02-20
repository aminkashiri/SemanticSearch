import time
import socket
import struct
import threading
import io
import json
from typing import Dict, Optional

import numpy as np
import torch
import zmq
import logging

from .multiagent_goat_agent import BaseMultiAgentGoatAgent

class CommunicationLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        # modify the message however you want
        return f"[COMM] {msg}", kwargs

class RealWorldGoatAgent(BaseMultiAgentGoatAgent):
    """
    Real-world multi-agent GOAT agent.
    Extends BaseMultiAgentGoat with UDP discovery + ZMQ communication.

    - Sender thread = discover neighbors + pairwise communicate
    - Receiver thread = accept incoming data + merge
    """

    BEACON_MAGIC = b"MAGT"

    def __init__(
        self,
        config,
        vocabulary,
        agent_id=None,
        device_id: int = 0,
        broadcast_port: int = 9900,
        data_port: int = 9901,
        beacon_interval: float = 1.0,
    ):
        super().__init__(config, vocabulary, agent_id, device_id)

        self.broadcast_port = broadcast_port
        self.data_port = data_port + agent_id
        self.beacon_interval = beacon_interval

        self.neighbors: Dict[int, dict] = {}
        self._state_lock = threading.Lock()
        self._comm_running = False
        self.comm_log = CommunicationLogger(self._log, {})


    def reset(self, scene_id, episode_id):
        self.stop_communication()
        super().reset(scene_id, episode_id)
        self.start_communication()

    def start_communication(self):
        if self._comm_running:
            return
        self._comm_running = True
        self._sender_thread = threading.Thread(
            target=self._sender_loop, name=f"Agent{self.agent_id}-Sender", daemon=True
        )
        self._receiver_thread = threading.Thread(
            target=self._receiver_loop, name=f"Agent{self.agent_id}-Receiver", daemon=True
        )
        self._sender_thread.start()
        self._receiver_thread.start()
        self.comm_log.info(f"Agent {self.agent_id} comm started on port {self.data_port}")

    def stop_communication(self):
        self._comm_running = False

    def act(self):
        with self._state_lock:
            return super().act()


    def _sender_loop(self):
        ctx = zmq.Context()

        tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        rx_sock.bind(("", self.broadcast_port))
        rx_sock.settimeout(0.2)

        last_beacon = 0

        while self._comm_running:
            now = time.time()

            # --- Discovery ---
            if now - last_beacon >= self.beacon_interval:
                self.comm_log.debug(f"Broadcasting beacon on port {self.broadcast_port}")
                msg = self.BEACON_MAGIC + struct.pack(
                    "IH", self.agent_id, self.data_port
                )
                try:
                    tx_sock.sendto(msg, ("<broadcast>", self.broadcast_port))
                except OSError:
                    pass
                last_beacon = now

            try:
                self.comm_log.debug(f"Listening for beacon on port {self.broadcast_port}")
                data, addr = rx_sock.recvfrom(64)
                if len(data) >= 10 and data[:4] == self.BEACON_MAGIC:
                    aid, port = struct.unpack("IH", data[4:10])
                    if aid != self.agent_id:
                        self.neighbors[aid] = {"ip": addr[0], "port": port}
            except socket.timeout:
                pass

            # --- Pairwise communication ---
            for aid, info in self.neighbors.items():
                with self._state_lock:
                    last = self.last_communication_time.get(aid, -1)
                    if self.total_timesteps - last < self.communication_cooldown:
                        continue
                    packed = self._pack_comm_data()

                if self._send_to_neighbor(ctx, aid, info, packed):
                    with self._state_lock:
                        self.last_communication_time[aid] = self.total_timesteps

            time.sleep(0.1)

        tx_sock.close()
        rx_sock.close()
        ctx.term()

    def _send_to_neighbor(
        self, ctx: zmq.Context, aid: int, info: dict, packed: bytes
    ) -> bool:
        try:
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.SNDTIMEO, 5000)
            sock.setsockopt(zmq.RCVTIMEO, 10000)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(f"tcp://{info['ip']}:{info['port']}")
            sock.send(packed)
            self.comm_log.debug("Waiting for ack")
            ack = sock.recv()  # blocks until receiver confirms
            self.comm_log.debug("Received ack")
            sock.close()
            self.comm_log.debug(
                f"Agent {self.agent_id} -> Agent {aid}[{info['ip']}:{info['port']}]: "
                f"sent {len(packed)} bytes at step {self.total_timesteps}"
            )
            return True
        except Exception as e:
            self.comm_log.warning(
                f"Agent {self.agent_id} -> Agent {aid}: send failed: {e}"
            )
            return False

    def _receiver_loop(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        sock.bind(f"tcp://*:{self.data_port}")

        while self._comm_running:
            try:
                self.comm_log.debug(f"Agent {self.agent_id} waiting for data on port {self.data_port}")
                raw = sock.recv()

                # From here we MUST send a reply no matter what
                try:
                    data = self._unpack_comm_data(raw)
                    if data is None:
                        self.comm_log.warning("Received invalid data")
                        sock.send(b"ERR")
                        continue

                    self.comm_log.debug(f"Received data from Agent {data['agent_id']} at step {self.total_timesteps}")
                    with self._state_lock:
                        self.comm_log.info(
                            f"Agent {self.agent_id} <- Agent {data['agent_id']}: "
                            f"received map at step {self.total_timesteps}"
                        )
                        self._merge_communication_data(data)
                        self.last_communication_time[data["agent_id"]] = self.total_timesteps

                    sock.send(b"OK")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.comm_log.error(f"Agent {self.agent_id} merge error: {e}")
                    print(f"Agent {self.agent_id} merge error: {e}")
                    sock.send(b"ERR")

            except zmq.Again:
                continue
            except Exception as e:
                self.comm_log.error(f"Agent {self.agent_id} receiver error: {e}")

        sock.close()
        ctx.term()


    def _pack_comm_data(self) -> bytes:
        data = self._get_communication_data()
        global_map = data.pop("map")
        buf = io.BytesIO()
        buf.write(struct.pack("I", self.agent_id))

        meta_bytes = json.dumps(data).encode()
        buf.write(struct.pack("I", len(meta_bytes)))
        buf.write(meta_bytes)

        buf.write(global_map.cpu().numpy().tobytes())
        return buf.getvalue()

    def _unpack_comm_data(self, raw: bytes) -> Optional[dict]:
        try:
            buf = io.BytesIO(raw)
            agent_id = struct.unpack("I", buf.read(4))[0]
            meta_len = struct.unpack("I", buf.read(4))[0]
            meta = json.loads(buf.read(meta_len).decode())
            map_shape = self.semantic_map.global_map.shape
            map_np = np.frombuffer(buf.read(), dtype=np.float32).reshape(map_shape)
            return {
                "agent_id": agent_id,
                "map": torch.from_numpy(map_np.copy()),
                **meta,
            }
        except Exception as e:
            self.comm_log.error(f"Unpack failed: {e}")
            return None

