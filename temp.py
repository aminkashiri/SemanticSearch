import socket, struct, time

MAGIC = b"MAGT"
MY_ID = 1  # change to 1 on the other robot
ADHOC_IP = "10.0.0.2"  # change per robot
PORT = 9900

tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
tx.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
rx.bind(("", PORT))
rx.settimeout(1.0)

broadcast_addr = ADHOC_IP.rsplit('.', 1)[0] + '.255'
print(broadcast_addr)

for _ in range(30):
    tx.sendto(MAGIC + struct.pack("IH", MY_ID, 9901), (broadcast_addr, PORT))
    try:
        data, addr = rx.recvfrom(64)
        if data[:4] == MAGIC:
            aid, port = struct.unpack("IH", data[4:10])
            if aid != MY_ID:
                print(f"Found robot {aid} at {addr[0]}:{port}")
    except socket.timeout:
        print("No beacon received")
    time.sleep(1)