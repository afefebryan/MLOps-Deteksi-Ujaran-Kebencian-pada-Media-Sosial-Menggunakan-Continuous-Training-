"""
load_test.py
Simulasi beban kerja ke 3 replica API untuk mengisi dashboard Grafana.
Jalankan: python load_test.py
"""

import requests
import random
import time
import threading

REPLICAS = [
    "http://localhost:5001",
    "http://localhost:5002",
    "http://localhost:5003",
]

SAMPLE_TEXTS = [
    "i hate all of you so much",
    "have a great day everyone",
    "you are the worst person alive",
    "let us work together for a better future",
    "kill yourself nobody likes you",
    "good morning, hope you have a wonderful day",
    "i will destroy everything you love",
    "thank you for your kindness",
    "nobody wants you here, get out",
    "wishing you all the best today",
]

def send_request(replica_url: str, texts: list):
    try:
        start    = time.time()
        response = requests.post(
            f"{replica_url}/predict",
            json={"texts": texts},
            timeout=5
        )
        latency  = time.time() - start
        status   = response.status_code
        print(f"  [{replica_url[-4:]}] status={status} | latency={latency:.3f}s")
    except Exception as e:
        print(f"  [{replica_url[-4:]}] error: {e}")


def worker(replica_url: str, n_requests: int, delay: float):
    for _ in range(n_requests):
        batch = random.sample(SAMPLE_TEXTS, k=random.randint(1, 4))
        send_request(replica_url, batch)
        time.sleep(delay)


if __name__ == "__main__":
    print("=" * 60)
    print("LOAD TEST — 3 Replica")
    print("=" * 60)
    print("Mengirim request ke semua replica selama 60 detik...\n")

    threads = []
    for url in REPLICAS:
        t = threading.Thread(
            target=worker,
            args=(url, 30, 1.0)   # 30 request per replica, 1 detik interval
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("\nLoad test selesai. Buka Grafana: http://localhost:3000")