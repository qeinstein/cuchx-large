"""List a gated Hugging Face ZIP through HTTP range requests.

This avoids downloading multi-gigabyte multimodal archives merely to inspect their
central directory. Authentication is read from the normal Hugging Face token store and
is never printed.
"""
from __future__ import annotations

import argparse
import binascii
import os
import random
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import requests
from huggingface_hub import get_token
from remotezip import RemoteZip
from urllib3.util.retry import Retry


class TimeoutSession(requests.Session):
    """Requests session with bounded CDN reads and retryable range requests."""

    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", (15, 90))
        return super().request(*args, **kwargs)


def make_session(token: str) -> requests.Session:
    session = TimeoutSession()
    session.headers.update({"Authorization": f"Bearer {token}"})
    retry = Retry(
        total=5, connect=5, read=5, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(("GET", "HEAD")),
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry, pool_maxsize=64)
    session.mount("https://", adapter)
    return session


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--repo", default="Kevin-Pal/CUHK-X_Large_Model_Track")
    parser.add_argument("--contains")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--extract-to")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument(
        "--direct", action="store_true",
        help="fetch each ZIP member range directly instead of opening one RemoteZip per worker",
    )
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("No Hugging Face token is configured")
    url = (
        f"https://huggingface.co/datasets/{args.repo}/resolve/main/"
        + quote(args.path, safe="/")
    )
    session = make_session(token)
    with RemoteZip(url, session=session) as archive:
        members = archive.infolist()
        if args.contains:
            members = [item for item in members if args.contains.lower() in item.filename.lower()]
        files = [item for item in members if not item.is_dir()]
        if args.shuffle_seed is not None:
            random.Random(args.shuffle_seed).shuffle(files)
        if args.summary:
            print(
                f"files={len(files)} uncompressed={sum(x.file_size for x in files)} "
                f"compressed={sum(x.compress_size for x in files)}"
            )
        elif args.extract_to:
            root = os.path.abspath(args.extract_to)
            os.makedirs(root, exist_ok=True)
            local = threading.local()
            lock = threading.Lock()
            progress = {"count": 0, "bytes": 0}
            member_span = archive.fp._member_position_to_size

            def worker_archive():
                if not hasattr(local, "archive"):
                    worker_session = make_session(token)
                    local.archive = RemoteZip(url, session=worker_session)
                return local.archive

            def direct_bytes(item):
                """Fetch and decode one complete member from its local-header range."""
                if not hasattr(local, "session"):
                    local.session = make_session(token)
                start = item.header_offset
                size = member_span[start]
                response = local.session.get(
                    url, headers={"Range": f"bytes={start}-{start + size - 1}"}
                )
                response.raise_for_status()
                raw = response.content
                if len(raw) != size:
                    raise IOError(f"short member range {len(raw)} != {size}")
                if len(raw) < 30 or raw[:4] != b"PK\x03\x04":
                    raise IOError("invalid ZIP local header")
                fields = struct.unpack("<IHHHHHIIIHH", raw[:30])
                name_len, extra_len = fields[-2], fields[-1]
                lo = 30 + name_len + extra_len
                comp = raw[lo:lo + item.compress_size]
                if len(comp) != item.compress_size:
                    raise IOError("truncated compressed member")
                if item.compress_type == 0:
                    data = comp
                elif item.compress_type == 8:
                    data = zlib.decompress(comp, -15)
                else:
                    raise NotImplementedError(f"ZIP compression {item.compress_type}")
                if len(data) != item.file_size:
                    raise IOError(f"decoded size {len(data)} != {item.file_size}")
                if (binascii.crc32(data) & 0xFFFFFFFF) != item.CRC:
                    raise IOError("CRC mismatch")
                return data

            def extract(item):
                target = os.path.abspath(os.path.join(root, item.filename))
                if os.path.commonpath((root, target)) != root:
                    raise ValueError(f"unsafe archive member: {item.filename}")
                if os.path.exists(target) and os.path.getsize(target) == item.file_size:
                    size = item.file_size
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    for attempt in range(5):
                        try:
                            if args.direct:
                                data = direct_bytes(item)
                                with open(target + ".partial", "wb") as sink:
                                    sink.write(data)
                            else:
                                remote = worker_archive()
                                with remote.open(remote.getinfo(item.filename)) as source, open(
                                    target + ".partial", "wb"
                                ) as sink:
                                    while True:
                                        block = source.read(1024 * 1024)
                                        if not block:
                                            break
                                        sink.write(block)
                            os.replace(target + ".partial", target)
                            break
                        except Exception:
                            if os.path.exists(target + ".partial"):
                                os.unlink(target + ".partial")
                            if hasattr(local, "archive"):
                                try:
                                    local.archive.close()
                                except Exception:
                                    pass
                                del local.archive
                            if attempt == 4:
                                raise
                    size = item.file_size
                with lock:
                    progress["count"] += 1
                    progress["bytes"] += size
                    if progress["count"] % 25 == 0 or progress["count"] == len(files):
                        print(
                            f"{progress['count']}/{len(files)} "
                            f"{progress['bytes'] / 1e6:.1f} MB",
                            flush=True,
                        )

            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futures = [pool.submit(extract, item) for item in files]
                for future in as_completed(futures):
                    future.result()
        else:
            for item in members:
                print(f"{item.file_size}\t{item.compress_size}\t{item.filename}")


if __name__ == "__main__":
    main()
