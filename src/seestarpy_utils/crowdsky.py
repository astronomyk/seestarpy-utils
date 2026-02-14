"""
CrowdSky module — WebDAV operations for uploading, retrieving, and
deleting observation data on the UCloud share.
"""

import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import requests


def _load_config() -> dict:
    """Find and load crowdsky_config.toml by walking up from this file."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        cfg = current / "crowdsky_config.toml"
        if cfg.exists():
            with open(cfg, "rb") as f:
                return tomllib.load(f)
        if current.parent == current:
            break
        current = current.parent
    raise FileNotFoundError(
        "crowdsky_config.toml not found in any parent directory"
    )


_config = _load_config()
_WEBDAV_URL = _config["ucloud"]["webdav_url"]
_SHARE_TOKEN = _config["ucloud"]["share_token"]
_AUTH = (_SHARE_TOKEN, "")


def prepare():
    """Prepare observation data for submission (stub)."""
    pass


def submit(local_path: str, remote_path: Optional[str] = None) -> None:
    """
    Upload a file or folder to the UCloud WebDAV share.

    Parameters
    ----------
    local_path : str
        Path to a local file or directory.
    remote_path : str, optional
        Destination path on the remote share.  Defaults to the local
        file/folder name.
    """
    local = Path(local_path)

    if not local.exists():
        raise FileNotFoundError(f"Local path not found: {local_path}")

    if local.is_dir():
        _upload_folder(local, remote_path)
    else:
        _upload_file(local, remote_path)


def _upload_folder(local: Path, remote_folder: Optional[str] = None) -> None:
    if remote_folder is None:
        remote_folder = local.name

    # Create remote folder via MKCOL
    folder_url = f"{_WEBDAV_URL}/{remote_folder}"
    print(f"Creating remote folder: {remote_folder}")

    response = requests.request("MKCOL", folder_url, auth=_AUTH)
    if response.status_code == 201:
        print("Folder created")
    elif response.status_code == 405:
        print("Folder already exists")
    else:
        print(f"Unexpected response: {response.status_code}")

    # Collect files
    files = [f for f in local.iterdir() if f.is_file()]
    total_files = len(files)
    total_size = sum(f.stat().st_size for f in files)

    print(
        f"Uploading {total_files} files "
        f"({total_size / 1024 / 1024:.1f} MB)..."
    )

    uploaded = 0
    failed = 0

    for idx, file_path in enumerate(files, 1):
        file_url = f"{_WEBDAV_URL}/{remote_folder}/{file_path.name}"
        file_size_mb = file_path.stat().st_size / 1024 / 1024

        print(
            f"  [{idx}/{total_files}] {file_path.name} "
            f"({file_size_mb:.2f} MB)",
            end="",
            flush=True,
        )

        try:
            with open(file_path, "rb") as f:
                resp = requests.put(file_url, data=f, auth=_AUTH)

            if resp.status_code in (200, 201, 204):
                print(" OK")
                uploaded += 1
            else:
                print(f" FAILED (HTTP {resp.status_code})")
                failed += 1
        except Exception as e:
            print(f" FAILED ({e})")
            failed += 1

    print(f"Upload complete: {uploaded} succeeded, {failed} failed")


def _upload_file(local: Path, remote_path: Optional[str] = None) -> bool:
    if remote_path is None:
        remote_path = local.name

    file_url = f"{_WEBDAV_URL}/{remote_path}"

    try:
        with open(local, "rb") as f:
            response = requests.put(file_url, data=f, auth=_AUTH)
        success = response.status_code in (200, 201, 204)
        if success:
            print(f"Uploaded {local.name}")
        else:
            print(f"Upload failed (HTTP {response.status_code})")
        return success
    except Exception as e:
        print(f"Upload failed: {e}")
        return False


def retrieve(
    remote_path: str, local_path: Optional[str] = None
) -> list[str] | bytes | None:
    """
    Retrieve a file or list a folder from the UCloud WebDAV share.

    Parameters
    ----------
    remote_path : str
        Remote path to retrieve.  If it looks like a folder (no extension
        or ends with ``/``), a PROPFIND listing is returned.  Otherwise
        the file contents are fetched via GET.
    local_path : str, optional
        If provided, save the downloaded content to this local path.

    Returns
    -------
    list[str]
        List of filenames when *remote_path* is a folder.
    bytes or None
        File content bytes when *remote_path* is a file and *local_path*
        is not given.  ``None`` when saved to disk.
    """
    url = f"{_WEBDAV_URL}/{remote_path.strip('/')}"

    # Decide if the remote path looks like a folder
    is_folder = remote_path.endswith("/") or "." not in Path(remote_path).name

    if is_folder:
        return _list_folder(url)
    else:
        return _download_file(url, local_path)


def _list_folder(url: str) -> list[str]:
    """PROPFIND to list the contents of a remote folder."""
    response = requests.request(
        "PROPFIND",
        url,
        auth=_AUTH,
        headers={"Depth": "1"},
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)
    # WebDAV namespace
    ns = {"d": "DAV:"}
    entries: list[str] = []

    for resp_el in root.findall("d:response", ns):
        href = resp_el.findtext("d:href", default="", namespaces=ns)
        name = href.rstrip("/").rsplit("/", 1)[-1]
        if name:
            entries.append(name)

    # First entry is usually the folder itself — drop it
    if entries:
        entries = entries[1:]

    return entries


def _download_file(
    url: str, local_path: Optional[str] = None
) -> bytes | None:
    """GET a single file from WebDAV."""
    response = requests.get(url, auth=_AUTH)
    response.raise_for_status()

    if local_path is not None:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_bytes(response.content)
        print(f"Saved to {local_path}")
        return None

    return response.content


def delete(remote_path: str) -> bool:
    """
    Delete a file or folder on the UCloud WebDAV share.

    Parameters
    ----------
    remote_path : str
        Remote path to delete.

    Returns
    -------
    bool
        ``True`` if the resource was deleted successfully, ``False``
        otherwise.
    """
    url = f"{_WEBDAV_URL}/{remote_path.strip('/')}"

    try:
        response = requests.request("DELETE", url, auth=_AUTH)
        if response.status_code in (200, 204):
            print(f"Deleted {remote_path}")
            return True
        elif response.status_code == 404:
            print(f"Not found: {remote_path}")
            return False
        else:
            print(f"Delete failed (HTTP {response.status_code})")
            return False
    except Exception as e:
        print(f"Delete failed: {e}")
        return False
