import ctypes
from ctypes import wintypes
import json
import os
import platform
import subprocess
from pathlib import Path
import sys

import cv2
import numpy as np
import psutil
import torch
from .common import put


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=30)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def ac_status():
    if os.name != "nt":
        battery = psutil.sensors_battery()
        return battery.power_plugged if battery else None
    class Power(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("battery", ctypes.c_ubyte), ("pct", ctypes.c_ubyte),
                    ("system", ctypes.c_ubyte), ("life", wintypes.DWORD), ("full", wintypes.DWORD)]
    status = Power()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)) or status.ac == 255:
        return None
    return bool(status.ac)


def cpu_status():
    fields = "PercentProcessorPerformance,PercentPerformanceLimit,PerformanceLimitFlags,ProcessorFrequency,PercentofMaximumFrequency"
    if os.name!="nt":
        return {"ac_online":ac_status()}
    query = ("Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation "
             "-Filter \"Name = '_Total'\" | Select-Object "+fields+" | ConvertTo-Json -Compress")
    output = command("powershell.exe","-NoProfile","-NonInteractive","-Command",query)
    try:
        counters=json.loads(output)
    except ValueError:
        counters={"unavailable":output[:240]}
    return dict(ac_online=ac_status(), power_profile=command("powercfg","/getactivescheme"), system_total_counters=counters)


def cpu_sets():
    if os.name != "nt":
        return []
    size = wintypes.ULONG()
    fn = ctypes.windll.kernel32.GetSystemCpuSetInformation
    fn(None, 0, ctypes.byref(size), None, 0)
    buf = ctypes.create_string_buffer(size.value)
    if not fn(buf, size, ctypes.byref(size), None, 0):
        raise OSError("GetSystemCpuSetInformation failed")
    off, result = 0, []
    while off < size.value:
        raw = buf.raw
        n = int.from_bytes(raw[off:off+4], "little")
        if n < 20:
            raise ValueError("Invalid Windows CPU set record")
        if int.from_bytes(raw[off+4:off+8], "little") == 0:
            result.append(dict(group=int.from_bytes(raw[off+12:off+14], "little"), logical=raw[off+14],
                               physical=raw[off+15], efficiency=raw[off+18]))
        off += n
    return result


def pin_cpu(threads):
    process = psutil.Process()
    available = process.cpu_affinity()
    topology = [c for c in cpu_sets() if c["logical"] in available and c["group"] == 0]
    logical = []
    if topology:
        efficiency = max(c["efficiency"] for c in topology)
        by_core = {}
        for c in topology:
            if c["efficiency"] == efficiency:
                by_core.setdefault(c["physical"], c["logical"])
        logical = list(by_core.values())[:threads]
    else:
        logical = available[:threads]
    if len(logical) != threads:
        raise ValueError(f"Requested {threads} physical cores, found {logical}")
    process.cpu_affinity(logical)
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    return dict(threads=threads, affinity=process.cpu_affinity(), topology=topology,
                ac_online=ac_status(), power_profile=command("powercfg", "/getactivescheme") if os.name=="nt" else None,
                throttling="Windows ProcessorInformation performance-limit flags sampled outside primary timing.")


def inventory(out):
    from importlib.metadata import version
    import openvino as ov
    versions = {k:version(k) for k in ("ultralytics","openvino","torch","torchvision","numpy","opencv-python",
                                       "scipy","psutil","pillow","matplotlib","threadpoolctl")}
    info = dict(os=platform.platform(), python=sys.version, packages=versions, cpu=platform.processor(),
                device=ov.Core().get_property("CPU", "FULL_DEVICE_NAME"), logical_cpus=os.cpu_count(),
                ram_gib=round(psutil.virtual_memory().total/2**30, 2), cuda=torch.version.cuda,
                gpu=command("nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"),
                ac_online=ac_status(), cpu_sets=cpu_sets(), power_profile=command("powercfg","/getactivescheme"),
                cudnn=torch.backends.cudnn.version(), cpu_status=cpu_status())
    put(Path(out)/"audit/environment.json", info)
    return info
