import time
import re
import json
import signal
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL_PATH = "/workspace/yolov5nu_opset11.onnx"
WARMUP = 5
RUNS = 50
TEGRA_INTERVAL_MS = 20

POM_5V_IN_RE = re.compile(r"POM_5V_IN\s+(\d+)\/(\d+)")
POM_5V_GPU_RE = re.compile(r"POM_5V_GPU\s+(\d+)\/(\d+)")
POM_5V_CPU_RE = re.compile(r"POM_5V_CPU\s+(\d+)\/(\d+)")
GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")


def make_input(sess):
    inp = sess.get_inputs()[0]
    shape = inp.shape

    fixed = []
    for i, s in enumerate(shape):
        if s is None or isinstance(s, str):
            if i == 0:
                fixed.append(10)
            elif i == 1:
                fixed.append(3)
            else:
                fixed.append(640)
        else:
            fixed.append(int(s))

    x = np.random.rand(*fixed).astype(np.float32)
    return inp.name, x, fixed


def find_tegrastats():
    candidates = [
        shutil.which("tegrastats"),
        "/usr/bin/tegrastats",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def tegrastats_reader_thread(proc, log_file):
    for line in proc.stdout:
        ts = time.perf_counter()
        log_file.write(f"{ts:.9f}\t{line}")
    log_file.flush()


def start_tegrastats(log_path, interval_ms=20):
    tegra_bin = find_tegrastats()
    if tegra_bin is None:
        print("tegrastats not found; power logging disabled")
        return None, None, None

    stdbuf_bin = shutil.which("stdbuf")
    log_file = open(log_path, "w", buffering=1)

    if stdbuf_bin is not None:
        cmd = [stdbuf_bin, "-oL", tegra_bin, "--interval", str(interval_ms)]
    else:
        cmd = [tegra_bin, "--interval", str(interval_ms)]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    reader = threading.Thread(
        target=tegrastats_reader_thread,
        args=(proc, log_file),
        daemon=True,
    )
    reader.start()

    return proc, log_file, reader


def stop_tegrastats(proc, log_file, reader):
    if proc is None:
        return

    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    # Join the reader before closing the file so the thread cannot write
    # to a closed file handle.
    if reader is not None:
        reader.join(timeout=2)

    if log_file is not None:
        log_file.close()


def extract_pair(regex, text):
    m = regex.search(text)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def extract_single(regex, text):
    m = regex.search(text)
    if not m:
        return None
    return int(m.group(1))


def parse_tegrastats(log_path, t_start=None, t_end=None):
    rows = []

    with open(log_path, "r") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if "\t" not in raw:
                continue

            ts_str, line = raw.split("\t", 1)
            try:
                ts = float(ts_str)
            except ValueError:
                continue

            if t_start is not None and ts < t_start:
                continue
            if t_end is not None and ts > t_end:
                continue

            pom_in_inst, pom_in_avg = extract_pair(POM_5V_IN_RE, line)
            pom_gpu_inst, pom_gpu_avg = extract_pair(POM_5V_GPU_RE, line)
            pom_cpu_inst, pom_cpu_avg = extract_pair(POM_5V_CPU_RE, line)
            gr3d = extract_single(GR3D_RE, line)

            rows.append({
                "ts": ts,
                "line": line,
                "pom_in_inst_mw": pom_in_inst,
                "pom_in_avg_mw": pom_in_avg,
                "pom_gpu_inst_mw": pom_gpu_inst,
                "pom_gpu_avg_mw": pom_gpu_avg,
                "pom_cpu_inst_mw": pom_cpu_inst,
                "pom_cpu_avg_mw": pom_cpu_avg,
                "gr3d_pct": gr3d,
            })

    return rows


def summarize_tegrastats_rows(rows):
    if not rows:
        return None

    def values(key):
        return np.array([r[key] for r in rows if r[key] is not None], dtype=np.float64)

    pom_in = values("pom_in_inst_mw")
    pom_gpu = values("pom_gpu_inst_mw")
    pom_cpu = values("pom_cpu_inst_mw")
    gr3d = values("gr3d_pct")

    out = {
        "samples": len(rows),
        "t_first": rows[0]["ts"],
        "t_last": rows[-1]["ts"],
    }

    if len(pom_in):
        out["power_in_mean_mw"] = pom_in.mean()
        out["power_in_std_mw"] = pom_in.std()
        out["power_in_p50_mw"] = np.percentile(pom_in, 50)
        out["power_in_p90_mw"] = np.percentile(pom_in, 90)
        out["power_in_p99_mw"] = np.percentile(pom_in, 99)
        out["power_in_avg_last_mw"] = next(
            (r["pom_in_avg_mw"] for r in reversed(rows) if r["pom_in_avg_mw"] is not None),
            None,
        )

    if len(pom_gpu):
        out["power_gpu_mean_mw"] = pom_gpu.mean()
        out["power_gpu_std_mw"] = pom_gpu.std()
        out["power_gpu_p50_mw"] = np.percentile(pom_gpu, 50)
        out["power_gpu_p90_mw"] = np.percentile(pom_gpu, 90)
        out["power_gpu_p99_mw"] = np.percentile(pom_gpu, 99)
        out["power_gpu_avg_last_mw"] = next(
            (r["pom_gpu_avg_mw"] for r in reversed(rows) if r["pom_gpu_avg_mw"] is not None),
            None,
        )

    if len(pom_cpu):
        out["power_cpu_mean_mw"] = pom_cpu.mean()
        out["power_cpu_std_mw"] = pom_cpu.std()
        out["power_cpu_p50_mw"] = np.percentile(pom_cpu, 50)
        out["power_cpu_p90_mw"] = np.percentile(pom_cpu, 90)
        out["power_cpu_p99_mw"] = np.percentile(pom_cpu, 99)
        out["power_cpu_avg_last_mw"] = next(
            (r["pom_cpu_avg_mw"] for r in reversed(rows) if r["pom_cpu_avg_mw"] is not None),
            None,
        )

    if len(gr3d):
        out["gr3d_mean_pct"] = gr3d.mean()
        out["gr3d_std_pct"] = gr3d.std()
        out["gr3d_p50_pct"] = np.percentile(gr3d, 50)
        out["gr3d_p90_pct"] = np.percentile(gr3d, 90)
        out["gr3d_p99_pct"] = np.percentile(gr3d, 99)

    return out


def load_ort_profile(profile_file):
    with open(profile_file, "r") as f:
        return json.load(f)


def select_ort_events(events):
    selected = []
    for e in events:
        if "ts" not in e:
            continue
        selected.append(dict(e))
    return selected


def find_first_event_ts(events, names):
    for e in events:
        if e.get("name") in names and "ts" in e:
            return e["ts"]
    return None


def normalize_ort_trace_to_zero(events):
    if not events:
        return []

    ts0 = min(e["ts"] for e in events if "ts" in e)
    out = []
    for e in events:
        x = dict(e)
        x["ts"] = x["ts"] - ts0
        out.append(x)
    return out


def tegrastats_rows_to_trace_events(rows, timeline_origin_s):
    """Convert tegrastats rows to Perfetto/chrome-trace counter events.

    All timestamps are expressed relative to *timeline_origin_s*, which must
    be the same perf_counter value used as the origin for every other event in
    the merged trace.
    """
    events = []

    for r in rows:
        ts_us = (r["ts"] - timeline_origin_s) * 1e6

        if r["gr3d_pct"] is not None:
            events.append({
                "name": "GR3D_FREQ",
                "cat": "tegrastats",
                "ph": "C",
                "ts": ts_us,
                "pid": 9000,
                "tid": 1,
                "args": {"percent": r["gr3d_pct"]},
            })

        if r["pom_in_inst_mw"] is not None:
            events.append({
                "name": "POM_5V_IN",
                "cat": "tegrastats",
                "ph": "C",
                "ts": ts_us,
                "pid": 9000,
                "tid": 1,
                "args": {"mW": r["pom_in_inst_mw"]},
            })

        if r["pom_gpu_inst_mw"] is not None:
            events.append({
                "name": "POM_5V_GPU",
                "cat": "tegrastats",
                "ph": "C",
                "ts": ts_us,
                "pid": 9000,
                "tid": 1,
                "args": {"mW": r["pom_gpu_inst_mw"]},
            })

        if r["pom_cpu_inst_mw"] is not None:
            events.append({
                "name": "POM_5V_CPU",
                "cat": "tegrastats",
                "ph": "C",
                "ts": ts_us,
                "pid": 9000,
                "tid": 1,
                "args": {"mW": r["pom_cpu_inst_mw"]},
            })

    return events


def make_region_event(name, t0_s, t1_s, origin_s, pid=8000, tid=1):
    return {
        "name": name,
        "cat": "benchmark",
        "ph": "X",
        "ts": (t0_s - origin_s) * 1e6,
        "dur": (t1_s - t0_s) * 1e6,
        "pid": pid,
        "tid": tid,
        "args": {},
    }


def make_instant_event(name, ts_s, origin_s, pid=8000, tid=1, args=None):
    return {
        "name": name,
        "cat": "benchmark",
        "ph": "i",
        "s": "g",
        "ts": (ts_s - origin_s) * 1e6,
        "pid": pid,
        "tid": tid,
        "args": {} if args is None else args,
    }


def merge_ort_and_tegrastats(
    profile_file,
    tegra_log,
    timeline_origin_s,
    sync_perf_s,
    aligned_rows,
    output_file,
):
    """Merge an ORT profile and tegrastats rows into a single chrome-trace JSON.

    Timeline layout
    ---------------
    All events share a single origin: *timeline_origin_s* (a perf_counter
    value captured once at the very start of the timed region, before
    tegrastats is started).

    ORT timestamps are in microseconds relative to an internal ORT clock.
    We align them by finding the first session.run event in the ORT trace
    and declaring that its wall-clock time equals *sync_perf_s* — the
    perf_counter value sampled immediately before the sync run was issued.

    Concretely:
        ort_event_wall_us = ort_event_ts_us
                            - sync_ort_us          # make sync-run the zero
                            + (sync_perf_s - timeline_origin_s) * 1e6
                                                   # shift to shared timeline

    Tegrastats rows already carry perf_counter timestamps, so their offset is
    simply  (row_ts - timeline_origin_s) * 1e6.
    """
    ort_events = select_ort_events(load_ort_profile(profile_file))
    ort_events = normalize_ort_trace_to_zero(ort_events)

    if not ort_events:
        with open(output_file, "w") as f:
            json.dump([], f)
        return output_file

    # ORT timestamp of the first session.run inside the profiled session.
    # After normalize_ort_trace_to_zero this is in µs relative to the first
    # ORT event in the file.
    sync_ort_us = find_first_event_ts(
        ort_events,
        names={"model_run", "SequentialExecutor::Execute", "session.run"},
    )
    if sync_ort_us is None:
        sync_ort_us = min(e["ts"] for e in ort_events if "ts" in e)

    # Wall-clock position of the sync run on the shared timeline (µs).
    sync_wall_us = (sync_perf_s - timeline_origin_s) * 1e6

    # Shift every ORT event so that sync_ort_us lands at sync_wall_us.
    ort_offset_us = sync_wall_us - sync_ort_us

    merged = []

    for e in ort_events:
        x = dict(e)
        x["ts"] = x["ts"] + ort_offset_us
        merged.append(x)

    # Tegrastats rows use the same timeline_origin_s reference.
    merged.extend(tegrastats_rows_to_trace_events(aligned_rows, timeline_origin_s))

    # Marker so the viewer shows exactly where the sync anchor lands.
    merged.append(
        make_instant_event("sync_anchor", sync_perf_s, timeline_origin_s, pid=8000, tid=1)
    )

    merged.sort(key=lambda e: e.get("ts", 0.0))

    with open(output_file, "w") as f:
        json.dump(merged, f)

    return output_file


def print_power_summary(power_stats, avg_batch, batch_size):
    if power_stats is None:
        print("\nPower: unavailable")
        return

    print("\nPower over timed region:")
    print(f"Samples:              {power_stats['samples']}")

    if "power_in_mean_mw" in power_stats:
        mean_power_w = power_stats["power_in_mean_mw"] / 1000.0
        joules_per_batch = mean_power_w * avg_batch
        joules_per_image = joules_per_batch / batch_size

        print(f"Board mean power:     {power_stats['power_in_mean_mw'] / 1000.0:.3f} W")
        print(f"Board std power:      {power_stats['power_in_std_mw'] / 1000.0:.3f} W")
        print(f"Board P50 power:      {power_stats['power_in_p50_mw'] / 1000.0:.3f} W")
        print(f"Board P90 power:      {power_stats['power_in_p90_mw'] / 1000.0:.3f} W")
        print(f"Board P99 power:      {power_stats['power_in_p99_mw'] / 1000.0:.3f} W")
        if power_stats["power_in_avg_last_mw"] is not None:
            print(f"Board last avg power: {power_stats['power_in_avg_last_mw'] / 1000.0:.3f} W")
        print(f"Energy / batch:       {joules_per_batch:.4f} J")
        print(f"Energy / image:       {joules_per_image:.4f} J")

    if "power_gpu_mean_mw" in power_stats:
        print(f"GPU rail mean power:  {power_stats['power_gpu_mean_mw'] / 1000.0:.3f} W")
        print(f"GPU rail P90 power:   {power_stats['power_gpu_p90_mw'] / 1000.0:.3f} W")
        if power_stats["power_gpu_avg_last_mw"] is not None:
            print(f"GPU rail last avg:    {power_stats['power_gpu_avg_last_mw'] / 1000.0:.3f} W")

    if "power_cpu_mean_mw" in power_stats:
        print(f"CPU rail mean power:  {power_stats['power_cpu_mean_mw'] / 1000.0:.3f} W")
        print(f"CPU rail P90 power:   {power_stats['power_cpu_p90_mw'] / 1000.0:.3f} W")
        if power_stats["power_cpu_avg_last_mw"] is not None:
            print(f"CPU rail last avg:    {power_stats['power_cpu_avg_last_mw'] / 1000.0:.3f} W")

    if "gr3d_mean_pct" in power_stats:
        print(f"GR3D mean util:       {power_stats['gr3d_mean_pct']:.1f} %")
        print(f"GR3D P90 util:        {power_stats['gr3d_p90_pct']:.1f} %")


def run_warmup(providers, input_name, x, num_threads=None):
    """Run warmup inference in a dedicated throw-away session.

    Using a separate session means the profiled session (created afterwards)
    will only contain events from the sync run and the timed benchmark loop,
    keeping the profile clean of warmup noise.
    """
    so = ort.SessionOptions()
    so.enable_profiling = False
    if num_threads is not None:
        so.intra_op_num_threads = num_threads

    sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=providers)
    for _ in range(WARMUP):
        sess.run(None, {input_name: x})
    # No end_profiling needed; the session is simply discarded.


def benchmark(providers, label, num_threads=None):
    print(f"\n--- {label} ---")

    # ------------------------------------------------------------------ #
    # Warmup — dedicated session, profiling disabled                       #
    # ------------------------------------------------------------------ #
    # We need the input shape first; use a lightweight session for that.
    so_probe = ort.SessionOptions()
    so_probe.enable_profiling = False
    sess_probe = ort.InferenceSession(MODEL_PATH, sess_options=so_probe, providers=providers)
    input_name, x, shape = make_input(sess_probe)
    batch_size = int(shape[0])
    del sess_probe  # discard immediately

    run_warmup(providers, input_name, x, num_threads=num_threads)

    # ------------------------------------------------------------------ #
    # Profiled session — contains only sync run + benchmark loop           #
    # ------------------------------------------------------------------ #
    so = ort.SessionOptions()
    so.enable_profiling = True
    if num_threads is not None:
        so.intra_op_num_threads = num_threads

    sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=providers)

    print("Requested providers:", providers)
    print("Actual providers:   ", sess.get_providers())
    print("intra_op_num_threads:", num_threads if num_threads is not None else "auto")
    print("Input shape:        ", shape)
    print("Batch size:         ", batch_size)

    # ------------------------------------------------------------------ #
    # Single shared timeline origin                                        #
    # ------------------------------------------------------------------ #
    # timeline_origin_s is the single perf_counter reference used by every
    # timestamp in the merged trace: tegrastats rows, ORT events, and region
    # markers all subtract this value to produce their trace timestamps.
    #
    # We capture it right before starting tegrastats so that the very first
    # tegrastats sample has a non-negative trace timestamp.
    timeline_origin_s = time.perf_counter()

    tegra_log = f"tegrastats_{label.lower()}.log"
    tegra_proc, tegra_file, tegra_reader = start_tegrastats(
        tegra_log,
        interval_ms=TEGRA_INTERVAL_MS,
    )

    batch_times = []
    benchmark_t0 = None
    benchmark_t1 = None
    profile_file = None
    merged_trace = None

    try:
        if tegra_proc is not None:
            time.sleep(0.1)  # let tegrastats emit at least one sample

        # ---------------------------------------------------------------- #
        # Sync run                                                          #
        # ---------------------------------------------------------------- #
        # sync_perf_s is sampled immediately before sess.run() is called.
        # Inside merge_ort_and_tegrastats we align the first ORT session.run
        # event to this perf_counter value so that ORT and tegrastats share
        # the same timeline_origin_s reference.
        sync_perf_s = time.perf_counter()
        sess.run(None, {input_name: x})

        # ---------------------------------------------------------------- #
        # Timed benchmark loop                                              #
        # ---------------------------------------------------------------- #
        benchmark_t0 = time.perf_counter()
        for _ in range(RUNS):
            t0 = time.perf_counter()
            sess.run(None, {input_name: x})
            t1 = time.perf_counter()
            batch_times.append(t1 - t0)
        benchmark_t1 = time.perf_counter()

    finally:
        try:
            profile_file = sess.end_profiling()
        except Exception as e:
            print(f"Warning: end_profiling failed: {e}")
        try:
            stop_tegrastats(tegra_proc, tegra_file, tegra_reader)
        except Exception as e:
            print(f"Warning: stop_tegrastats failed: {e}")

    batch_times = np.array(batch_times, dtype=np.float64)
    image_times = batch_times / batch_size

    avg_batch = batch_times.mean()
    std_batch = batch_times.std()
    p50_batch = np.percentile(batch_times, 50)
    p90_batch = np.percentile(batch_times, 90)
    p99_batch = np.percentile(batch_times, 99)

    avg_image = image_times.mean()
    std_image = image_times.std()
    p50_image = np.percentile(image_times, 50)
    p90_image = np.percentile(image_times, 90)
    p99_image = np.percentile(image_times, 99)

    throughput_fps = batch_size / avg_batch

    print("\nPer-batch latency:")
    print(f"Average latency: {avg_batch * 1000:.2f} ms")
    print(f"Std latency:     {std_batch * 1000:.2f} ms")
    print(f"P50 latency:     {p50_batch * 1000:.2f} ms")
    print(f"P90 latency:     {p90_batch * 1000:.2f} ms")
    print(f"P99 latency:     {p99_batch * 1000:.2f} ms")

    print("\nPer-image latency:")
    print(f"Average latency: {avg_image * 1000:.2f} ms")
    print(f"Std latency:     {std_image * 1000:.2f} ms")
    print(f"P50 latency:     {p50_image * 1000:.2f} ms")
    print(f"P90 latency:     {p90_image * 1000:.2f} ms")
    print(f"P99 latency:     {p99_image * 1000:.2f} ms")

    print(f"\nThroughput:      {throughput_fps:.2f} FPS")

    # Parse tegrastats over the full aligned window (sync → benchmark end)
    # and the power-only window (benchmark start → benchmark end) in one
    # file pass, then slice in memory.
    all_rows = parse_tegrastats(tegra_log, t_start=sync_perf_s, t_end=benchmark_t1)
    power_rows = [r for r in all_rows if r["ts"] >= benchmark_t0]
    power_stats = summarize_tegrastats_rows(power_rows)

    print_power_summary(power_stats, avg_batch, batch_size)

    if profile_file and Path(profile_file).exists() and Path(tegra_log).exists():
        merged_trace = f"merged_trace_{label.lower()}.json"
        merge_ort_and_tegrastats(
            profile_file=profile_file,
            tegra_log=tegra_log,
            timeline_origin_s=timeline_origin_s,
            sync_perf_s=sync_perf_s,
            aligned_rows=all_rows,
            output_file=merged_trace,
        )

    print("\nArtifacts:")
    print("ONNX profile:    ", profile_file)
    print("Tegrastats log:  ", tegra_log)
    if merged_trace is not None:
        print("Merged trace:    ", merged_trace)
    else:
        print("Merged trace:     unavailable")

    return {
        "avg_batch": avg_batch,
        "avg_image": avg_image,
        "throughput_fps": throughput_fps,
        "power_stats": power_stats,
        "profile_file": profile_file,
        "tegrastats_log": tegra_log,
        "merged_trace": merged_trace,
    }


def main():
    print("onnxruntime:", ort.__version__)
    print("available providers:", ort.get_available_providers())

    gpu_stats = benchmark(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "GPU",
    )

    cpu_stats = benchmark(
        ["CPUExecutionProvider"],
        "CPU",
        num_threads=4,
    )

    print("\n=== Summary ===")
    print(f"GPU avg batch latency:  {gpu_stats['avg_batch'] * 1000:.2f} ms")
    print(f"GPU avg image latency:  {gpu_stats['avg_image'] * 1000:.2f} ms")
    print(f"GPU throughput:         {gpu_stats['throughput_fps']:.2f} FPS")
    if gpu_stats["power_stats"] is not None and "power_gpu_mean_mw" in gpu_stats["power_stats"]:
        print(f"GPU rail mean power:    {gpu_stats['power_stats']['power_gpu_mean_mw'] / 1000.0:.3f} W")
    if gpu_stats["power_stats"] is not None and "gr3d_mean_pct" in gpu_stats["power_stats"]:
        print(f"GPU mean GR3D:          {gpu_stats['power_stats']['gr3d_mean_pct']:.1f} %")

    print(f"CPU avg batch latency:  {cpu_stats['avg_batch'] * 1000:.2f} ms")
    print(f"CPU avg image latency:  {cpu_stats['avg_image'] * 1000:.2f} ms")
    print(f"CPU throughput:         {cpu_stats['throughput_fps']:.2f} FPS")
    if cpu_stats["power_stats"] is not None and "power_cpu_mean_mw" in cpu_stats["power_stats"]:
        print(f"CPU rail mean power:    {cpu_stats['power_stats']['power_cpu_mean_mw'] / 1000.0:.3f} W")

    print(f"Batch speedup:          {cpu_stats['avg_batch'] / gpu_stats['avg_batch']:.2f}x")
    print(f"Image speedup:          {cpu_stats['avg_image'] / gpu_stats['avg_image']:.2f}x")

    print("\nPreserved original files and merged files:")
    print("GPU ORT profile:        ", gpu_stats["profile_file"])
    print("GPU tegrastats log:     ", gpu_stats["tegrastats_log"])
    print("GPU merged trace:       ", gpu_stats["merged_trace"])
    print("CPU ORT profile:        ", cpu_stats["profile_file"])
    print("CPU tegrastats log:     ", cpu_stats["tegrastats_log"])
    print("CPU merged trace:       ", cpu_stats["merged_trace"])


if __name__ == "__main__":
    main()