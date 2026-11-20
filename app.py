"""
app.py — Streamlit web app for the Multi-Track Stereo Crosstalk Analyser

Upload a stereo WAV file recorded from an alternating Left/Right test record
(e.g. Shure TTR-109) and the app detects all tracks, measures crosstalk,
phase and channel balance for each pair, and displays bar charts and tables.

Deploy on Streamlit Community Cloud:
  1. Put this file + requirements.txt in a GitHub repo
  2. Connect the repo at share.streamlit.io
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import streamlit as st
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

# -----------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Track Crosstalk Analyser",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 Multi-Track Stereo Crosstalk Analyser")
st.markdown(
    "Upload a stereo WAV file from an alternating L/R channel test record "
    "(e.g. **Shure TTR-109**). The app auto-detects each track, measures "
    "**crosstalk**, **phase** and **channel balance**, and shows bar charts "
    "plus a results table."
)

# -----------------------------------------------------------------------
# Sidebar — settings
# -----------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

user_description = st.sidebar.text_input(
    "Run notes (optional)",
    placeholder="e.g. Shure V15-IV, VTF 1.25g, VTA −6°",
)

plot_style = st.sidebar.radio("Plot style", ["Bar chart", "Line plot"], index=0)
plot_style_key = "bar" if plot_style == "Bar chart" else "line"

st.sidebar.markdown("---")
st.sidebar.subheader("Y-axis limits")

xtalk_auto = st.sidebar.checkbox("Auto-scale crosstalk axis", value=True)
col1, col2 = st.sidebar.columns(2)
xtalk_min = col1.number_input("Min (dB)", value=-55, step=5, disabled=xtalk_auto)
xtalk_max = col2.number_input("Max (dB)", value=-20, step=5, disabled=xtalk_auto)
xtalk_ylim = None if xtalk_auto else (xtalk_min, xtalk_max)

phase_auto = st.sidebar.checkbox("Auto-scale phase axis", value=True)
col3, col4 = st.sidebar.columns(2)
phase_min = col3.number_input("Min (°)", value=-180, step=45, disabled=phase_auto)
phase_max = col4.number_input("Max (°)", value=180, step=45, disabled=phase_auto)
phase_ylim = None if phase_auto else (phase_min, phase_max)

st.sidebar.markdown("---")
st.sidebar.subheader("Detection parameters")

track_min_dur = st.sidebar.number_input(
    "Min track duration (s)", value=5.0, min_value=0.5, step=0.5,
    help="Segments shorter than this are ignored (removes transition noise)"
)
trim_frac = st.sidebar.slider(
    "Steady-state trim", 0.05, 0.35, 0.15, 0.05,
    help="Fraction trimmed from each end of a track to avoid clicks"
)
analysis_max_s = st.sidebar.number_input(
    "Max analysis window (s)", value=10.0, min_value=1.0, step=1.0,
)

# -----------------------------------------------------------------------
# Analysis constants
# -----------------------------------------------------------------------
CLASSIFY_WIN_S   = 0.1
NOISE_THRESH_FRAC = 0.01
ZERO_PAD_FACTOR  = 8
FREQ_MIN_HZ      = 200.0
FREQ_MAX_HZ      = 20000.0
FREQ_HW_FRAC     = 0.05
BP_ORDER         = 4
BP_HW_FRAC       = 0.08
BP_HW_MIN_HZ     = 25.0
LABEL_FONTSIZE   = 13   # value labels inside bars


# -----------------------------------------------------------------------
# Analysis functions
# -----------------------------------------------------------------------

def detect_tracks(L, R, sr, min_dur):
    win = max(1, int(CLASSIFY_WIN_S * sr))
    n_win = min(len(L), len(R)) // win
    peak = max(np.sqrt(np.mean(L**2)), np.sqrt(np.mean(R**2)))
    thresh = NOISE_THRESH_FRAC * peak

    labels = []
    for i in range(n_win):
        rL = np.sqrt(np.mean(L[i*win:(i+1)*win]**2))
        rR = np.sqrt(np.mean(R[i*win:(i+1)*win]**2))
        if max(rL, rR) < thresh:
            labels.append("S")
        else:
            labels.append("L" if rL >= rR else "R")

    segs, start, cur = [], 0, labels[0]
    for i, lbl in enumerate(labels[1:], 1):
        if lbl != cur:
            segs.append((cur, start*win, i*win))
            start, cur = i, lbl
    segs.append((cur, start*win, n_win*win))

    return [(ch, s, e) for ch, s, e in segs
            if ch in ("L","R") and (e-s)/sr >= min_dur]


def bandpass(x, sr, fc):
    hw = max(BP_HW_MIN_HZ, fc * BP_HW_FRAC)
    lo = max(1.0, fc - hw)
    hi = min(sr/2*0.98, fc + hw)
    b, a = butter(BP_ORDER, [lo, hi], btype="bandpass", fs=sr)
    return filtfilt(b, a, x)


def dominant_freq(x, sr):
    n_fft = int(2**np.ceil(np.log2(len(x)*ZERO_PAD_FACTOR)))
    spec = np.abs(np.fft.rfft(x*np.hanning(len(x)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    mask = (freqs >= FREQ_MIN_HZ) & (freqs <= FREQ_MAX_HZ)
    return freqs[int(np.argmax(spec*mask))]


def analyze_segment(x, sr, fc):
    xf = bandpass(x, sr, fc)
    n = len(xf)
    n_fft = int(2**np.ceil(np.log2(n*ZERO_PAD_FACTOR)))
    spec = np.fft.rfft(xf*np.hanning(n), n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    hw = max(BP_HW_MIN_HZ, fc*FREQ_HW_FRAC)
    mask = (freqs >= fc-hw) & (freqs <= fc+hw)
    mags = np.abs(spec[mask])
    pk = int(np.argmax(mags))
    mag = mags[pk] / (0.5*n/2)
    phase = np.angle(spec[mask][pk])
    return mag, phase


def wrap_deg(a):
    return ((a + 180.0) % 360.0) - 180.0


def run_analysis(L, R, sr, trim_frac, analysis_max_s, min_dur, full_scale):
    tracks = detect_tracks(L, R, sr, min_dur)
    if not tracks:
        return [], []

    results = []
    prog = st.progress(0, text="Analysing tracks…")
    for i, (ch, s, e) in enumerate(tracks):
        prog.progress((i+1)/len(tracks), text=f"Track {i+1} of {len(tracks)}…")
        dur = e - s
        trim = int(dur * trim_frac)
        ss, se = s+trim, e-trim
        ml = int(analysis_max_s * sr)
        if se-ss > ml:
            mid = (ss+se)//2
            ss, se = mid-ml//2, mid+ml//2

        main_sig = L[ss:se] if ch=="L" else R[ss:se]
        leak_sig = R[ss:se] if ch=="L" else L[ss:se]
        leak_ch = "R" if ch=="L" else "L"

        freq = dominant_freq(main_sig, sr)
        mag_main, ph_main = analyze_segment(main_sig, sr, freq)
        mag_leak, ph_leak = analyze_segment(leak_sig, sr, freq)

        main_filt = bandpass(main_sig, sr, freq)
        rms_db = 20*np.log10(np.sqrt(np.mean((main_filt/full_scale)**2)) + 1e-12)

        xtalk = 20*np.log10(mag_leak/mag_main) if mag_main > 0 else float("nan")
        phase = wrap_deg(np.degrees(ph_leak - ph_main))

        results.append(dict(
            track=i+1, main=ch, leak=leak_ch,
            freq=freq, xtalk_db=xtalk, phase_diff_deg=phase,
            rms_main_db=rms_db, t_start=s/sr, t_end=e/sr,
        ))
    prog.empty()

    lr = [r for r in results if r["main"]=="L"]
    rl = [r for r in results if r["main"]=="R"]
    pairs = list(zip(lr, rl))
    return results, pairs


# -----------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------

COLORS = {"L->R": "#4FC3F7", "R->L": "#FF8A65"}
RCPARAMS = {
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "text.color": "white", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white", "axes.titlecolor": "white",
}


def make_charts(results, pairs, filename, user_desc, plot_style,
                xtalk_ylim, phase_ylim):
    plt.style.use("dark_background")
    plt.rcParams.update(RCPARAMS)

    lr = [r for r in results if r["main"]=="L"]
    rl = [r for r in results if r["main"]=="R"]

    fig, (ax_bal, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 14))
    fig.subplots_adjust(hspace=0.55, top=0.92)
    title = "Multi-Track Stereo Crosstalk & Phase Analysis"
    sub = filename + (f"\n{user_desc}" if user_desc else "")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.962, sub, ha="center", va="top", fontsize=11, color="white")

    # --- Crosstalk + Phase bars ---
    for label, group, marker in (("L->R", lr, "o"), ("R->L", rl, "s")):
        if not group:
            continue
        xts = [r["xtalk_db"] for r in group]
        phs = [r["phase_diff_deg"] for r in group]

        if plot_style == "bar":
            bar_w = 0.35
            offset = -bar_w/2 if label=="L->R" else +bar_w/2
            pair_idxs = [(r["track"]+1)//2 for r in group]
            xpos = [p+offset for p in pair_idxs]
            bars1 = ax1.bar(xpos, xts, width=bar_w, color=COLORS[label],
                            label=label, alpha=0.88, edgecolor="#333333")
            bars2 = ax2.bar(xpos, phs, width=bar_w, color=COLORS[label],
                            label=label, alpha=0.88, edgecolor="#333333")
            for bar, val in zip(bars1, xts):
                ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()-0.5,
                         f"{val:.1f}", ha="center", va="top",
                         fontsize=LABEL_FONTSIZE, color="white", fontweight="bold")
            for bar, val in zip(bars2, phs):
                va = "top" if val >= 0 else "bottom"
                y  = val-2  if val >= 0 else val+2
                ax2.text(bar.get_x()+bar.get_width()/2, y, f"{val:.0f}°",
                         ha="center", va=va,
                         fontsize=LABEL_FONTSIZE, color="white", fontweight="bold")
        else:
            trks = [r["track"] for r in group]
            ax1.plot(trks, xts, marker+"-", color=COLORS[label],
                     label=label, linewidth=2, markersize=7)
            ax2.plot(trks, phs, marker+"-", color=COLORS[label],
                     label=label, linewidth=2, markersize=7)

    ax1.set_ylabel("Crosstalk (dB)")
    ax1.set_title("Crosstalk vs. " + ("Track Pair" if plot_style=="bar" else "Track Number"))
    ax1.legend(loc="upper left"); ax1.grid(True, alpha=0.3, axis="y")
    if xtalk_ylim:
        ax1.set_ylim(xtalk_ylim)
    elif plot_style == "bar":
        all_xt = [r["xtalk_db"] for r in results]
        ax1.set_ylim(min(all_xt)-abs(min(all_xt))*0.08, 0)

    ax2.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax2.set_ylabel("Phase Difference (deg)")
    ax2.set_title("Leakage-to-Main Phase vs. " + ("Track Pair" if plot_style=="bar" else "Track Number"))
    ax2.legend(loc="best"); ax2.grid(True, alpha=0.3, axis="y")
    if phase_ylim:
        ax2.set_ylim(phase_ylim)

    if plot_style == "bar":
        n_pairs = max((r["track"]+1)//2 for r in results)
        for ax in (ax1, ax2):
            ax.set_xticks(range(1, n_pairs+1))
            ax.set_xticklabels([f"Pair {p}" for p in range(1, n_pairs+1)],
                                color="white", fontsize=12)
            ax.set_xlim(0.4, n_pairs+0.6)
    else:
        for ax in (ax1, ax2):
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.set_xlabel("Track Pair" if plot_style=="bar" else "Track Number")

    # --- Channel balance panel ---
    if pairs:
        pair_nums = np.array([i+1 for i in range(len(pairs))])
        l_levels = [p[0]["rms_main_db"] for p in pairs]
        r_levels = [p[1]["rms_main_db"] for p in pairs]
        diffs    = [l-r for l, r in zip(l_levels, r_levels)]

        if plot_style == "bar":
            bar_w = 0.3
            bl = ax_bal.bar(pair_nums-bar_w/2, l_levels, width=bar_w,
                            color="#4FC3F7", label="L level",
                            alpha=0.88, edgecolor="#333333")
            br = ax_bal.bar(pair_nums+bar_w/2, r_levels, width=bar_w,
                            color="#FF8A65", label="R level",
                            alpha=0.88, edgecolor="#333333")
            for bar, val in zip(bl, l_levels):
                ax_bal.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                            f"{val:.1f}", ha="center", va="bottom",
                            fontsize=LABEL_FONTSIZE, color="white", fontweight="bold")
            for bar, val in zip(br, r_levels):
                ax_bal.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                            f"{val:.1f}", ha="center", va="bottom",
                            fontsize=LABEL_FONTSIZE, color="white", fontweight="bold")
            ax_bal.set_xticks(pair_nums)
            ax_bal.set_xticklabels([f"Pair {p}" for p in pair_nums],
                                   color="white", fontsize=12)
            ax_bal.set_xlim(0.4, len(pairs)+0.6)
        else:
            ax_bal.plot(pair_nums, l_levels, "o-", color="#4FC3F7",
                        label="L level", linewidth=2, markersize=7)
            ax_bal.plot(pair_nums, r_levels, "s-", color="#FF8A65",
                        label="R level", linewidth=2, markersize=7)
            ax_bal.xaxis.set_major_locator(ticker.MultipleLocator(1))

        ax_bal_r = ax_bal.twinx()
        ax_bal_r.plot(pair_nums, diffs, "^--", color="#A5D6A7",
                      label="L−R diff", linewidth=2, markersize=9, zorder=5)
        for x, d in zip(pair_nums, diffs):
            ax_bal_r.text(x, d+(0.02 if d >= 0 else -0.02), f"{d:+.2f} dB",
                          ha="center", va="bottom" if d >= 0 else "top",
                          fontsize=LABEL_FONTSIZE, color="#A5D6A7", fontweight="bold")
        ax_bal_r.axhline(0, color="#A5D6A7", linestyle=":", linewidth=1)
        ax_bal_r.set_ylabel("L−R Difference (dB)", color="#A5D6A7", fontsize=12)
        ax_bal_r.tick_params(axis="y", colors="#A5D6A7")
        ax_bal_r.legend(loc="upper right")

        all_levels = l_levels + r_levels
        mid = (max(all_levels)+min(all_levels))/2
        hs  = min(max(1.0, (max(all_levels)-min(all_levels))/2)+1.0, 5.0)
        ax_bal.set_ylim(mid-hs-1.5, mid+hs)
        ax_bal.set_ylabel("Level (dBFS)")
        ax_bal.set_xlabel("Track Pair" if plot_style=="bar" else "Track Pair")
        ax_bal.set_title("Channel Balance — Bandpass-Filtered RMS Level at Test Tone")
        ax_bal.legend(loc="upper left"); ax_bal.grid(True, alpha=0.3, axis="y")

    for ax in (ax_bal, ax1, ax2):
        for lbl in ax.get_xticklabels()+ax.get_yticklabels():
            lbl.set_color("white"); lbl.set_fontsize(12)

    return fig


def make_table_fig(results, pairs, filename, user_desc):
    plt.style.use("dark_background")
    plt.rcParams.update(RCPARAMS)

    lr = [r for r in results if r["main"]=="L"]
    rl = [r for r in results if r["main"]=="R"]

    def avg(lst, key):
        vals = [r[key] for r in lst if not np.isnan(r[key])]
        return np.mean(vals) if vals else float("nan")

    avg_lr_xt = avg(lr,"xtalk_db"); avg_rl_xt = avg(rl,"xtalk_db")
    avg_lr_ph = avg(lr,"phase_diff_deg"); avg_rl_ph = avg(rl,"phase_diff_deg")

    n_rows = len(results) + len(pairs)
    fig_h = max(6.0, 0.42*(n_rows+8))
    fig = plt.figure(figsize=(13, fig_h))
    fig.suptitle("Results Table", fontsize=15, fontweight="bold", y=0.99)
    fig.text(0.5, 0.96, filename+(f"\n{user_desc}" if user_desc else ""),
             ha="center", va="top", fontsize=11, color="white")

    ax_top = fig.add_axes([0.03, 0.50, 0.94, 0.40])
    ax_bot = fig.add_axes([0.03, 0.02, 0.94, 0.43])
    ax_top.axis("off"); ax_bot.axis("off")

    # Crosstalk & phase table
    rows_top = [[str(r["track"]), f"{r['main']}->{r['leak']}",
                 f"{r['freq']:.0f}", f"{r['xtalk_db']:.2f}", f"{r['phase_diff_deg']:.2f}"]
                for r in results]
    rows_top += [["Avg","L->R","—",f"{avg_lr_xt:.2f}",f"{avg_lr_ph:.2f}"],
                 ["Avg","R->L","—",f"{avg_rl_xt:.2f}",f"{avg_rl_ph:.2f}"]]
    tbl = ax_top.table(cellText=rows_top,
                       colLabels=["Trk","Path","Freq (Hz)","Xtalk (dB)","Phase (deg)"],
                       loc="center", cellLoc="center",
                       colWidths=[0.08,0.10,0.14,0.14,0.14])
    tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1, 1.9)
    n_avg = 2
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#555555")
        if row == 0:
            cell.set_text_props(fontweight="bold", color="white", fontsize=12)
            cell.set_facecolor("#333333")
        elif row > len(rows_top)-n_avg:
            cell.set_text_props(fontweight="bold", color="#FFD54F", fontsize=12)
            cell.set_facecolor("#222222")
        else:
            cell.set_text_props(color="white", fontsize=12)
            cell.set_facecolor("#111111")
    ax_top.set_title("Crosstalk & Phase", color="white",
                     fontsize=13, fontweight="bold", pad=8)

    # Balance table
    rows_bot = []
    for i, (l_r, r_r) in enumerate(pairs):
        diff = l_r["rms_main_db"]-r_r["rms_main_db"]
        rows_bot.append([str(i+1), str(l_r["track"]), f"{l_r['rms_main_db']:.2f}",
                         str(r_r["track"]), f"{r_r['rms_main_db']:.2f}", f"{diff:+.2f}"])
    if rows_bot:
        avg_diff = np.mean([float(r[5]) for r in rows_bot])
        rows_bot.append(["Avg","—","—","—","—",f"{avg_diff:+.2f}"])
    tbl2 = ax_bot.table(cellText=rows_bot,
                        colLabels=["Pair","Trk L","L Level (dBFS)",
                                   "Trk R","R Level (dBFS)","L−R Diff (dB)"],
                        loc="center", cellLoc="center",
                        colWidths=[0.07,0.07,0.15,0.07,0.15,0.14])
    tbl2.auto_set_font_size(False); tbl2.set_fontsize(12); tbl2.scale(1, 1.9)
    for (row, col), cell in tbl2.get_celld().items():
        cell.set_edgecolor("#555555")
        if row == 0:
            cell.set_text_props(fontweight="bold", color="white", fontsize=12)
            cell.set_facecolor("#333333")
        elif rows_bot and row == len(rows_bot):
            cell.set_text_props(fontweight="bold", color="#FFD54F", fontsize=12)
            cell.set_facecolor("#222222")
        else:
            try:
                dv = abs(float(rows_bot[row-1][5]))
            except (ValueError, IndexError):
                dv = 0
            if col == 5 and dv > 0.5:
                cell.set_text_props(color="#FF8A65", fontsize=12, fontweight="bold")
            else:
                cell.set_text_props(color="white", fontsize=12)
            cell.set_facecolor("#111111")
    ax_bot.set_title("Channel Balance (bandpass-filtered RMS at test tone)",
                     color="white", fontsize=13, fontweight="bold", pad=8)
    return fig


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# -----------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------

uploaded = st.file_uploader(
    "Upload a stereo WAV file",
    type=["wav"],
    help="Alternating L/R channel tracks — abrupt transitions, no silence needed.",
)

if uploaded is not None:
    with st.spinner("Reading WAV file…"):
        wav_bytes = uploaded.read()
        sr, data = wavfile.read(io.BytesIO(wav_bytes))

    if data.ndim != 2 or data.shape[1] != 2:
        st.error("Please upload a **stereo** (2-channel) WAV file.")
        st.stop()

    dur_s = data.shape[0] / sr
    st.success(
        f"✅ **{uploaded.name}** — {sr} Hz, {data.dtype}, {dur_s:.1f} s, "
        f"{data.shape[0]:,} samples"
    )

    L = data[:, 0].astype(np.float64)
    R = data[:, 1].astype(np.float64)

    # Full-scale value for dBFS, from the file's actual sample format:
    #   8-bit unsigned (centred on 128), 16/24/32-bit integer, or 32/64-bit float.
    # 24-bit PCM is read by scipy as left-justified int32, so it uses the int32 scale.
    if data.dtype == np.uint8:
        L -= 128.0
        R -= 128.0
        full_scale = 127.0
    elif np.issubdtype(data.dtype, np.integer):
        full_scale = float(np.iinfo(data.dtype).max)
    else:
        full_scale = 1.0

    with st.spinner("Detecting and analysing tracks…"):
        results, pairs = run_analysis(
            L, R, sr, trim_frac, analysis_max_s, track_min_dur, full_scale
        )

    if not results:
        st.error(
            "No tracks detected. Try lowering **Min track duration** in "
            "the sidebar, or check that the file contains alternating L/R tracks."
        )
        st.stop()

    st.info(f"Detected **{len(results)} tracks** → **{len(pairs)} L/R pairs**.")

    with st.spinner("Building charts…"):
        fig_charts = make_charts(
            results, pairs, uploaded.name, user_description,
            plot_style_key, xtalk_ylim, phase_ylim,
        )
        fig_table = make_table_fig(
            results, pairs, uploaded.name, user_description
        )

    st.subheader("Charts")
    st.pyplot(fig_charts)

    st.subheader("Results Table")
    st.pyplot(fig_table)

    # Native dataframe for easy copy/paste
    lr = [r for r in results if r["main"]=="L"]
    rl = [r for r in results if r["main"]=="R"]

    st.subheader("Crosstalk & Phase — copy-pasteable")
    df_xt = pd.DataFrame([{
        "Track": r["track"], "Path": f"{r['main']}->{r['leak']}",
        "Freq (Hz)": f"{r['freq']:.0f}",
        "Xtalk (dB)": f"{r['xtalk_db']:.2f}",
        "Phase (deg)": f"{r['phase_diff_deg']:.2f}",
        "Start (s)": f"{r['t_start']:.1f}", "End (s)": f"{r['t_end']:.1f}",
    } for r in results])
    st.dataframe(df_xt, use_container_width=True, hide_index=True)

    st.subheader("Channel Balance — copy-pasteable")
    rows_bal = []
    for i, (l_r, r_r) in enumerate(pairs):
        diff = l_r["rms_main_db"] - r_r["rms_main_db"]
        rows_bal.append({
            "Pair": i+1, "Trk L": l_r["track"],
            "L Level (dBFS)": f"{l_r['rms_main_db']:.2f}",
            "Trk R": r_r["track"],
            "R Level (dBFS)": f"{r_r['rms_main_db']:.2f}",
            "L−R Diff (dB)": f"{diff:+.2f}",
        })
    if rows_bal:
        st.dataframe(pd.DataFrame(rows_bal), use_container_width=True, hide_index=True)

    # Downloads
    st.subheader("⬇️ Download")
    c1, c2, c3 = st.columns(3)
    stem = uploaded.name.rsplit(".", 1)[0]
    with c1:
        st.download_button("Charts (PNG)", fig_to_png(fig_charts),
                           f"{stem}_charts.png", "image/png")
    with c2:
        st.download_button("Table (PNG)", fig_to_png(fig_table),
                           f"{stem}_table.png", "image/png")
    with c3:
        csv = df_xt.to_csv(index=False).encode()
        st.download_button("Data (CSV)", csv,
                           f"{stem}_results.csv", "text/csv")

else:
    st.info("👆 Upload a WAV file to get started.")
    st.markdown("""
**What this app analyses:**
- Alternating L/R channel test records (Shure TTR-109 and similar)
- No silence gaps needed between tracks — abrupt channel switches are handled
- Automatically detects each track and measures:
  - **Crosstalk (dB)** — leakage from the driven into the silent channel
  - **Phase difference (deg)** — between leakage and main signal
  - **Channel balance (dBFS)** — bandpass-filtered RMS at the test tone
    (rumble and RIAA-boosted low-frequency noise are rejected before measurement)

**How to use:**
1. Upload your stereo WAV recording
2. Optionally adjust detection settings in the sidebar
3. Download the chart and table PNGs, or copy the CSV data
""")
