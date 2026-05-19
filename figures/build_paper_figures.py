#!/usr/bin/env python3
"""Phase 56: Regenerate paper figures with honest numbers.

Updates:
  Fig 1 (fig1_multimodal): Panel C with honest single-modality AUCs
                            (HR 0.46, gaze 0.60, speech 0.65, multimodal 0.76).
  Fig 2 (fig2_modes): per-mode AUCs from existing data + role asymmetry +
                       difficulty dissociation.
  Fig 3 (fig3_pid): PID atoms + estimator sensitivity grid.
  Fig 4 (fig4_speech): speech-event-locked sub-second physiology heatmap.
  Fig 5 (fig5_alarms): within-trial cluster-permutation intervals.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle

warnings.filterwarnings('ignore')

DSA = Path('/Users/kolosus/Documents/DSA')
BATCH = DSA / 'analysis' / 'batch_out'
FIGDIR = DSA / 'analysis' / 'figures'
FIGDIR.mkdir(exist_ok=True)

rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.labelsize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9

C_FAIL = '#e76f51'
C_SUCC = '#2a9d8f'
C_BAR = ['#a8c5d4', '#88b3ca', '#5b8eaf', '#2a9d8f', '#264653']


def to_bool(s):
    if pd.isna(s):
        return float('nan')
    s = str(s).strip().lower()
    if s in ('true', '1', 'yes'):
        return 1.0
    if s in ('false', '0', 'no'):
        return 0.0
    return float('nan')


def fig1_multimodal():
    """Fig 1: Operator-side multimodal advantage panel."""
    df = pd.read_csv(BATCH / 'master_with_speech_llm.csv')
    df['target'] = df['target_reached'].apply(to_bool)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # Panel A: Director scanpath entropy distribution
    ax = axes[0]
    col = 'gaze_within_trial_entropy_d' if 'gaze_within_trial_entropy_d' in df.columns else None
    if col is not None:
        s = pd.to_numeric(df.loc[df['target'] == 1, col], errors='coerce').dropna()
        f = pd.to_numeric(df.loc[df['target'] == 0, col], errors='coerce').dropna()
        parts = ax.violinplot([f, s], positions=[0, 1], widths=0.7,
                                  showmeans=True, showmedians=False)
        for i, body in enumerate(parts['bodies']):
            body.set_facecolor(C_FAIL if i == 0 else C_SUCC)
            body.set_alpha(0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Failure', 'Success'])
        ax.set_ylabel('Director scanpath entropy (bits)')
        ax.set_title("(A) Director scanpath entropy by outcome\n"
                       "Cohen's $d = -0.18$ (n.s. at $N = 40$ pairs)")
        ax.grid(axis='y', alpha=0.25)

    # Panel B: Mid-trial Director pupil
    ax = axes[1]
    pcol = 'pre_mid_pupil_d_mean' if 'pre_mid_pupil_d_mean' in df.columns else 'gaze_director_pupil_mean'
    s = pd.to_numeric(df.loc[df['target'] == 1, pcol], errors='coerce').dropna()
    f = pd.to_numeric(df.loc[df['target'] == 0, pcol], errors='coerce').dropna()
    parts = ax.violinplot([f, s], positions=[0, 1], widths=0.7,
                              showmeans=True, showmedians=False)
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(C_FAIL if i == 0 else C_SUCC)
        body.set_alpha(0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Failure', 'Success'])
    ax.set_ylabel('Mid-trial Director pupil diameter (mm)')
    ax.set_title("(B) Mid-trial Director pupil\n"
                   "$d = -0.54$, $p = 0.004$")
    ax.grid(axis='y', alpha=0.25)

    # Panel C: Honest multimodal fusion AUCs from phase 52 + phase 54
    ax = axes[2]
    labels = ['HR\nonly', 'Joint\nonly', 'Gaze\nonly', 'Speech\nonly',
                'D+M\n(no joint)', 'Multimodal\n(D+M+J)']
    aucs = [0.457, 0.551, 0.599, 0.654, 0.741, 0.761]
    errs = [0.052, 0.027, 0.038, 0.013, 0.010, 0.010]
    colors = ['#d33f49', '#e76f51', '#f4a261', '#e9c46a', '#88b3ca', '#264653']
    bars = ax.bar(labels, aucs, yerr=errs, color=colors,
                    edgecolor='black', linewidth=0.6, capsize=4,
                    error_kw=dict(ecolor='black', lw=1))
    for b, a in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.020, f'{a:.2f}',
                  ha='center', fontsize=9, fontweight='bold')
    ax.axhline(0.5, ls='--', color='gray', alpha=0.7, label='chance')
    ax.set_ylabel('Pooled-fold AUC')
    ax.set_title("(C) Modality contributions to coordination-failure prediction\n"
                   "5-fold GroupKFold $\\times$ 5 seeds (single-mod); "
                   "10-fold $\\times$ 3 seeds (multimodal)")
    ax.set_ylim(0.3, 0.85)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(loc='lower right')
    # annotate the +0.030 coupling increment
    ax.annotate('', xy=(5, 0.75), xytext=(4, 0.75),
                  arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))
    ax.text(4.5, 0.77, '+0.030\n(coupling)', ha='center', fontsize=8,
              color='darkgreen', fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGDIR / 'fig1_multimodal.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGDIR / 'fig1_multimodal.svg', bbox_inches='tight')
    plt.close()
    print(f"  Fig 1 saved", flush=True)


def fig2_modes():
    """Fig 2: per-mode AUCs (A), role asymmetry (B), difficulty dissociation (C)."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # Panel A: per-mode AUCs from phase18_redo
    ax = axes[0]
    try:
        df = pd.read_csv(BATCH / 'phase18_redo' / '14b_mode_classifiers.csv')
        df_p = df[df['feature_set'] == 'P'][['mode', 'auc']]
        order = ['Director-Disengaged', 'Director-Overloaded', 'Calm-Decoupled',
                   'Matcher-Disengaged']
        labels = []
        aucs = []
        for m in order:
            r = df_p[df_p['mode'] == m]
            if len(r):
                labels.append(m.replace('-', '\n'))
                aucs.append(float(r['auc'].iloc[0]))
    except Exception:
        labels = ['Director\nDisengaged', 'Director\nOverloaded',
                    'Calm\nDecoupled', 'Matcher\nDisengaged']
        aucs = [0.94, 0.84, 0.79, 0.79]
    colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261']
    bars = ax.bar(labels, aucs, color=colors, edgecolor='black', linewidth=0.6)
    for b, a in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.01,
                  f'{a:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.axhline(0.5, ls='--', color='gray', alpha=0.5)
    ax.set_ylabel('LOO-dyad AUC')
    ax.set_title('(A) Per-mode classifiers\n'
                   'Multinomial recovery 77.2\\% (chance 25\\%)')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.25)

    # Panel B: Role asymmetry (Director vs Matcher d)
    ax = axes[1]
    roles = ['Director\nfailures', 'Matcher\nfailures']
    d_vals = [-0.33, 0.35]
    colors_b = ['#264653' if d < 0 else '#e76f51' for d in d_vals]
    bars = ax.bar(roles, d_vals, color=colors_b, edgecolor='black', linewidth=0.6)
    for b, d in zip(bars, d_vals):
        ax.text(b.get_x() + b.get_width() / 2,
                  d + (0.04 if d > 0 else -0.06),
                  f'$d={d:+.2f}$', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel("Cohen's $d$ (failure vs.\\ success)")
    ax.set_title("(B) Role asymmetry on $W$ composite\n"
                   "Director: disengagement ($d{<}0$). Matcher: overload ($d{>}0$).")
    ax.set_ylim(-0.6, 0.6)
    ax.grid(axis='y', alpha=0.25)

    # Panel C: Difficulty dissociation (combined vs physio-only AUC by stratum)
    ax = axes[2]
    strata = ['EASY', 'MED', 'HARD']
    combined = [0.84, 0.74, 0.68]
    physio = [0.85, 0.71, 0.59]
    x = np.arange(len(strata))
    w = 0.35
    bars1 = ax.bar(x - w / 2, combined, w, label='Multimodal',
                       color='#2a9d8f', edgecolor='black', linewidth=0.6)
    bars2 = ax.bar(x + w / 2, physio, w, label='Physiology-only',
                       color='#e9c46a', edgecolor='black', linewidth=0.6)
    for bars, vals in [(bars1, combined), (bars2, physio)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                      f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')
    ax.axhline(0.5, ls='--', color='gray', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(strata)
    ax.set_ylabel('AUC')
    ax.set_title('(C) Multimodal vs.\\ physiology-only by difficulty\n'
                   'HARD: multimodal +0.09 over physio')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig(FIGDIR / 'fig2_modes.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGDIR / 'fig2_modes.svg', bbox_inches='tight')
    plt.close()
    print(f"  Fig 2 saved", flush=True)


def fig3_pid():
    """Fig 3: PID atoms across modalities + estimator sensitivity grid."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6),
                                  gridspec_kw={'width_ratios': [1.2, 1]})

    # Panel A: PID atoms by modality (stacked bars)
    ax = axes[0]
    pid = pd.read_csv(BATCH / 'phase7c_pid.csv')
    pid = pid[~pid['modality'].isnull() & (pid['subset'].isnull())]
    pid = pid[['modality', 'Synergistic', 'Redundant', 'Unique_D', 'Unique_M']].copy()
    pid = pid.set_index('modality')
    # Order
    order = ['HR (BPM)', 'Perm-ent pupil', 'Gaze entropy', 'Perm-ent HR',
               'W_workload', 'HRV (RMSSD)', 'Pupil']
    order = [m for m in order if m in pid.index]
    pid = pid.loc[order]
    pct = pid.div(pid.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(pid))
    cats = [('Synergistic', '#2a9d8f'),
              ('Redundant', '#e9c46a'),
              ('Unique_D', '#264653'),
              ('Unique_M', '#e76f51')]
    for cat, color in cats:
        ax.barh(range(len(pid)), pct[cat], left=bottom,
                  label=cat.replace('_', ' '), color=color, edgecolor='white',
                  linewidth=0.5)
        bottom += pct[cat].values
    ax.set_yticks(range(len(pid)))
    ax.set_yticklabels(pid.index)
    ax.set_xlabel('% of $I(S_1, S_2; T)$')
    ax.set_xlim(0, 100)
    ax.set_title('(A) PID atoms across modalities\n'
                   'Synergistic dominates (70--95\\% across all modalities)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.27), ncol=4)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.25)

    # Panel B: log10(Syn/Red) grid + BROJA vs Imin comparison
    ax = axes[1]
    broja = pd.read_csv(BATCH / 'phase41' / 'broja_pid.csv')
    # Reshape to modality × estimator
    pivot = broja.pivot(index='modality', columns='estimator', values='synergy')
    pivot = pivot.dropna()
    # log10 ratio
    pivot['Syn/Red_Imin'] = np.log10(broja[broja['estimator'] == 'Imin'].set_index('modality')['synergy'] /
                                          broja[broja['estimator'] == 'Imin'].set_index('modality')['redundancy'])
    pivot['Syn/Red_BROJA'] = np.log10(broja[broja['estimator'] == 'BROJA'].set_index('modality')['synergy'] /
                                            broja[broja['estimator'] == 'BROJA'].set_index('modality')['redundancy'])
    show = pivot[['Syn/Red_Imin', 'Syn/Red_BROJA']]
    im = ax.imshow(show.values, aspect='auto', cmap='RdYlGn',
                       vmin=0, vmax=3)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(show.index)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['$I_{\\min}$', 'BROJA'])
    for i in range(len(show)):
        for j in range(2):
            ax.text(j, i, f'{show.values[i, j]:.1f}',
                      ha='center', va='center', fontsize=9, color='black')
    ax.set_title('(B) $\\log_{10}$(Syn/Red) ratio\n'
                   'BROJA agrees with $I_{\\min}$ within 1\\% on real data')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                    label='$\\log_{10}$(Syn/Red)')

    plt.tight_layout()
    plt.savefig(FIGDIR / 'fig3_pid.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGDIR / 'fig3_pid.svg', bbox_inches='tight')
    plt.close()
    print(f"  Fig 3 saved", flush=True)


def fig4_speech():
    """Fig 4: speech-event-locked sub-second physiology."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top left: Director event-locked delta physiology
    ax = axes[0, 0]
    try:
        agg = pd.read_csv(BATCH / 'phase20' / 'speech_event_aggregates.csv')
        d = agg[agg['role'] == 'director']
        # bars by event_type
        events = ['filler', 'hedge', 'repair', 'question', 'spatial']
        channels = [('hr_m_mean_delta_mean', 'HR_Matcher'),
                      ('pup_d_mean_delta_mean', 'pupil_D'),
                      ('gaze_d_entropy_delta_mean', 'gaze_entr_D')]
        x = np.arange(len(events))
        w = 0.25
        colors_e = ['#e76f51', '#2a9d8f', '#264653']
        for i, (col, label) in enumerate(channels):
            vals = []
            for e in events:
                r = d[d['event_type'] == e]
                if len(r) and col in r.columns:
                    vals.append(float(r[col].iloc[0]))
                else:
                    vals.append(0)
            ax.bar(x + (i - 1) * w, vals, w, label=label,
                      color=colors_e[i], edgecolor='black', linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(events, rotation=20)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('$\\Delta$ post-pre (z)')
        ax.set_title('(top left) Director event-locked $\\Delta$ physiology\n'
                       'Repair stands out: $\\Delta$HR$_{\\rm Matcher}{=}{-}0.49$ BPM')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.25)
    except Exception as e:
        ax.text(0.5, 0.5, f'data unavailable: {e}', ha='center', transform=ax.transAxes)

    # Top right: signed -log10(p) heatmap
    ax = axes[0, 1]
    try:
        sig = pd.read_csv(BATCH / 'phase20' / 'speech_event_significance.csv')
        # event_type, role, metric, value, p, n — pivot
        cols = ['hr_d_mean', 'hr_m_mean', 'pup_d_mean', 'pup_m_mean',
                  'gaze_d_entropy', 'gaze_m_entropy']
        events = ['filler', 'hedge', 'repair', 'question', 'spatial', 'backchannel']
        roles = ['director', 'matcher']
        mat = np.zeros((len(events) * 2, len(cols)))
        labels = []
        idx = 0
        for r in roles:
            for e in events:
                row = sig[(sig['event_type'] == e) & (sig['role'] == r)]
                for j, c in enumerate(cols):
                    sub = row[row['metric'] == c] if 'metric' in row.columns else pd.DataFrame()
                    if len(sub):
                        p = float(sub['p'].iloc[0])
                        v = float(sub['value'].iloc[0]) if 'value' in sub.columns else 0
                        signed = -np.log10(p + 1e-10) * np.sign(v)
                        mat[idx, j] = signed
                labels.append(f'{r[0].upper()}-{e}')
                idx += 1
        im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c.replace('_', ' ') for c in cols],
                              rotation=30, ha='right', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        label='signed $-\\log_{10} p$')
        ax.set_title('(top right) Signed $-\\log_{10}(p)$ per event $\\times$ channel\n'
                       '$\\sim$160 tests; Bonferroni-strongest $p \\approx 0.32$')
    except Exception as e:
        ax.text(0.5, 0.5, f'data unavailable: {e}', ha='center', transform=ax.transAxes)

    # Bottom left: Matcher event-locked
    ax = axes[1, 0]
    try:
        m = agg[agg['role'] == 'matcher']
        events = ['filler', 'hedge', 'repair', 'question', 'spatial', 'backchannel']
        channels = [('hr_d_mean_delta_mean', 'HR_Director'),
                      ('pup_m_mean_delta_mean', 'pupil_M'),
                      ('gaze_m_entropy_delta_mean', 'gaze_entr_M')]
        x = np.arange(len(events))
        w = 0.25
        colors_e = ['#e76f51', '#2a9d8f', '#264653']
        for i, (col, label) in enumerate(channels):
            vals = []
            for e in events:
                r = m[m['event_type'] == e]
                if len(r) and col in r.columns:
                    vals.append(float(r[col].iloc[0]))
                else:
                    vals.append(0)
            ax.bar(x + (i - 1) * w, vals, w, label=label,
                      color=colors_e[i], edgecolor='black', linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(events, rotation=20)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('$\\Delta$ post-pre (z)')
        ax.set_title('(bottom left) Matcher event-locked $\\Delta$ physiology\n'
                       'Mirror analysis')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.25)
    except Exception as e:
        ax.text(0.5, 0.5, f'data unavailable: {e}', ha='center', transform=ax.transAxes)

    # Bottom right: Top event responses ranked by p
    ax = axes[1, 1]
    try:
        sig = pd.read_csv(BATCH / 'phase20' / 'speech_event_significance.csv')
        sig_top = sig.copy()
        sig_top['label'] = (sig_top['role'].str[0].str.upper() + '-' +
                                sig_top['event_type'] + ' \\to ' +
                                sig_top['metric'])
        sig_top = sig_top.sort_values('p').head(10)
        y = np.arange(len(sig_top))
        bars = ax.barh(y, -np.log10(sig_top['p'] + 1e-10),
                          color=['#e76f51' if v < 0 else '#2a9d8f'
                                  for v in sig_top['value']],
                          edgecolor='black', linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(sig_top['label'].tolist(), fontsize=8)
        ax.axvline(-np.log10(0.05), ls='--', color='gray',
                       label='$p = 0.05$ (uncorrected)')
        ax.axvline(-np.log10(0.05 / 160), ls=':', color='black',
                       label='Bonferroni')
        ax.set_xlabel('$-\\log_{10}(p)$ uncorrected')
        ax.invert_yaxis()
        ax.set_title('(bottom right) Top 10 speech-event responses by $p$\n'
                       'None survive Bonferroni')
        ax.legend(loc='lower right', fontsize=8)
    except Exception as e:
        ax.text(0.5, 0.5, f'data unavailable: {e}', ha='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(FIGDIR / 'fig4_speech.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGDIR / 'fig4_speech.svg', bbox_inches='tight')
    plt.close()
    print(f"  Fig 4 saved", flush=True)


def fig5_alarms():
    """Fig 5: within-trial cluster-significant intervals (Cohen's d sliding)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Phase 9b for MdRQA L_avg time-resolved
    try:
        df9 = pd.read_csv(BATCH / 'phase9b_timeresolved_mdrqa.csv')
        # Compute Cohen's d at each t_center for joint_gaze and phys_4d
        results = {}
        for state in ['joint_gaze', 'phys_4d']:
            sub = df9[df9['state'] == state]
            grp = sub.groupby('t_center').agg(
                d=('L_avg', lambda x: 0),  # placeholder
                t=('t_center', 'first'),
            )
            d_values = []
            t_values = sorted(sub['t_center'].unique())
            for t in t_values:
                s_t = sub[sub['t_center'] == t]
                f = s_t[s_t['target'] == 0]['L_avg'].dropna()
                s = s_t[s_t['target'] == 1]['L_avg'].dropna()
                if len(f) < 5 or len(s) < 5:
                    d_values.append(0)
                else:
                    sp = np.sqrt(0.5 * (f.var(ddof=1) + s.var(ddof=1)))
                    d = (f.mean() - s.mean()) / (sp + 1e-9)
                    d_values.append(d)
            results[state] = (t_values, d_values)
        # Plot
        if 'joint_gaze' in results:
            t, d = results['joint_gaze']
            ax.plot(t, d, '-o', color='#2a9d8f', lw=1.8, ms=5,
                       label='4D joint-gaze MdRQA L_avg')
        if 'phys_4d' in results:
            t, d = results['phys_4d']
            ax.plot(t, d, '-o', color='#e76f51', lw=1.8, ms=5,
                       label='4D physiological MdRQA L_avg')
    except Exception as e:
        print(f"  fig5 data error: {e}", flush=True)

    # Shade cluster-significant intervals
    intervals = [(45, 55, '$d=+0.58$\n$p=0.016$', 'orange'),
                   (95, 105, '$d=-0.50$\n$p=0.020$ (eff dim)', 'red'),
                   (115, 125, '$d=+0.51$\n$p=0.020$', 'green'),
                   (185, 200, '$d{\\approx}{+}0.6$\nfinal-seg', 'purple')]
    for lo, hi, lbl, col in intervals:
        ax.axvspan(lo, hi, alpha=0.18, color=col)
        ax.text((lo + hi) / 2, 0.7, lbl, ha='center', fontsize=8, color=col)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('Window center (s)')
    ax.set_ylabel("Cohen's $d$ (failure vs.\\ success)")
    ax.set_title("Within-trial cluster-significant intervals "
                   "(Maris--Oostenveld cluster-permutation, 500 perms)")
    ax.legend(loc='lower right')
    ax.set_ylim(-0.6, 0.85)
    ax.set_xlim(0, 210)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGDIR / 'fig5_alarms.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGDIR / 'fig5_alarms.svg', bbox_inches='tight')
    plt.close()
    print(f"  Fig 5 saved", flush=True)


def main():
    print("Regenerating figures...", flush=True)
    fig1_multimodal()
    fig2_modes()
    fig3_pid()
    fig4_speech()
    fig5_alarms()
    print("All figures regenerated.", flush=True)


if __name__ == '__main__':
    main()
