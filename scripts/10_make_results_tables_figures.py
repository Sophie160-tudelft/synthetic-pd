from __future__ import annotations
import argparse, zipfile, shutil, textwrap
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FULL_CONDITION_MAP = {
    'E0': ('Room complexity', 'Neutral', 'Bright/day', 'Frontal/table', 'None', 'Neutral technical baseline'),
    'E1': ('Room complexity', 'Living room with table', 'Bright/day', 'Frontal/table', 'None', 'Living-room baseline'),
    'E2': ('Camera view', 'Living room without table', 'Bright/day', 'Frontal static 1.55 m', 'None', 'High frontal static camera'),
    'E3': ('Camera view', 'Living room without table', 'Bright/day', 'Frontal/table 0.95 m', 'None', 'Stable frontal/table setup'),
    'E4': ('Camera view', 'Living room without table', 'Bright/day', 'Upper corner', 'None', 'Upper-corner viewpoint'),
    'E5': ('Camera view', 'Living room without table', 'Bright/day', 'Frontal, subject centred', 'None', 'Centred frontal setup'),
    'E6': ('Lighting', 'Neutral', 'Dim/night', 'Frontal best angle', 'None', 'Dim neutral condition'),
    'E7': ('Lighting', 'Living room without table', 'Dim/night', 'Frontal best angle', 'None', 'Dim living-room condition'),
    'E8': ('Occlusion', 'Living room with table', 'Bright/day', 'Frontal/table', 'Both-leg occlusion, all frames', 'Strong lower-body occlusion'),
    'E9': ('Occlusion', 'Living room with table', 'Bright/day', 'Upper frontal/static', 'No/partial both-leg occlusion', 'Tracking-instability case'),
    'E10': ('Occlusion', 'Living room with table', 'Bright/day', 'Upper corner', 'Partial frame occlusion', 'Upper-corner partial occlusion'),
    'E11': ('Occlusion', 'Living room with table', 'Bright/day', 'Frontal/table', 'One-leg occlusion, partial frames', 'Frontal partial one-leg occlusion'),
    'E12': ('Occlusion', 'Living room with table', 'Bright/day', 'Frontal/static', 'One-leg occlusion, partial frames', 'Stable partial-occlusion setup'),
    'E13': ('Occlusion', 'Living room with table', 'Bright/day', 'Upper corner', 'One-leg occlusion, partial frames', 'Upper-corner one-leg occlusion'),
}

REDUCED_MAP = {
    'R0': ('E0', 'Neutral baseline', 'Neutral', 'Bright/day', 'Frontal/table', 'None'),
    'R1': ('E1', 'Home baseline', 'Living room with table', 'Bright/day', 'Frontal/table', 'None'),
    'R2': ('E2', 'Best/stable frontal setup', 'Living room without table', 'Bright/day', 'Frontal/table 0.95 m', 'None'),
    'R3': ('E3', 'Upper-corner viewpoint', 'Living room without table', 'Bright/day', 'Upper corner', 'None'),
    'R4': ('E4', 'Strong occlusion', 'Living room with table', 'Bright/day', 'Frontal/table', 'Both-leg occlusion'),
    'R5': ('E5', 'Partial occlusion, frontal', 'Living room with table', 'Bright/day', 'Frontal/table', 'One-leg occlusion, partial frames'),
}

PRIMARY_REDUCED_MAP = {'R0':'E0','R1':'E1','R2':'E3','R3':'E4','R4':'E8','R5':'E12'}
REPLICATION_REDUCED_MAP = {'R0':'E0','R1':'E1','R2':'E2','R3':'E3','R4':'E4','R5':'E5'}
PAIRS = [
    ('C1','R1','R0','Home baseline vs neutral baseline','Does room complexity affect recovered motion?'),
    ('C2','R2','R1','Stable frontal setup vs home baseline','Does the selected frontal setup remain close to the home baseline?'),
    ('C3','R3','R2','Upper-corner view vs stable frontal setup','Does the upper-corner viewpoint degrade recovery relative to the frontal setup?'),
    ('C4','R4','R1','Strong occlusion vs home baseline','How much degradation is introduced by strong lower-body occlusion?'),
    ('C5','R5','R1','Partial frontal occlusion vs home baseline','Does partial one-leg occlusion remain recoverable?'),
    ('C6','R5','R4','Partial occlusion vs strong occlusion','Is partial occlusion less harmful than strong both-leg occlusion?'),
]
METRICS = [
    'pa_mpjpe_mean_mm','step_contact_timing_mae_s','cadence_abs_error_steps_per_min',
    'foot_clearance_mean_abs_error_mm','step_length_mean_abs_error_mm','stride_length_mean_abs_error_mm','knee_rom_mean_abs_error_deg'
]
PRETTY = {
    'pa_mpjpe_mean_mm':'PA-MPJPE (mm)',
    'step_contact_timing_mae_s':'Contact timing MAE (s)',
    'cadence_abs_error_steps_per_min':'Cadence error (steps/min)',
    'foot_clearance_mean_abs_error_mm':'Foot clearance error (mm)',
    'step_length_mean_abs_error_mm':'Step length error (mm)',
    'stride_length_mean_abs_error_mm':'Stride length error (mm)',
    'knee_rom_mean_abs_error_deg':'Knee ROM error (deg)'
}

def unzip_if_needed(zip_path: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out)
    return out

def latex_escape(x):
    if pd.isna(x): return ''
    s=str(x)
    repl={'&':'\\&','%':'\\%','$':'\\$','#':'\\#','_':'\\_','{':'\\{','}':'\\}','~':'\\textasciitilde{}','^':'\\textasciicircum{}'}
    return ''.join(repl.get(c,c) for c in s)

def fmt(x, nd=2):
    if pd.isna(x): return ''
    if isinstance(x, (int, np.integer)): return str(int(x))
    try:
        f=float(x)
        if abs(f - round(f)) < 1e-9: return str(int(round(f)))
        return f'{f:.{nd}f}'
    except Exception:
        return latex_escape(x)

def table_tex(df, caption, label, cols=None, nd=2, resize=True):
    if cols is not None:
        df=df[cols].copy()
    lines=[]
    align='l' * len(df.columns)
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append(f'\\caption{{{latex_escape(caption)}}}')
    lines.append(f'\\label{{{label}}}')
    if resize: lines.append('\\resizebox{\\textwidth}{!}{%')
    lines.append(f'\\begin{{tabular}}{{{align}}}')
    lines.append('\\toprule')
    lines.append(' & '.join(latex_escape(c) for c in df.columns) + ' \\\\')
    lines.append('\\midrule')
    for _,r in df.iterrows():
        lines.append(' & '.join(fmt(r[c], nd) for c in df.columns) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}%')
    if resize: lines.append('}')
    lines.append('\\end{table}')
    return '\n'.join(lines)

def load_metric_summaries(metrics_root: Path, seqs: pd.DataFrame):
    base = metrics_root / 'clinically_relevant_metrics_v1' / 'batch_wham_experiments'
    out={}
    for _,s in seqs.iterrows():
        sub=s['subject']; seq=s['sequence']
        p=base/sub/seq/f'{sub}_{seq}_all_experiments_metric_summary.csv'
        if not p.exists():
            raise FileNotFoundError(p)
        df=pd.read_csv(p)
        df['subject']=sub; df['sequence']=seq
        out[sub]=df
    return out

def load_validation(render_root: Path, seqs: pd.DataFrame):
    base = render_root / 'render_input_validation'
    rows=[]
    for _,s in seqs.iterrows():
        sub=s['subject']; seq=s['sequence']
        for stage, folder, prefix in [
            ('SMPL to SMPL-X render input', f'{sub}_{seq}', 'render_input_motion_validation_recommended_final_recommended_metrics.csv'),
            ('Coordinate correction', f'root_rotation_effect_{seq}', 'root_rotation_effect_recommended_final_recommended_metrics.csv')]:
            p=base/folder/prefix
            df=pd.read_csv(p)
            d={'subject':sub,'sequence':seq,'stage':stage}
            for _,r in df.iterrows():
                d[r['metric']]=r['value']
            rows.append(d)
    val=pd.DataFrame(rows)
    return val

def build_reduced(summary_by_subject, seqs):
    rows=[]
    for _,s in seqs.iterrows():
        sub=s['subject']; seq=s['sequence']
        df=summary_by_subject[sub]
        mapping = PRIMARY_REDUCED_MAP if sub == 'SUB01' else REPLICATION_REDUCED_MAP
        for rid,eid in mapping.items():
            if eid not in set(df['experiment']):
                continue
            r=df[df['experiment']==eid].iloc[0]
            row={'subject':sub,'sequence':seq,'reduced_id':rid,'source_experiment':eid,'condition':REDUCED_MAP[rid][1],
                 'frames_used':r.get('frames_used', np.nan),'gt_events':r.get('gt_num_gait_events', np.nan),'wham_events':r.get('wham_num_gait_events', np.nan)}
            for m in METRICS:
                row[m]=r.get(m, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)

def build_pairwise(reduced):
    rows=[]
    for sub,grp in reduced.groupby('subject'):
        d=grp.set_index('reduced_id')
        for cid,test,base,name,question in PAIRS:
            if test not in d.index or base not in d.index: continue
            row={'subject':sub,'comparison':cid,'comparison_name':name,'question':question,'test':test,'baseline':base,
                 'test_events':d.loc[test,'wham_events'],'baseline_events':d.loc[base,'wham_events']}
            for m in METRICS:
                row[f'delta_{m}']=d.loc[test,m]-d.loc[base,m]
            rows.append(row)
    return pd.DataFrame(rows)

def plot_bar(df, x, y, title, ylabel, path, order=None, hue=None, width=9, height=5):
    fig, ax = plt.subplots(figsize=(width,height))
    if hue is None:
        if order is not None:
            df=df.set_index(x).loc[order].reset_index()
        ax.bar(df[x].astype(str), df[y])
    else:
        cats=order or list(df[x].drop_duplicates())
        hues=list(df[hue].drop_duplicates())
        pos=np.arange(len(cats)); bw=0.8/max(1,len(hues))
        for i,h in enumerate(hues):
            vals=[]
            for c in cats:
                sub=df[(df[x]==c)&(df[hue]==h)]
                vals.append(float(sub[y].iloc[0]) if len(sub) else np.nan)
            ax.bar(pos+(i-(len(hues)-1)/2)*bw, vals, bw, label=h)
        ax.set_xticks(pos); ax.set_xticklabels(cats)
        ax.legend(fontsize=8)
    ax.axhline(0, linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)

def plot_line_frame(metrics_base, subject, sequence, exp_map, out_path):
    fig, ax=plt.subplots(figsize=(10,4.8))
    root=metrics_base/subject/sequence
    for label,eid in exp_map.items():
        candidates=list((root/eid).glob('*per_frame_metrics.csv'))
        if not candidates: continue
        df=pd.read_csv(candidates[0])
        ax.plot(df['frame'], df['pa_mpjpe_mm'], label=label, linewidth=1.2)
    ax.set_title(f'Frame-level PA-MPJPE: {subject}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('PA-MPJPE (mm)')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path,dpi=220); plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--metrics-zip', default='/mnt/data/clinically_relevant_metrics_v1.zip')
    ap.add_argument('--render-zip', default='/mnt/data/render_input_validation.zip')
    ap.add_argument('--sequences-csv', default='/mnt/data/selected_first_four_bmclab_sequences.csv')
    ap.add_argument('--out-dir', default='/mnt/data/updated_results_pack')
    args=ap.parse_args()
    out=Path(args.out_dir); shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    work=out/'_work'; work.mkdir()
    metrics_root=unzip_if_needed(Path(args.metrics_zip), work/'metrics')
    render_root=unzip_if_needed(Path(args.render_zip), work/'render')
    seqs=pd.read_csv(args.sequences_csv)
    summary_by_subject=load_metric_summaries(metrics_root, seqs)
    validation=load_validation(render_root, seqs)
    reduced=build_reduced(summary_by_subject, seqs)
    pairwise=build_pairwise(reduced)
    tables_dir=out/'tables'; figs_dir=out/'figures'; tables_dir.mkdir(); figs_dir.mkdir()
    # Tables
    seq_table=seqs[['subject','sequence','UPDRS_GAIT','freezer_group','medication','frames','fps','duration_s']].rename(columns={'subject':'Subject','sequence':'Sequence','UPDRS_GAIT':'MDS-UPDRS gait','freezer_group':'Freezer group','medication':'Medication','frames':'Frames','fps':'FPS','duration_s':'Duration (s)'})
    seq_table['Duration (s)']=seq_table['Duration (s)'].map(lambda x: round(float(x),2))
    (tables_dir/'table_1_sequences.tex').write_text(table_tex(seq_table,'Selected walking sequences used in the results.','tab:result_sequences'), encoding='utf-8')
    # validation compact
    val_rows=[]
    for (stage),g in validation.groupby('stage'):
        val_rows.append({'Validation stage':stage,
                         'Mean PA-MPJPE (mm)':g['PA-MPJPE'].mean(), 'Max PA-MPJPE (mm)':g['PA-MPJPE'].max(),
                         'Mean cadence error':g['Cadence error'].mean(), 'Max foot clearance error (mm)':g['Foot clearance error'].max(),
                         'Interpretation':'Motion preserved before WHAM' if 'SMPL' in stage else 'Coordinate correction did not materially distort gait'})
    val_table=pd.DataFrame(val_rows)
    (tables_dir/'table_2_validation.tex').write_text(table_tex(val_table,'Pipeline-validation summary across the selected walking sequences.','tab:pipeline_validation_updated'), encoding='utf-8')
    full_cond=pd.DataFrame([{'ID':k,'Factor':v[0],'Environment':v[1],'Lighting':v[2],'Camera view':v[3],'Occlusion':v[4],'Role':v[5]} for k,v in FULL_CONDITION_MAP.items()])
    (tables_dir/'table_3_full_matrix.tex').write_text(table_tex(full_cond,'Full experiment matrix for the primary sequence.','tab:full_matrix_results'), encoding='utf-8')
    # SUB01 full key
    sub01=summary_by_subject['SUB01'].copy()
    sub01['Condition']=[FULL_CONDITION_MAP[e][5] for e in sub01['experiment']]
    sub01_table=sub01[['experiment','Condition','frames_used','pa_mpjpe_mean_mm','gt_num_gait_events','wham_num_gait_events','foot_clearance_mean_abs_error_mm','step_length_mean_abs_error_mm','stride_length_mean_abs_error_mm','knee_rom_mean_abs_error_deg']]
    sub01_table=sub01_table.rename(columns={'experiment':'ID','frames_used':'Frames','pa_mpjpe_mean_mm':'PA-MPJPE','gt_num_gait_events':'GT events','wham_num_gait_events':'WHAM events','foot_clearance_mean_abs_error_mm':'Foot clearance error','step_length_mean_abs_error_mm':'Step length error','stride_length_mean_abs_error_mm':'Stride length error','knee_rom_mean_abs_error_deg':'Knee ROM error'})
    (tables_dir/'table_4_sub01_full_metrics.tex').write_text(table_tex(sub01_table,'Main WHAM recovery metrics for the full primary experiment matrix.','tab:sub01_full_metrics'), encoding='utf-8')
    red_cond=pd.DataFrame([{'Reduced ID':rid,'Output ID':eid,'Purpose':purpose,'Environment':env,'Lighting':light,'Camera view':cam,'Occlusion':occ} for rid,(eid,purpose,env,light,cam,occ) in REDUCED_MAP.items()])
    (tables_dir/'table_5_reduced_matrix.tex').write_text(table_tex(red_cond,'Reduced experiment matrix used for the additional walking sequences.','tab:reduced_matrix_results'), encoding='utf-8')
    red_table=reduced[['subject','reduced_id','source_experiment','condition','frames_used','pa_mpjpe_mean_mm','gt_events','wham_events','foot_clearance_mean_abs_error_mm','step_length_mean_abs_error_mm','stride_length_mean_abs_error_mm']].rename(columns={'subject':'Subject','reduced_id':'Reduced ID','source_experiment':'Output ID','condition':'Condition','frames_used':'Frames','pa_mpjpe_mean_mm':'PA-MPJPE','gt_events':'GT events','wham_events':'WHAM events','foot_clearance_mean_abs_error_mm':'Foot clearance error','step_length_mean_abs_error_mm':'Step length error','stride_length_mean_abs_error_mm':'Stride length error'})
    (tables_dir/'table_6_reduced_metrics_all_sequences.tex').write_text(table_tex(red_table,'Reduced-matrix WHAM recovery metrics across all selected sequences.','tab:reduced_metrics_all_sequences'), encoding='utf-8')
    pair_key=pairwise[['subject','comparison','comparison_name','delta_pa_mpjpe_mean_mm','delta_foot_clearance_mean_abs_error_mm','delta_step_length_mean_abs_error_mm','delta_stride_length_mean_abs_error_mm','test_events','baseline_events']].rename(columns={'subject':'Subject','comparison':'Comparison','comparison_name':'Comparison name','delta_pa_mpjpe_mean_mm':'Delta PA-MPJPE','delta_foot_clearance_mean_abs_error_mm':'Delta foot clearance','delta_step_length_mean_abs_error_mm':'Delta step length','delta_stride_length_mean_abs_error_mm':'Delta stride length','test_events':'Test events','baseline_events':'Baseline events'})
    (tables_dir/'table_7_pairwise_comparisons.tex').write_text(table_tex(pair_key,'Pairwise reduced-matrix comparisons. Positive deltas indicate higher error in the test condition.','tab:pairwise_reduced_comparisons'), encoding='utf-8')
    # Reliability classification manual based on actual findings
    rel=pd.DataFrame([
        {'Finding':'Pre-render pipeline','Evidence':'SMPL-to-SMPL-X PA-MPJPE was about 14.8--14.9 mm and gait-feature errors remained small across all four sequences.','Interpretation':'The render-input motion was sufficiently preserved before WHAM.'},
        {'Finding':'Room complexity','Evidence':'R1--R0 did not consistently increase PA-MPJPE, but gait-event counts and stride-related errors varied by sequence.','Interpretation':'Room complexity alone was not the dominant failure source under bright frontal conditions.'},
        {'Finding':'Camera viewpoint','Evidence':'Upper-corner conditions often had low PA-MPJPE but larger foot-clearance or stride errors, especially in the primary sequence.','Interpretation':'Upper-corner views are risky for clinical gait features, even when aligned pose error is low.'},
        {'Finding':'Strong occlusion','Evidence':'R4--R1 increased PA-MPJPE in all selected sequences.','Interpretation':'Strong lower-body occlusion reduces local pose reliability.'},
        {'Finding':'Partial occlusion','Evidence':'R5 often reduced PA-MPJPE compared with R4, but clinical gait metrics were mixed.','Interpretation':'Partial frontal occlusion can be more recoverable than strong occlusion, but it is not uniformly reliable.'},
        {'Finding':'Metric mismatch','Evidence':'Several low-PA-MPJPE conditions still showed poor gait-event or foot/stride metrics.','Interpretation':'PA-MPJPE alone is insufficient for setup recommendations.'},
    ])
    (tables_dir/'table_8_reliability_summary.tex').write_text(table_tex(rel,'Reliability interpretation across the full and reduced analyses.','tab:reliability_summary_updated'), encoding='utf-8')
    # CSV copies
    reduced.to_csv(out/'reduced_matrix_metrics_all_sequences.csv',index=False)
    pairwise.to_csv(out/'reduced_matrix_pairwise_all_sequences.csv',index=False)
    validation.to_csv(out/'pipeline_validation_all_sequences.csv',index=False)
    sub01.to_csv(out/'sub01_full_matrix_metrics.csv',index=False)

    # Figures
    fig1=validation.copy();
    # validation PA-MPJPE by stage
    fig,ax=plt.subplots(figsize=(8,4.5))
    stages=list(fig1['stage'].unique()); subs=list(seqs['subject'])
    pos=np.arange(len(subs)); bw=0.35
    for i,stage in enumerate(stages):
        vals=[]
        for sub in subs:
            vals.append(float(fig1[(fig1.subject==sub)&(fig1.stage==stage)]['PA-MPJPE'].iloc[0]))
        ax.bar(pos+(i-0.5)*bw, vals, bw, label=stage)
    ax.set_xticks(pos); ax.set_xticklabels(subs); ax.set_ylabel('PA-MPJPE (mm)'); ax.set_title('Pre-WHAM validation PA-MPJPE')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(figs_dir/'fig_1_pipeline_validation_pa_mpjpe.png',dpi=220); plt.close(fig)
    # SUB01 PA bar
    plot_bar(sub01, 'experiment', 'pa_mpjpe_mean_mm', 'SUB01 full matrix: PA-MPJPE by experiment', 'PA-MPJPE (mm)', figs_dir/'fig_2_sub01_pa_mpjpe_by_experiment.png', order=list(FULL_CONDITION_MAP.keys()), width=10)
    # SUB01 spatial errors grouped
    fig,ax=plt.subplots(figsize=(11,5))
    exps=list(sub01['experiment']); pos=np.arange(len(exps)); bw=0.25
    for i,m in enumerate(['foot_clearance_mean_abs_error_mm','step_length_mean_abs_error_mm','stride_length_mean_abs_error_mm']):
        ax.bar(pos+(i-1)*bw, sub01[m], bw, label=PRETTY[m])
    ax.set_xticks(pos); ax.set_xticklabels(exps, rotation=30); ax.set_ylabel('Error (mm)'); ax.set_title('SUB01 full matrix: spatial gait-feature errors')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(figs_dir/'fig_3_sub01_spatial_gait_errors.png',dpi=220); plt.close(fig)
    # Scatter PA vs stride SUB01
    fig,ax=plt.subplots(figsize=(6.5,5))
    ax.scatter(sub01['pa_mpjpe_mean_mm'], sub01['stride_length_mean_abs_error_mm'])
    for _,r in sub01.iterrows(): ax.annotate(r['experiment'], (r['pa_mpjpe_mean_mm'],r['stride_length_mean_abs_error_mm']), xytext=(4,4), textcoords='offset points', fontsize=8)
    ax.set_xlabel('PA-MPJPE (mm)'); ax.set_ylabel('Stride length error (mm)'); ax.set_title('SUB01: PA-MPJPE versus stride-length error')
    fig.tight_layout(); fig.savefig(figs_dir/'fig_4_sub01_pa_vs_stride_scatter.png',dpi=220); plt.close(fig)
    # Frame-level SUB01 selected
    metrics_base=metrics_root/'clinically_relevant_metrics_v1'/'batch_wham_experiments'
    plot_line_frame(metrics_base, 'SUB01', 'SUB01_off_walk_1', {'E1':'E1','E3':'E3','E8':'E8','E10':'E10','E12':'E12'}, figs_dir/'fig_5_sub01_frame_level_pa_mpjpe_selected.png')
    # Reduced PA by subject condition
    plot_bar(reduced, 'reduced_id', 'pa_mpjpe_mean_mm','Reduced matrix: PA-MPJPE by condition and sequence','PA-MPJPE (mm)', figs_dir/'fig_6_reduced_pa_mpjpe_by_condition_subject.png', order=['R0','R1','R2','R3','R4','R5'], hue='subject', width=10)
    # Reduced foot clearance
    plot_bar(reduced, 'reduced_id', 'foot_clearance_mean_abs_error_mm','Reduced matrix: foot-clearance error by condition and sequence','Foot-clearance error (mm)', figs_dir/'fig_7_reduced_foot_clearance_by_condition_subject.png', order=['R0','R1','R2','R3','R4','R5'], hue='subject', width=10)
    # Reduced stride error
    plot_bar(reduced, 'reduced_id', 'stride_length_mean_abs_error_mm','Reduced matrix: stride-length error by condition and sequence','Stride-length error (mm)', figs_dir/'fig_8_reduced_stride_length_by_condition_subject.png', order=['R0','R1','R2','R3','R4','R5'], hue='subject', width=10)
    # Pairwise deltas
    plot_bar(pairwise, 'comparison', 'delta_pa_mpjpe_mean_mm','Pairwise setup effects: delta PA-MPJPE','Delta PA-MPJPE (mm)', figs_dir/'fig_9_pairwise_delta_pa_mpjpe.png', order=[p[0] for p in PAIRS], hue='subject', width=10)
    plot_bar(pairwise, 'comparison', 'delta_foot_clearance_mean_abs_error_mm','Pairwise setup effects: delta foot-clearance error','Delta foot-clearance error (mm)', figs_dir/'fig_10_pairwise_delta_foot_clearance.png', order=[p[0] for p in PAIRS], hue='subject', width=10)
    plot_bar(pairwise, 'comparison', 'delta_stride_length_mean_abs_error_mm','Pairwise setup effects: delta stride-length error','Delta stride-length error (mm)', figs_dir/'fig_11_pairwise_delta_stride_length.png', order=[p[0] for p in PAIRS], hue='subject', width=10)
    # Frame-level for replication subjects R0-R5
    for _,s in seqs.iterrows():
        sub=s['subject']; seq=s['sequence']
        if sub=='SUB01': continue
        plot_line_frame(metrics_base, sub, seq, {'R0':'E0','R1':'E1','R2':'E2','R3':'E3','R4':'E4','R5':'E5'}, figs_dir/f'fig_frame_pa_mpjpe_{sub}.png')

    # Build LaTeX result section
    # Include table contents inline
    table_texts={p.stem:p.read_text(encoding='utf-8') for p in sorted(tables_dir.glob('*.tex'))}
    result = rf'''
\chapter{{Results}}
\label{{chap:results}}

This chapter presents the results of the synthetic rendering and WHAM pose-recovery experiments. The analysis consists of two parts. First, the full experiment matrix is evaluated for the primary sequence \texttt{{SUB01\_off\_walk\_1}}. This full matrix is used to identify how room complexity, camera viewpoint, lighting, and lower-body occlusion affect recovered motion. Second, a reduced experiment matrix is evaluated for three additional walking sequences to check whether the main setup-related patterns also appear across different gait patterns and clinical profiles.

The results are interpreted using both technical pose-preservation metrics and clinically motivated gait-preservation metrics. PA-MPJPE is used as the main aligned 3D pose metric. However, the clinical interpretation also considers gait-event count, step/contact timing error, cadence error, foot-clearance error, step-length error, stride-length error, and knee range-of-motion error. This distinction is important because several conditions produced a plausible aligned body pose while failing to preserve lower-limb gait features.

\section{{Selected walking sequences}}
\label{{sec:results_selected_sequences}}

Table~\ref{{tab:result_sequences}} shows the four walking sequences used in the final analysis. The primary sequence, \texttt{{SUB01\_off\_walk\_1}}, was used for the full experiment matrix. The three additional sequences were used for the reduced replication matrix.

{table_texts['table_1_sequences']}

The additional sequences differ in gait score and freezer group. This makes the reduced analysis useful as a robustness check: the aim is not to repeat the full sensitivity analysis for every subject, but to test whether the main visual setup effects are specific to one walking sequence or also appear in other gait patterns.

\section{{Pre-render pipeline validation}}
\label{{sec:results_pipeline_validation_updated}}

Before interpreting WHAM recovery, the input motion was validated in two steps. First, the original BMCLab SMPL motion was compared with the final SMPL-X render-input motion. Second, the raw fitted SMPL-X motion was compared with the corrected render-ready SMPL-X motion. Table~\ref{{tab:pipeline_validation_updated}} summarizes these validation results across the four selected sequences.

{table_texts['table_2_validation']}

The validation results show that the pre-render pipeline preserved the underlying walking motion. The SMPL-to-SMPL-X render-input validation produced PA-MPJPE values of approximately 14.8--14.9 mm across all four sequences. The coordinate-correction validation produced even smaller PA-MPJPE values of approximately 3.4--3.7 mm. Cadence errors were zero in both validation stages, and the local gait-feature errors remained small. Therefore, the larger errors found after WHAM recovery are interpreted as degradation introduced mainly by RGB rendering, visual recording conditions, tracking, and monocular pose recovery rather than by the SMPL-to-SMPL-X conversion itself.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{{fig_1_pipeline_validation_pa_mpjpe.png}}
\caption{{Pre-WHAM validation PA-MPJPE for the selected walking sequences.}}
\label{{fig:pipeline_validation_pa}}
\end{{figure}}

Figure~\ref{{fig:pipeline_validation_pa}} supports the conclusion that the render-input and coordinate-correction stages introduced only limited pose error compared with the later WHAM recovery stage.

\section{{Full experiment matrix for the primary sequence}}
\label{{sec:results_full_matrix_primary}}

Table~\ref{{tab:full_matrix_results}} shows the full experiment matrix used for the primary sequence. The experiments are grouped by the recording factor they test: room complexity, camera viewpoint, lighting, and lower-body occlusion.

{table_texts['table_3_full_matrix']}

The main WHAM recovery results for the primary sequence are shown in Table~\ref{{tab:sub01_full_metrics}}. This table is the central result table for the full experiment matrix.

{table_texts['table_4_sub01_full_metrics']}

Across the full-length primary experiments, PA-MPJPE ranged from approximately 23 mm to 42 mm. The lowest PA-MPJPE values were found in E10, E4, and E13. However, these conditions did not preserve clinically relevant gait features well. E10 had a PA-MPJPE of 23.22 mm, but detected only six gait events compared with eight in the ground truth, and showed a foot-clearance error of 84.99 mm, a step-length error of 142.63 mm, and a stride-length error of 272.55 mm. E4 and E13 showed the same type of mismatch: low PA-MPJPE but high foot-clearance and stride-length errors.

By contrast, E3 and E12 were more balanced. E3 preserved the correct gait-event count, had no cadence error, and had the lowest stride-length error among the full-length primary conditions. E12 also preserved the correct gait-event count and had the lowest foot-clearance error. These results show that reliability should be interpreted as a combination of pose accuracy, gait-event recovery, and lower-limb feature preservation rather than by PA-MPJPE alone.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.90\textwidth]{{fig_2_sub01_pa_mpjpe_by_experiment.png}}
\caption{{PA-MPJPE for the full primary experiment matrix.}}
\label{{fig:sub01_pa_by_experiment}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{fig_3_sub01_spatial_gait_errors.png}}
\caption{{Spatial gait-feature errors for the full primary experiment matrix.}}
\label{{fig:sub01_spatial_errors}}
\end{{figure}}

Figures~\ref{{fig:sub01_pa_by_experiment}} and~\ref{{fig:sub01_spatial_errors}} should be interpreted together. The comparison shows that the lowest technical pose error does not necessarily correspond to the best gait-feature preservation. This is especially visible for the upper-corner and upper-corner occlusion conditions.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.75\textwidth]{{fig_4_sub01_pa_vs_stride_scatter.png}}
\caption{{Relationship between PA-MPJPE and stride-length error for the primary sequence.}}
\label{{fig:sub01_pa_vs_stride}}
\end{{figure}}

Figure~\ref{{fig:sub01_pa_vs_stride}} directly illustrates the mismatch between technical pose accuracy and clinical gait-feature preservation. Several low-PA-MPJPE conditions still have large stride-length errors. This supports the use of clinically motivated lower-limb metrics in addition to PA-MPJPE.

\section{{Effect of recording factors in the primary sequence}}
\label{{sec:results_factor_effects_primary}}

Room complexity was evaluated by comparing E0 and E1. E1 did not increase PA-MPJPE relative to E0 and showed lower foot-clearance, step-length, and stride-length errors. This suggests that room complexity alone was not the dominant source of degradation when lighting was bright, the camera view was frontal, and no occlusion was present. However, E1 detected one additional gait event, showing that temporal gait-event extraction can still vary under visually more realistic conditions.

Camera viewpoint was evaluated using E2--E5. The best camera view was E3, the frontal/table view at 0.95 m. E3 did not have the lowest PA-MPJPE, but it preserved the correct gait-event count, had no cadence error, and showed the lowest stride-length error. The upper-corner view E4 produced a much lower PA-MPJPE, but performed poorly on gait-event count, foot clearance, step length, and stride length. Therefore, E3 was the most reliable camera setup for gait assessment, while E4 was a clear example of a technically plausible but clinically unreliable pose recovery.

Lighting was evaluated using E6 and E7. Dim lighting increased PA-MPJPE in both the neutral and living-room settings. The effect was visible but less severe than the upper-corner and occlusion-related failure modes. Therefore, dim lighting is interpreted as a moderate degradation factor rather than the dominant source of failure in this experiment.

Lower-body occlusion was evaluated using E8--E13. E8 produced the highest PA-MPJPE among the full-length primary experiments, indicating poorer local pose recovery under strong lower-body occlusion. E10 and E13 again showed the mismatch between low PA-MPJPE and poor clinical gait features. E12 was the most recoverable occlusion condition: despite partial one-leg occlusion, it preserved the correct gait-event count and had the lowest foot-clearance error. E9 was treated separately as a tracking-instability case because it did not produce a directly comparable full 671-frame output.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{fig_5_sub01_frame_level_pa_mpjpe_selected.png}}
\caption{{Frame-level PA-MPJPE for selected primary-sequence conditions.}}
\label{{fig:sub01_frame_pa}}
\end{{figure}}

Figure~\ref{{fig:sub01_frame_pa}} shows that frame-level PA-MPJPE follows an oscillating pattern across the walking sequence. This is likely related to the periodic gait cycle: frames in which the legs are separated, crossing, partially hidden, or moving rapidly are more difficult to reconstruct than frames in which the body is more upright and the lower limbs are clearly visible. E8 shows higher peaks than the frontal baseline conditions, indicating that strong lower-body occlusion amplifies difficult gait phases.

\section{{Reduced matrix across additional walking sequences}}
\label{{sec:results_reduced_matrix}}

After the full matrix was evaluated on the primary sequence, a reduced experiment matrix was applied to three additional walking sequences. Table~\ref{{tab:reduced_matrix_results}} shows the reduced matrix. In the output folders for the additional sequences, these reduced conditions are stored as E0--E5, but they correspond to R0--R5 in the thesis interpretation.

{table_texts['table_5_reduced_matrix']}

Table~\ref{{tab:reduced_metrics_all_sequences}} gives the reduced-matrix results across all four selected sequences. For SUB01, the reduced IDs refer to selected conditions from the full experiment matrix. For SUB02, SUB08, and SUB17, the output IDs E0--E5 correspond directly to R0--R5.

{table_texts['table_6_reduced_metrics_all_sequences']}

The reduced-matrix results show that the exact best condition was not identical for every walking sequence. This is expected because the selected sequences differ in gait pattern, number of gait events, and clinical profile. However, the reduced analysis supports three broader findings. First, room complexity alone was not consistently the strongest source of degradation. Second, upper-corner views again showed that PA-MPJPE can be misleading: in several sequences, the upper-corner condition had a low PA-MPJPE but high foot-clearance or stride-related errors. Third, strong occlusion increased PA-MPJPE relative to the home baseline in all selected sequences, indicating that lower-body visibility loss reduces local pose reliability.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{fig_6_reduced_pa_mpjpe_by_condition_subject.png}}
\caption{{PA-MPJPE for the reduced matrix across all selected sequences.}}
\label{{fig:reduced_pa_by_condition}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{fig_7_reduced_foot_clearance_by_condition_subject.png}}
\caption{{Foot-clearance error for the reduced matrix across all selected sequences.}}
\label{{fig:reduced_foot_clearance}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{fig_8_reduced_stride_length_by_condition_subject.png}}
\caption{{Stride-length error for the reduced matrix across all selected sequences.}}
\label{{fig:reduced_stride}}
\end{{figure}}

Figures~\ref{{fig:reduced_pa_by_condition}}--\ref{{fig:reduced_stride}} show that the reduced-matrix results vary between sequences. This means that the experiment should not be interpreted as producing a universal ranking of recording conditions. Instead, the consistent conclusion is that technical pose metrics and gait-feature metrics must be interpreted together.

\section{{Pairwise reduced-matrix comparisons}}
\label{{sec:results_pairwise_reduced}}

Table~\ref{{tab:pairwise_reduced_comparisons}} presents the pairwise comparisons used to interpret the reduced matrix. Positive values indicate that the test condition produced higher error than the baseline condition.

{table_texts['table_7_pairwise_comparisons']}

The pairwise comparisons make the reduced analysis easier to interpret. The room-complexity comparison, R1--R0, did not show a consistent increase in PA-MPJPE across sequences. The upper-corner comparison, R3--R2, reduced PA-MPJPE in all selected sequences, but this was not a sign of better clinical recovery: the same comparison often increased foot-clearance error and sometimes increased stride-related error. The strong-occlusion comparison, R4--R1, increased PA-MPJPE in all selected sequences, showing that strong lower-body occlusion reliably worsened local pose recovery. The partial-occlusion comparison, R5--R4, usually reduced PA-MPJPE relative to strong occlusion, but the clinical gait-feature metrics remained mixed.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{fig_9_pairwise_delta_pa_mpjpe.png}}
\caption{{Pairwise reduced-matrix effects on PA-MPJPE. Positive values indicate higher error in the test condition.}}
\label{{fig:pairwise_delta_pa}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{fig_10_pairwise_delta_foot_clearance.png}}
\caption{{Pairwise reduced-matrix effects on foot-clearance error. Positive values indicate higher error in the test condition.}}
\label{{fig:pairwise_delta_foot}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{fig_11_pairwise_delta_stride_length.png}}
\caption{{Pairwise reduced-matrix effects on stride-length error. Positive values indicate higher error in the test condition.}}
\label{{fig:pairwise_delta_stride}}
\end{{figure}}

Figures~\ref{{fig:pairwise_delta_pa}}--\ref{{fig:pairwise_delta_stride}} show why a single metric cannot support setup recommendations. A condition can improve PA-MPJPE while worsening foot-clearance or stride-length error. This was especially visible for the upper-corner comparison.

\section{{Reliability interpretation}}
\label{{sec:results_reliability_updated}}

Table~\ref{{tab:reliability_summary_updated}} summarizes the reliability interpretation across the full and reduced analyses.

{table_texts['table_8_reliability_summary']}

The overall result is that the synthetic framework is useful for identifying setup-sensitive failure modes, but the reliability of a recording condition depends on the metric being considered. The most important finding is the repeated mismatch between PA-MPJPE and clinically motivated gait features. PA-MPJPE is useful for evaluating aligned 3D body-pose recovery, but it does not guarantee that gait-event timing, foot clearance, step length, or stride length are preserved.

For practical home-recording recommendations, the primary sequence suggests that a frontal/table camera view around 0.95 m is preferable to an upper-corner view. Across the additional sequences, this exact ranking was not always repeated, but the broader warning remained: upper-corner views can look good according to PA-MPJPE while producing unreliable clinical gait features. Strong lower-body occlusion consistently increased PA-MPJPE compared with the home baseline, confirming that lower-body visibility should be prioritized. Partial frontal occlusion was more recoverable than strong occlusion in terms of PA-MPJPE, but it did not always preserve the clinical gait features. Therefore, reliable home recording should prioritize frontal camera placement, adequate lighting, full or near-full lower-body visibility, and continuous tracking throughout the walking sequence.

\section{{Summary of results}}
\label{{sec:results_summary_updated}}

The results answer the experimental questions as follows. First, the pre-render validation showed that the original BMCLab motion was preserved sufficiently before WHAM recovery. Second, the full primary experiment showed that room complexity alone was not the main source of degradation under good lighting and frontal visibility. Third, camera viewpoint had a strong effect on clinical gait-feature preservation: the primary frontal/table view was the most reliable, while the upper-corner view was misleadingly good according to PA-MPJPE but poor according to gait features. Fourth, dim lighting produced moderate degradation. Fifth, lower-body occlusion reduced reliability, especially when combined with upper-corner viewpoint or tracking instability.

The additional sequences support a more cautious conclusion. They do not produce one universal ranking of all recording conditions, but they confirm that setup effects are metric-dependent and sequence-dependent. The most stable conclusion is therefore methodological as much as practical: robustness evaluation for home-based Parkinsonian gait assessment should not rely on PA-MPJPE alone. It should combine pose-aligned errors, gait-event recovery, lower-limb gait features, frame-level reliability, and tracking continuity.
'''
    (out/'updated_results_section.tex').write_text(result.strip()+"\n", encoding='utf-8')
    # README
    (out/'README.txt').write_text(textwrap.dedent('''
    Updated BEP results package
    ===========================
    Contents:
    - updated_results_section.tex: copy-paste LaTeX results chapter section with embedded tables and figure references.
    - tables/: individual LaTeX table files.
    - figures/: generated PNG figures.
    - reduced_matrix_metrics_all_sequences.csv: reduced-matrix metrics used in tables/figures.
    - reduced_matrix_pairwise_all_sequences.csv: pairwise comparisons used in tables/figures.
    - pipeline_validation_all_sequences.csv: validation metrics used in tables/figures.
    - make_updated_results_tables_figures.py: the code used to generate this pack.

    LaTeX preamble requirements:
    \\usepackage{graphicx}
    \\usepackage{booktabs}
    \\usepackage{array}

    Put the PNG files in the same folder as the .tex file, or update the figure paths.
    ''').strip()+"\n", encoding='utf-8')
    # copy script into output
    shutil.copy2(Path(__file__), out/'make_updated_results_tables_figures.py')
    # zip
    zip_path=out.with_suffix('.zip')
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if '_work' in p.parts: continue
            z.write(p,p.relative_to(out))
    print(zip_path)

if __name__=='__main__':
    main()
