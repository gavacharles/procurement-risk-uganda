"""Follow-on step 4: what is associated with slow delivery? Survival and predictive models (exploratory).

A. Contract level, satellite sample (upgrade roads with a verdict). Duration = tender invitation -> satellite
   completion (>= 80% sealed), right-censored at the last image. Kaplan-Meier, and Cox proportional hazards with
   one covariate at a time (n ~ 17, ~8 events: anything more would overfit): contract length, procurement
   red-flag score, award above the engineer's estimate (> 1.1), local-government buyer, formal dispute during
   construction (data/manual/road_disputes_checked.csv). Hazard ratio < 1 = slower completion.

B. Project-visit level, government monitoring reports (progress_reports.py), all road and bridge projects with
   both reported and planned progress. Outcome: "behind schedule" = planned - actual >= 10 points. Features: the
   reasons monitors give (right of way, contractor capacity, payment/funding, weather, design/variation,
   materials, COVID), time progress, bridge vs road works, report year. Logistic regression (odds ratios) and gradient boosting (XGBoost) with cross-validation grouped
   by project, and SHAP values for the boosted model.
   Caution: monitors explain delays, so a reason is more likely to be written down when a project is behind;
   the coefficients describe which explanations accompany slippage, not what causes it.

Out: paper/delay_models.json, chart p3_f19_models
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))
CAUSES = ["cause_row", "cause_contractor", "cause_payment", "cause_weather", "cause_design", "cause_materials", "cause_covid"]
LABEL = {"cause_row": "right of way / compensation", "cause_contractor": "contractor capacity / cash flow",
         "cause_payment": "late payment / funding", "cause_weather": "weather", "cause_design": "design / variation",
         "cause_materials": "materials / quarries", "cause_covid": "COVID-19", "time_progress": "time progress",
         "type_rehab": "rehabilitation", "type_bridge": "bridge", "type_maint": "maintenance", "type_other": "other works",
         "year": "report year", "time_missing": "time progress not stated"}


# ------------------------------------------------------------------ A. survival, satellite sample
def part_a():
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
    R = pd.read_parquet(P / "road_progress_roads.parquet")
    R = R[(R.kind == "upgrade") & R.t_tender.notna()
          & ~R.status.str.startswith(("excluded", "insufficient", "sealed by the first"))].copy()
    R["event"] = R.t_done.notna().astype(int)
    R["T"] = np.where(R.event == 1, R.t_done, R.t_last) - R.t_tender
    disp = pd.read_csv(ROOT / "data" / "manual" / "road_disputes_checked.csv")
    dc = disp[(disp.relevant == "yes") & (disp.stage == "construction")].road_id.unique()
    dp = disp[(disp.relevant == "yes") & (disp.stage == "procurement")].road_id.unique()
    R["dispute_construction"] = R.id.isin(dc).astype(int)
    R["dispute_procurement"] = R.id.isin(dp).astype(int)
    R["log_km"] = np.log(R.contract_km)
    R["risk_z"] = (R.risk_score - R.risk_score.mean()) / R.risk_score.std()
    R["award_above_est"] = (R.award_to_est > 1.1).astype(float).where(R.award_to_est.notna())
    R["local_buyer"] = (~R.buyer.str.contains("National Roads", na=False)).astype(int)
    covs = ["log_km", "risk_z", "award_above_est", "local_buyer", "dispute_construction", "dispute_procurement"]
    res = {}
    for c in covs:
        d = R[["T", "event", c]].dropna()
        if d[c].nunique() < 2 or d.event.sum() < 3:
            res[c] = dict(n=int(len(d)), note="no variation or too few events")
            continue
        try:
            m = CoxPHFitter(penalizer=0.1).fit(d, "T", "event")
            s = m.summary.loc[c]
            res[c] = dict(n=int(len(d)), events=int(d.event.sum()), hr=round(float(s["exp(coef)"]), 2),
                          ci=[round(float(s["exp(coef) lower 95%"]), 2), round(float(s["exp(coef) upper 95%"]), 2)],
                          p=round(float(s["p"]), 3))
        except Exception as e:  # noqa: BLE001
            res[c] = dict(n=int(len(d)), note=str(e)[:80])
    km = KaplanMeierFitter().fit(R["T"], R.event)
    med = km.median_survival_time_
    lr = logrank_test(R[R.dispute_construction == 1]["T"], R[R.dispute_construction == 0]["T"],
                      R[R.dispute_construction == 1].event, R[R.dispute_construction == 0].event)
    return R, dict(n=int(len(R)), events=int(R.event.sum()), km_median_years=None if not np.isfinite(med) else round(float(med), 2),
                   cox_univariate=res, logrank_dispute=dict(p=round(float(lr.p_value), 3),
                                                            n_dispute=int(R.dispute_construction.sum())),
                   note="univariate, penalised (0.1); exploratory: ~17 roads")


# ------------------------------------------------------------------ B. monitoring-report panel
def ptype(name):
    n = name.lower()
    if re.search(r"bridge|crossing|culvert", n):
        return "bridge"
    if re.search(r"maint|framework|term\b|routine|periodic|call.?off|reseal", n):
        return "maint"
    if re.search(r"rehab|reconstruct|widen|overlay", n):
        return "rehab"
    if re.search(r"upgrad|construction of|[–\-—]|bypass|expressway|road", n):
        return "upgrade"
    return "other"


def part_b():
    D = pd.read_parquet(P / "reported_progress.parquet")
    X = D[D.physical.notna() & D.planned.notna() & (D.planned >= D.physical - 50)].copy()
    X["pkey"] = X.project.str.lower().str.replace(r"\(.*?\)|[^a-z]", "", regex=True).str[:22]
    X = X.sort_values("how").drop_duplicates(["file", "pkey"])  # one row per project per report
    X["slip"] = X.planned - X.physical
    X["behind"] = (X.slip >= 10).astype(int)
    X["type"] = X.project.map(ptype)
    X["type_bridge"] = (X.type == "bridge").astype(int)  # other types too rare (< 10 rows): pooled with roads
    X["time_missing"] = X.time.isna().astype(int)
    X["time_progress"] = X.time.fillna(X.time.median())
    X["year"] = X["asof"].dt.year + (X["asof"].dt.month == 12) * 0.5
    feats = CAUSES + ["time_progress", "time_missing", "type_bridge", "year"]
    for c in CAUSES:
        X[c] = X[c].astype(int)
    y = X.behind.to_numpy()
    groups = X.pkey.to_numpy()
    out = dict(n_rows=int(len(X)), n_projects=int(X.pkey.nunique()), n_reports=int(X.file.nunique()),
               share_behind=round(float(y.mean()), 2), median_slip=round(float(X.slip.median()), 1),
               cause_share=X[CAUSES].mean().round(3).to_dict(),
               cause_share_behind=X[X.behind == 1][CAUSES].mean().round(3).to_dict(),
               cause_share_on_track=X[X.behind == 0][CAUSES].mean().round(3).to_dict())
    # logistic regression with cluster-robust SE by project
    import statsmodels.api as sm
    Xm = sm.add_constant(X[feats].assign(year=X.year - X.year.mean(), time_progress=X.time_progress / 10).astype(float))
    keep = Xm.columns[Xm.std() > 0].tolist()
    m = sm.Logit(y, Xm[keep]).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})
    ors = {k: dict(or_=round(float(np.exp(m.params[k])), 2), ci=[round(float(np.exp(m.conf_int().loc[k, 0])), 2),
                                                                 round(float(np.exp(m.conf_int().loc[k, 1])), 2)],
                   p=round(float(m.pvalues[k]), 3)) for k in keep if k != "const"}
    out["logit_odds_ratios"] = ors
    out["logit_note"] = "time_progress per 10 points; year centred; SE clustered by project"
    # boosted trees, grouped CV, SHAP
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBClassifier
    Xf = X[feats].astype(float).to_numpy()
    p_x, p_l = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(Xf, y, groups):
        xg = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                           eval_metric="logloss", verbosity=0).fit(Xf[tr], y[tr])
        p_x[te] = xg.predict_proba(Xf[te])[:, 1]
        lr = LogisticRegression(max_iter=2000).fit((Xf[tr] - Xf[tr].mean(0)) / (Xf[tr].std(0) + 1e-9), y[tr])
        p_l[te] = lr.predict_proba((Xf[te] - Xf[tr].mean(0)) / (Xf[tr].std(0) + 1e-9))[:, 1]
    out["cv_auc"] = dict(xgboost=round(float(roc_auc_score(y, p_x)), 3), logistic=round(float(roc_auc_score(y, p_l)), 3),
                         folds="5, grouped by project")
    import shap
    xg = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                       eval_metric="logloss", verbosity=0).fit(Xf, y)
    sv = shap.TreeExplainer(xg).shap_values(Xf)
    imp = pd.Series(np.abs(sv).mean(0), index=feats).sort_values(ascending=False)
    out["shap_mean_abs"] = imp.round(4).to_dict()
    return X, out, sv, feats


def chart(R, A, X, B, sv, feats):
    from lifelines import KaplanMeierFitter
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw=dict(width_ratios=[1, 1.1, 1.1]))
    a = ax[0]
    for flag, col, lab in [(0, "#2a78d6", "no formal dispute"), (1, "#eb6834", "dispute during works")]:
        d = R[R.dispute_construction == flag]
        if len(d) >= 2:
            KaplanMeierFitter().fit(d["T"], d.event, label=f"{lab} (n={len(d)})").plot_survival_function(ax=a, ci_show=False, color=col)
    a.set_xlabel("years since tender invitation")
    a.set_ylabel("share of roads not yet complete")
    a.set_title("a  Time to completion (satellite)", loc="left")
    a.legend(loc="lower left", fontsize=7)
    b = ax[1]
    o = B["logit_odds_ratios"]
    ks = [k for k in CAUSES + ["type_bridge", "time_progress"] if k in o]
    yv = np.arange(len(ks))
    b.errorbar([o[k]["or_"] for k in ks], yv, xerr=[[o[k]["or_"] - o[k]["ci"][0] for k in ks],
                                                     [o[k]["ci"][1] - o[k]["or_"] for k in ks]],
               fmt="o", color="#2a78d6", ms=4, capsize=2)
    b.axvline(1, color="#8a8a85", lw=0.8)
    b.set_xscale("log")
    b.set_yticks(yv, [LABEL.get(k, k) for k in ks])
    b.invert_yaxis()
    b.set_xlabel("odds ratio of being ≥10 points behind plan")
    b.set_title("b  Reasons monitors give (association)", loc="left")
    c = ax[2]
    imp = pd.Series(B["shap_mean_abs"]).head(10)[::-1]
    c.barh([LABEL.get(k, k) for k in imp.index], imp.values, color="#1baf7a")
    c.set_xlabel(f"mean |SHAP| (boosted model; CV AUC {B['cv_auc']['xgboost']:.2f})")
    c.set_title("c  Predictive weight", loc="left")
    fig.tight_layout()
    f = ROOT / "posts" / "media" / "chart_delay_models.png"
    fig.savefig(f, dpi=200)
    shutil.copy(f, ROOT / "paper" / "figures" / "p3_f19_models.png")


def main():
    R, A = part_a()
    X, B, sv, feats = part_b()
    out = dict(A_survival_satellite=A, B_monitoring_panel=B)
    (ROOT / "paper" / "delay_models.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))
    chart(R, A, X, B, sv, feats)


if __name__ == "__main__":
    main()
