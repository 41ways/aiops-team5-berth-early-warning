"""
LLM 사이드카: 모델 변경 이력 일일 리포트 (교수님 Wrap-Up "LLM sidecar").

서빙 요청 경로 밖에서 따로 돈다 (cron 등으로 하루 1번). 예측·승격 결정에는 관여하지 않고 읽기만 한다.

  읽는 것  : MLflow run·모델 버전 이력, logs/aiops.log, reports/backtest_v1.0/summary.json
  만드는 것: reports/change_report_YYYYMMDD.md
             - ANTHROPIC_API_KEY 와 anthropic 패키지가 있으면 LLM이 요약한 글
             - 없으면 같은 사실로 규칙 기반 요약 + LLM에 보낼 프롬프트(…_prompt.txt)

실행: python scripts/llm_report.py [--since 2026-10-08]
"""
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_NAME = "BERTH_ARRIVAL_Corrector"
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")


def collect(since: str | None) -> dict:
    facts = {"generated_at": datetime.now().isoformat(timespec="seconds")}
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        c = MlflowClient()
        runs = mlflow.search_runs(search_all_experiments=True, order_by=["start_time ASC"])
        rows = []
        for _, r in runs.iterrows():
            t = r["start_time"].tz_convert("Asia/Seoul") if r["start_time"].tzinfo else r["start_time"]
            if since and str(t) < since:
                continue
            rows.append({"time": str(t)[:19], "run": r.get("tags.mlflow.runName"), "mode": r.get("params.mode"),
                         "mae": r.get("metrics.mae"), "prod_mae": r.get("metrics.prod_mae"),
                         "terminal_mae": r.get("metrics.baseline_mae"), "n": r.get("params.n_cases") or r.get("params.n_train")})
        facts["runs"] = rows
        facts["versions"] = [{"version": v.version, "stage": v.current_stage, "run_id": v.run_id[:8],
                              "created": datetime.fromtimestamp(v.creation_timestamp / 1000).isoformat(timespec="seconds")}
                             for v in sorted(c.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version))]
    except Exception as e:
        facts["mlflow_error"] = str(e)
    if os.path.exists("logs/aiops.log"):
        lines = open("logs/aiops.log", encoding="utf-8").read().splitlines()
        facts["aiops_log"] = [l for l in lines if not since or l[:10] >= since][-60:]
    p = "reports/backtest_v1.0/summary.json"
    if os.path.exists(p):
        facts["backtest"] = json.load(open(p, encoding="utf-8"))
    return facts


PROMPT = """너는 부산신항 HJNC 접안시각 보정 모델의 운영 리포트 작성자다. 아래 사실(JSON)만 근거로,
항만공사 담당자가 읽을 한국어 일일 변경 리포트를 써라. 사실에 없는 숫자나 원인을 만들지 마라.

구성:
1. 오늘 한 줄 요약 (Production 버전이 바뀌었는지, 터미널 예정보다 나은지)
2. 무엇이 바뀌었나: 드리프트 감지 시각과 이유(relative=터미널 예정보다 못함 / absolute=오차 1시간 초과), 재학습 결과, 승격·차단·폴백
3. 왜 그렇게 판단했나: 게이트 숫자 (새 모델 MAE, 현재 Production MAE, 터미널 예정 MAE)
4. 사람이 확인할 것: 시뮬레이션(simulation shift)으로 생긴 변경인지, 실제 데이터 변화인지 구분해서

사실:
{facts}
"""


def rule_based(f: dict) -> str:
    out = [f"# 모델 변경 리포트 ({f['generated_at'][:10]})", ""]
    vs = f.get("versions", [])
    prod = [v for v in vs if v["stage"] == "Production"]
    out.append(f"- Production: v{prod[-1]['version']}" if prod else "- Production 없음")
    out.append(f"- 등록된 버전 {len(vs)}개: " + ", ".join(f"v{v['version']}({v['stage']})" for v in vs))
    log = f.get("aiops_log", [])
    n_drift = sum("[WARN] drift detected" in l for l in log)
    n_ok = sum("[OK] new_rmse" in l for l in log)
    n_fail = sum("gate failed" in l for l in log)
    n_fb = sum("fallback to terminal plan" in l for l in log)
    sim = any("simulation:" in l for l in log)
    out.append(f"- 드리프트 감지 {n_drift}회, 승격 {n_ok}회, 게이트 차단 {n_fail}회, 폴백 {n_fb}회"
               + (" (시뮬레이션 지연 주입 포함 — 실제 변화와 구분할 것)" if sim else ""))
    out += ["", "## 재학습 기록", "| 시각 | 종류 | 새 모델 MAE | 현재 Prod MAE | 터미널 MAE |", "|---|---|---|---|---|"]
    fmt = lambda x: "-" if x is None or x != x else f"{x:.3f}"
    for r in f.get("runs", []):
        out.append(f"| {r['time']} | {r['mode']} | {fmt(r['mae'])} | {fmt(r['prod_mae'])} | {fmt(r['terminal_mae'])} |")
    bt = f.get("backtest")
    if bt:
        a = bt["arrival"]
        out += ["", "## 기준 성능 (백테스트)",
                f"- {bt['test_period']}, {bt['n_vessels']}척: 터미널 {a['terminal_plan']['mae_h']}h → 모델 {a['model_v1']['mae_h']}h"]
    out += ["", "## 로그 (최근)", "```", *log[-20:], "```"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="YYYY-MM-DD 이후만")
    a = ap.parse_args()
    facts = collect(a.since)
    os.makedirs("reports", exist_ok=True)
    stem = f"reports/change_report_{datetime.now():%Y%m%d}"
    prompt = PROMPT.format(facts=json.dumps(facts, ensure_ascii=False, indent=1, default=str))
    text = None
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            msg = anthropic.Anthropic().messages.create(model=LLM_MODEL, max_tokens=2000,
                                                        messages=[{"role": "user", "content": prompt}])
            text = msg.content[0].text
        except Exception as e:
            print(f"[llm] 호출 실패, 규칙 기반으로 대신 작성: {e}")
    if text is None:
        text = rule_based(facts)
        open(f"{stem}_prompt.txt", "w", encoding="utf-8").write(prompt)
    open(f"{stem}.md", "w", encoding="utf-8").write(text)
    print(f"{stem}.md 작성")


if __name__ == "__main__":
    main()
