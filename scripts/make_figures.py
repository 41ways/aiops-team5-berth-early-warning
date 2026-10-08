"""
발표용 그림 4장 (PNG) → reports/figures_v2.0/
  fig1_월별오차.png        터미널 예정 vs 모델, 월별 접안 평균 오차(분)
  fig2_핵심지표.png        평균 오차·30분 안·1시간 초과: 터미널 / 같은 입력 단순 평균 / 모델
  fig3_일정표정확도.png    "X시간 안에 맞힌 비율" 곡선: 접안(터미널·모델), 출항(모델)
  fig4_AIOps흐름.png       드리프트 → 백그라운드 재학습 → 게이트 → 승격 / 되돌리기 / 폴백
데이터: reports/backtest_v1.0/ (scripts/backtest.py 결과)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "AppleGothic", "Malgun Gothic", "sans-serif"],
                     "axes.unicode_minus": False, "font.size": 12})

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
MODEL, TERMINAL, MEAN20 = "#2a78d6", "#eb6834", "#1baf7a"   # 검증된 범주색 1·2·3번 (dataviz 팔레트)
SRC = "reports/backtest_v1.0"
OUT = "reports/figures_v2.0"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def fig1(monthly):
    m = monthly.copy()
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    style(ax)
    for col, color, name in (("terminal_mae_h", TERMINAL, "터미널 예정"), ("model_v1_mae_h", MODEL, "모델")):
        y = m[col].values * 60
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(x[-1] + 0.12, y[-1], f"{name} {y[-1]:.0f}분", color=INK, va="center", fontsize=12)
    i8 = list(m["month"]).index("2026-08")
    ax.annotate(f"8월 혼잡기\n터미널 {m['terminal_mae_h'][i8]*60:.0f}분 → 모델 {m['model_v1_mae_h'][i8]*60:.0f}분",
                xy=(i8, m["terminal_mae_h"][i8] * 60), xytext=(i8 - 2.3, 20.5), color=INK, fontsize=11,
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=1))
    ax.set_xticks(x, [f"{int(s[-2:])}월" for s in m["month"]])
    ax.set_ylim(0, 22)
    ax.set_ylabel("접안 평균 오차 (분)", color=INK2)
    ax.set_xlim(-0.3, len(m) - 1 + 1.4)
    ax.set_title("접안 10시간 전 예측 오차: 매달 모델이 터미널 예정보다 작다 (2026년, 430척)", loc="left", color=INK, fontsize=13)
    ax.legend(handles=[plt.Line2D([], [], color=TERMINAL, lw=2, marker="o", label="터미널 예정"),
                       plt.Line2D([], [], color=MODEL, lw=2, marker="o", label="모델")],
              frameon=False, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_월별오차.png", dpi=200)
    plt.close(fig)


def fig2(s):
    a = s["arrival"]
    groups = [("터미널 예정", a["terminal_plan"], TERMINAL), ("같은 입력\n단순 평균", a["baseline_mean_of_same_20"], MEAN20),
              ("모델 (LSTM)", a["model_v1"], MODEL)]
    panels = [("평균 오차 (분)", lambda d: d["mae_h"] * 60, "{:.1f}분"),
              ("30분 안에 맞힌 비율 (%)", lambda d: d["within_30m"] * 100, "{:.1f}%"),
              ("1시간 넘게 틀린 배 (430척 중)", lambda d: d["over_1h"] * 430, "{:.0f}척")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), facecolor=SURFACE)
    for ax, (title, fn, fmt) in zip(axes, panels):
        style(ax)
        vals = [fn(d) for _, d, _ in groups]
        bars = ax.bar(range(3), vals, color=[c for _, _, c in groups], width=0.62, edgecolor=SURFACE, linewidth=2)
        top = max(vals) * 1.22
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.02, fmt.format(v), ha="center", va="bottom", color=INK, fontsize=12)
        ax.set_xticks(range(3), [g for g, _, _ in groups], fontsize=10.5, color=INK2)
        ax.set_ylim(0, top)
        ax.set_title(title, loc="left", color=INK, fontsize=12.5)
    fig.suptitle("모델이 터미널 예정을 이긴다. 다만 같은 입력의 단순 평균과는 차이가 작다 → 개선은 입력 설계에서 나온다",
                 x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{OUT}/fig2_핵심지표.png", dpi=200)
    plt.close(fig)


def fig3(v):
    hs = np.linspace(0, 6, 121)
    series = [("접안 — 모델", v["err_v1"].abs(), MODEL, "-"), ("접안 — 터미널 예정", v["err_terminal"].abs(), TERMINAL, "-"),
              ("출항 — 모델", v["err_dep_schedule"].dropna().abs(), MODEL, (0, (2, 2)))]
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    for name, e, color, ls in series:
        y = [(e <= h).mean() * 100 for h in hs]
        ax.plot(hs, y, color=color, linewidth=2, linestyle=ls)
    e_m, e_t, e_d = (series[i][1] for i in range(3))
    ax.text(1.05, 88, f"접안 30분 안: 모델 {(e_m <= .5).mean()*100:.0f}% · 터미널 {(e_t <= .5).mean()*100:.0f}%\n접안 1시간 안: 모델 {(e_m <= 1).mean()*100:.0f}% · 터미널 {(e_t <= 1).mean()*100:.0f}%",
            color=INK, fontsize=11, va="top")
    ax.text(3.05, (e_d <= 3).mean() * 100 - 9, f"출항 — 모델 (점선)\n1시간 안 {(e_d <= 1).mean()*100:.0f}% · 3시간 안 {(e_d <= 3).mean()*100:.0f}%",
            color=INK, fontsize=11)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 102)
    ax.set_xlabel("허용 오차 (시간)", color=INK2)
    ax.set_ylabel("이 안에 맞힌 비율 (%)", color=INK2)
    ax.set_title("일정표 정확도: 접안은 믿을 만하고, 출항(선석이 비는 시각)은 아직 넓다", loc="left", color=INK, fontsize=13)
    ax.legend(handles=[plt.Line2D([], [], color=MODEL, lw=2, label="접안 — 모델"),
                       plt.Line2D([], [], color=TERMINAL, lw=2, label="접안 — 터미널 예정"),
                       plt.Line2D([], [], color=MODEL, lw=2, ls=(0, (2, 2)), label="출항 — 모델")],
              frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_일정표정확도.png", dpi=200)
    plt.close(fig)


def fig4():
    fig, ax = plt.subplots(figsize=(12, 4.6), facecolor=SURFACE)
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, title, sub, edge=GRID):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="white", edgecolor=edge, linewidth=1.5))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", color=INK, fontsize=12.5, weight="bold")
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", color=INK2, fontsize=10)

    def arrow(x1, y1, x2, y2, label=""):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=INK2, linewidth=1.3))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.16, label, ha="center", color=INK2, fontsize=9.5)

    box(0.1, 1.75, 2.0, 1.1, "예측·기록", "배마다 보정값\n실제 접안이 나오면 채점")
    box(2.7, 1.75, 2.0, 1.1, "드리프트 감지", "최근 21척\n터미널보다 나쁨 / 오차>1h")
    box(5.3, 1.75, 2.1, 1.1, "백그라운드 재학습", "최근 21일 배로 fine-tune\n응답은 바로")
    box(7.9, 1.75, 1.6, 1.1, "게이트", "같은 홀드아웃에서\n누가 가장 나은가")
    box(10.15, 3.25, 1.85, 1.0, "새 모델 승격", "[OK] v2", edge=MODEL)
    box(10.15, 1.8, 1.85, 1.0, "되돌리기", "[OK] 예전 버전 v1", edge=MODEL)
    box(10.15, 0.35, 1.85, 1.0, "유지 + 폴백", "터미널 예정으로 응답", edge=TERMINAL)
    arrow(2.1, 2.3, 2.7, 2.3)
    arrow(4.7, 2.3, 5.3, 2.3)
    arrow(7.4, 2.3, 7.9, 2.3)
    arrow(9.5, 2.5, 10.15, 3.6)
    arrow(9.5, 2.3, 10.15, 2.3)
    arrow(9.5, 2.1, 10.15, 0.95)
    ax.text(0.1, 4.25, "AIOps 루프: 나빠지면 스스로 고치고, 못 고치면 기존 예정으로 물러난다", color=INK, fontsize=13, va="center")
    ax.text(0.1, 0.55, "시연: 정상 21척 → +2시간 주입 → v2 승격 → 다시 실제 → v1로 되돌림\n"
                       "실제 데이터(8월 혼잡 포함)에서는 드리프트가 거의 나지 않았다", color=INK2, fontsize=10, va="center")
    fig.savefig(f"{OUT}/fig4_AIOps흐름.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    monthly = pd.read_csv(f"{SRC}/monthly.csv")
    summary = json.load(open(f"{SRC}/summary.json", encoding="utf-8"))
    vessels = pd.read_csv(f"{SRC}/vessels.csv")
    fig1(monthly)
    fig2(summary)
    fig3(vessels)
    fig4()
    print("saved:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
