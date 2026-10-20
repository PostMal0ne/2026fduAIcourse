#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成本—准确率前沿分析（补充分析，读取已有实验产物，不调用模型）

为什么需要它：
  本项目题库对强模型已经"封顶"（测试集 25/25），所以"提高准确率"没有空间；
  路由真正的价值在于"同样的准确率、更低的延迟"。本脚本从每次调用的经验记录
  （experience.csv）与测试矩阵（summary.json）出发，扫描 λ，画出这条前沿，
  并给出 λ=0（纯准确率，应当退化为最优单臂）作为对照。

用法：
    python analyze.py --dir . --w 3 --lat-ref 8 --out frontier.md
"""

import argparse
import csv
import glob
import json
import os
import statistics

KIND_CN = {"multiply": "三位数乘法", "mixed": "多步混合运算", "percent": "百分比",
           "logic": "逆运算推理", "extract": "字符串抽取"}


def load_runs(d):
    runs = []
    for sd in sorted(glob.glob(os.path.join(d, "runs", "run_*"))):
        sump = os.path.join(sd, "summary.json")
        expp = os.path.join(sd, "experience.csv")
        if not (os.path.exists(sump) and os.path.exists(expp)):
            continue
        summary = json.load(open(sump, encoding="utf-8"))
        stats = {}
        with open(expp, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                key = (row["kind"], row["arm"])
                s = stats.setdefault(key, {"n": 0, "ok": 0.0, "lat": []})
                s["n"] += 1
                s["ok"] += float(row["score"])
                s["lat"].append(float(row["latency_s"]))
        runs.append({"seed": summary["seed"], "matrix": summary["matrix"], "stats": stats,
                     "summary": summary})
    return runs


def shrunk(stats, kind, arm, kinds, w):
    """题型内准确率向该臂全局均值收缩（与 rsl.py 一致）。"""
    g_ok = sum(stats.get((k, arm), {}).get("ok", 0.0) for k in kinds)
    g_n = sum(stats.get((k, arm), {}).get("n", 0) for k in kinds)
    g = g_ok / g_n if g_n else 0.0
    s = stats.get((kind, arm))
    if not s:
        return g, None
    acc = (s["ok"] + w * g) / (s["n"] + w)
    return acc, statistics.mean(s["lat"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--w", type=float, default=3.0)
    ap.add_argument("--lat-ref", type=float, default=8.0)
    ap.add_argument("--out", default="frontier.md")
    args = ap.parse_args()

    runs = load_runs(args.dir)
    if not runs:
        raise SystemExit("没找到 runs/run_*/summary.json + experience.csv，请先跑 rsl.py")
    kinds = list(runs[0]["matrix"][list(runs[0]["matrix"])[0]].keys())
    arms = list(runs[0]["matrix"].keys())
    lams = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]

    def evaluate(choices):
        """choices: {kind: arm} → 跨 seed 汇总的测试准确率与延迟"""
        ok = n = 0
        lats = []
        for r in runs:
            for k, arm in choices.items():
                c = r["matrix"][arm][k]
                ok += c["ok"]
                n += c["n"]
                lats.extend(c["lat"])
        return ok, n, statistics.mean(lats)

    rows = []
    choices_by_lam = {}
    for lam in lams:
        per_seed = []
        ok = n = 0
        lats = []
        for r in runs:
            ch = {}
            for k in kinds:
                best, bv = None, -1e9
                for a in arms:
                    acc, lat = shrunk(r["stats"], k, a, kinds, args.w)
                    v = acc - lam * ((lat or args.lat_ref) / args.lat_ref)
                    if v > bv:
                        bv, best = v, a
                ch[k] = best
            per_seed.append(ch)
            for k, arm in ch.items():
                c = r["matrix"][arm][k]
                ok += c["ok"]
                n += c["n"]
                lats.extend(c["lat"])
        choices_by_lam[lam] = per_seed[0]
        rows.append({"lam": lam, "ok": ok, "n": n, "acc": ok / n,
                     "lat": statistics.mean(lats), "choices": per_seed})

    # 全用强模型（参照点）
    strong = "strong" if "strong" in arms else arms[0]
    ok = n = 0
    lats = []
    for r in runs:
        for k in kinds:
            c = r["matrix"][strong][k]
            ok += c["ok"]
            n += c["n"]
            lats.extend(c["lat"])
    base = {"acc": ok / n, "lat": statistics.mean(lats)}

    L = ["# 成本—准确率前沿（补充分析）\n",
         "> 由 `analyze.py` 从已有实验产物（`runs/*/experience.csv` + `summary.json`）离线计算，不调用模型。",
         "> 收缩伪计数 w=%.1f，延迟参考值 %.1f s，%d 个 seed 合并统计。\n" % (args.w, args.lat_ref, len(runs)),
         "## 1. λ 扫描\n",
         "| λ | 各题型选择的臂（seed %d） | 测试准确率 | 平均延迟 | 相对全用强模型 |" % runs[0]["seed"],
         "|---|---|---|---|---|"]
    for r in rows:
        desc = "；".join("%s→`%s`" % (KIND_CN.get(k, k), v) for k, v in r["choices"][0].items())
        L.append("| %g | %s | %.1f%% | %.2f s | %+.0f%% 延迟 |"
                 % (r["lam"], desc, 100 * r["acc"], r["lat"],
                    100 * (r["lat"] / base["lat"] - 1)))
    L.append("| — | 全部用 `%s` | %.1f%% | %.2f s | 基准 |"
             % (strong, 100 * base["acc"], base["lat"]))
    L.append("")
    L.append("## 2. 怎么读这张表\n")
    L.append("- **λ=0**：只优化准确率。收缩估计应当使它退化为「全用强模型」，"
             "即与上表最后一行持平——这是收缩修正是否生效的自检。")
    L.append("- **λ>0**：开始用准确率换延迟。延迟下降来自把一些题型交给弱模型"
             "（弱模型单次约 2–3 s，强模型 9–27 s）。")
    L.append("- 本题库对强模型已经封顶（测试集接近满分），因此**准确率没有上升空间**；"
             "路由的价值只体现在「相同的准确率下更低的延迟」，以及为未来更难的题库预留策略框架。")
    L.append("")
    L.append("## 3. 逐题型明细（测试集，跨 seed 合并）\n")
    L.append("| 题型 | " + " | ".join("`%s` 准确率 / 延迟" % a for a in arms) + " |")
    L.append("|---|" + "---|" * len(arms))
    for k in kinds:
        cells = []
        for a in arms:
            ok = sum(r["matrix"][a][k]["ok"] for r in runs)
            n = sum(r["matrix"][a][k]["n"] for r in runs)
            lat = statistics.mean([x for r in runs for x in r["matrix"][a][k]["lat"]])
            cells.append("%.1f%% / %.1fs" % (100 * ok / n, lat))
        L.append("| %s | %s |" % (KIND_CN.get(k, k), " | ".join(cells)))
    L.append("")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("已生成", args.out)


if __name__ == "__main__":
    main()
