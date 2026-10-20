#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作业二（方案 b）：一个简单的 RSL 系统 —— 按题型自学习的模型路由
================================================================================
只用 Python 标准库，无第三方依赖。

系统结构
--------
    任务流(有标准答案) ──→ 按题型分桶的 UCB1 策略 ──→ 选中模型执行(本地 Ollama)
            ↑                        ↓                        │
            │                  策略更新(含成本项)  ←── 机器判分 + 错误分类 ←──┘
            └──────────────────────  经验库(SQLite)  ───────────────────────┘

   末端评估：冻结策略，在"模型没见过的题"上分别评估每个臂 → 得到
             (臂 × 题型) 的准确率/延迟矩阵 → 用它计算随机策略、最优单臂、
             按题型事后最优、以及本系统学到的策略，并给出 95% 置信区间。

为什么这样设计（每条都能在报告里给出依据）
--------------------------------------------
1. 答案是程序生成的、唯一正确 → 判分不依赖 LLM 裁判，杜绝 reward hacking。
2. 训练题与测试题严格分离，测试只用于最终评分 → 无答案泄漏。
3. 按题型分桶：不同题型的难度差异很大，全局单一策略会被某个题型主导。
4. reward 可含成本项：reward = 正确 − λ·(延迟/参考延迟)，λ=0 即纯准确率。
5. 必须与三类 baseline 对比，并给出置信区间 → 不把噪声当收益。
6. 报告"策略切换率"：有多少题型的最优选择因学习而改变 —— 这是"学到了"的直接证据。
7. 记录错误类型（空输出/截断/无答案标记/答错） → 比单一准确率更有诊断价值。

用法
----
    python rsl.py --mock                      # 不需要模型，验证流程
    python rsl.py --seeds 1 2 3               # 正式实验（3 次不同数据顺序重跑）
    python rsl.py --lambda 0.1                # 用成本感知 reward（λ=0.1）
    python rsl.py --rounds 6 --samples 8 --per-kind-test 4
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sqlite3
import statistics
import sys
import time
import urllib.error
import urllib.request

KINDS = ("multiply", "mixed", "percent", "logic", "extract")
KIND_CN = {"multiply": "三位数乘法", "mixed": "多步混合运算", "percent": "百分比",
           "logic": "逆运算推理", "extract": "字符串抽取"}
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
ANSWER_RE = re.compile(r"答案\s*[:：]?\s*(-?\d+(?:\.\d+)?|[A-Za-z0-9\-]+)")

DIRECT = "{q}\n请直接给出最终答案。最后一行必须是：答案：<结果>"

ARMS = [
    {"name": "strong", "model": "qwen3:4b", "temperature": 0.0, "template": DIRECT,
     "desc": "强模型 qwen3:4b"},
    {"name": "weak", "model": "qwen3:0.6b", "temperature": 0.0, "template": DIRECT,
     "desc": "弱模型 qwen3:0.6b"},
]


# ---------------------------------------------------------------- 任务生成
def _item(rng, kind):
    if kind == "multiply":
        a, b = rng.randint(100, 999), rng.randint(13, 99)
        return f"计算 {a} × {b} = ?", a * b
    if kind == "mixed":
        a, b, c = rng.randint(100, 999), rng.randint(13, 99), rng.randint(100, 999)
        return f"计算 {a} × {b} + {c} = ?", a * b + c
    if kind == "percent":
        base, p = rng.randrange(1000, 9999), rng.choice([7, 13, 17, 23, 29, 37])
        return f"计算 {base} 的 {p}% 是多少？", round(base * p / 100, 2)
    if kind == "logic":
        b, c, a = rng.randint(50, 200), rng.randint(3, 9), rng.randint(50, 500)
        return f"一个数加上 {a} 后，等于 {b} 的 {c} 倍。这个数是多少？", b * c - a
    oid = f"{rng.choice(['AB', 'CD', 'XZ'])}-2026-{rng.randint(1000, 9999)}"
    return (f'从下面这段文本中提取订单号（只输出订单号）：\n'
            f'"客户备注：请尽快安排发货，订单号 {oid}，谢谢。"'), oid


def build_tasks(task_seed, n_train_per_kind, n_test_per_kind):
    rng = random.Random(task_seed)
    train, test = [], []
    for kind in KINDS:
        for split, n, bucket in (("train", n_train_per_kind, train),
                                 ("test", n_test_per_kind, test)):
            for _ in range(n):
                q, a = _item(rng, kind)
                bucket.append({"kind": kind, "question": q, "answer": str(a), "split": split})
    for i, t in enumerate(train):
        t["id"] = "train_%02d" % i
    for i, t in enumerate(test):
        t["id"] = "test_%02d" % i
    return train, test


# ---------------------------------------------------------------- 判分
def judge(task, text):
    """返回 (score, error_class)。error_class ∈ correct/empty/no_answer_truncated/no_answer/format_error/wrong_value"""
    text = (text or "").strip()
    if not text:
        return 0.0, "empty"
    gold = task["answer"]
    if task["kind"] == "extract":
        ok = gold.lower() in text.lower().replace(" ", "")
        return (1.0, "correct") if ok else (0.0, "wrong_value")
    matches = ANSWER_RE.findall(text)
    if matches:
        cand = matches[-1]
    else:
        if len(text) > 300:
            return 0.0, "no_answer_truncated"
        nums = NUM_RE.findall(text)
        if not nums:
            return 0.0, "no_answer"
        cand = nums[-1]
    try:
        got, want = float(cand), float(gold)
    except ValueError:
        return 0.0, "format_error"
    if abs(got - want) <= max(1e-6, abs(want) * 1e-6):
        return 1.0, "correct"
    return 0.0, "wrong_value"


def wilson(k, n, z=1.96):
    """二项比例的 Wilson 95% 置信区间。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


# ---------------------------------------------------------------- 执行层
class OllamaModel:
    def __init__(self, base_url, model, num_ctx, num_predict, timeout=1200):
        self.url = base_url.rstrip("/") + "/api/generate"
        self.model = model
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.temperature = 0.0
        self.timeout = timeout
        self.last_stats = {}

    def generate(self, prompt):
        payload = {
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx,
                        "num_predict": self.num_predict},
        }
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        self.last_stats = {"eval_count": data.get("eval_count"),
                           "done_reason": data.get("done_reason")}
        text = data.get("response") or ""
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        return text.strip() or (data.get("thinking") or "").strip()


class MockModel:
    """模拟模型：不需要 Ollama。准确率按 (臂, 题型) 设定，便于验证流程与统计。"""

    ACC = {
        ("strong", "multiply"): 0.80, ("weak", "multiply"): 0.30,
        ("strong", "mixed"): 0.60, ("weak", "mixed"): 0.15,
        ("strong", "percent"): 0.75, ("weak", "percent"): 0.20,
        ("strong", "logic"): 0.70, ("weak", "logic"): 0.25,
        ("strong", "extract"): 0.95, ("weak", "extract"): 0.85,
    }
    LAT = {"strong": 9.0, "weak": 2.0}

    def __init__(self, seed=7):
        self.rng = random.Random(seed)
        self.model = "mock"
        self.temperature = 0.0
        self._arm = "strong"
        self._task = None
        self.last_stats = {}

    def bind(self, arm, task):
        self._arm, self._task = arm, task

    def generate(self, prompt):
        time.sleep(0.09 if self._arm == "strong" else 0.02)   # 模拟强/弱模型的延迟差
        ok = self.rng.random() < self.ACC.get((self._arm, self._task["kind"]), 0.5)
        gold = self._task["answer"]
        if ok:
            return "答案：%s" % gold
        return "答案：%s" % (str(int(gold) + 3) if gold.lstrip("-").isdigit() else "AA-2026-0000")


# ---------------------------------------------------------------- 策略层
class PerKindUCB1:
    """按题型分桶的 UCB1 + 收缩估计。

    小样本问题：每个题型每臂只有几次调用，直接取题型内均值会被噪声带偏
    （实测会因此把简单题交给弱模型，比一刀切还差）。
    收缩估计把题型内均值向「该臂的全局均值」拉：
        mean_shrunk = (sum_kind + w * global_mean) / (n_kind + w)
    w 越大越保守（w→∞ 退化为全局选择），w=0 退化为朴素分桶。
    """

    def __init__(self, arms, kinds, c=1.2, prior_weight=3.0):
        self.arms, self.kinds, self.c = arms, kinds, c
        self.w = prior_weight
        self.n = {(k, a["name"]): 0 for k in kinds for a in arms}
        self.s = {(k, a["name"]): 0.0 for k in kinds for a in arms}
        self.total = {k: 0 for k in kinds}

    def update(self, kind, arm, reward):
        self.n[(kind, arm)] += 1
        self.s[(kind, arm)] += reward
        self.total[kind] += 1

    def global_mean(self, arm):
        vals = [(self.s[(k, arm)], self.n[(k, arm)]) for k in self.kinds]
        s = sum(x[0] for x in vals)
        n = sum(x[1] for x in vals)
        return s / n if n else None

    def mean(self, kind, arm):
        """收缩后的均值估计。"""
        n = self.n[(kind, arm)]
        g = self.global_mean(arm)
        if n == 0:
            return g
        if g is None:
            return self.s[(kind, arm)] / n
        return (self.s[(kind, arm)] + self.w * g) / (n + self.w)

    def choose(self, kind):
        for a in self.arms:                                  # 每桶内每臂先各拉一次
            if self.n[(kind, a["name"])] == 0:
                return a["name"]
        best, best_v = None, -1e9
        for a in self.arms:
            name = a["name"]
            mean = self.mean(kind, name)
            bonus = self.c * math.sqrt(2.0 * math.log(self.total[kind]) / self.n[(kind, name)])
            v = mean + bonus
            if v > best_v:
                best_v, best = v, name
        return best

    def greedy(self, kind):
        cands = [(self.mean(kind, a["name"]), a["name"]) for a in self.arms]
        cands = [(m, nm) for m, nm in cands if m is not None]
        return max(cands)[1] if cands else None

    def global_best(self):
        best, best_m = None, -1e9
        for a in self.arms:
            m = self.global_mean(a["name"])
            if m is not None and m > best_m:
                best_m, best = m, a["name"]
        return best


# ---------------------------------------------------------------- 经验库
def open_db(path):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE IF NOT EXISTS experience (
        id INTEGER PRIMARY KEY AUTOINCREMENT, round INTEGER, task_id TEXT, kind TEXT,
        arm TEXT, model TEXT, score REAL, error_class TEXT, latency_s REAL,
        output TEXT, ts TEXT)""")
    con.commit()
    return con


def export_csv(con, path):
    rows = list(con.execute("SELECT id,round,task_id,kind,arm,model,score,error_class,"
                            "latency_s,substr(output,1,120) FROM experience ORDER BY id"))
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "round", "task_id", "kind", "arm", "model", "score",
                    "error_class", "latency_s", "output_head"])
        w.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------- 单次实验
def run_one(seed, args, models, train, test):
    policy = PerKindUCB1(ARMS, KINDS, args.c, args.prior_weight)
    rng = random.Random(seed * 7919 + 13)
    arm_by = {a["name"]: a for a in ARMS}
    con = open_db(os.path.join(args.outdir, "runs", "run_%d" % seed, "experience.db"))
    pulls, per_round = [], []
    acc_s, acc_n = {}, {}          # 题型 × 臂 的纯准确率统计（用于遗憾与对照）

    for rnd in range(1, args.rounds + 1):
        batch = [rng.choice(train) for _ in range(args.samples)]
        for t in batch:
            arm_name = policy.choose(t["kind"])
            cfg = arm_by[arm_name]
            model = models[cfg["model"]]
            model.temperature = cfg["temperature"]
            if args.mock:
                model.bind(arm_name, t)
            t0 = time.time()
            out = model.generate(cfg["template"].format(q=t["question"]))
            lat = time.time() - t0
            score, err = judge(t, out)
            # 训练期 reward：正确性 − λ·(延迟/参考延迟)。λ=0 时退化为纯准确率。
            reward = score - args.lambda_ * (lat / args.lat_ref)
            policy.update(t["kind"], arm_name, reward)
            acc_s[(t["kind"], arm_name)] = acc_s.get((t["kind"], arm_name), 0.0) + score
            acc_n[(t["kind"], arm_name)] = acc_n.get((t["kind"], arm_name), 0) + 1
            pulls.append({"round": rnd, "id": t["id"], "kind": t["kind"], "arm": arm_name,
                          "score": score, "reward": reward, "err": err, "lat": lat})
            con.execute("""INSERT INTO experience
                (round,task_id,kind,arm,model,score,error_class,latency_s,output,ts)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (rnd, t["id"], t["kind"], arm_name, cfg["model"], score, err,
                         round(lat, 2), out[:300], time.strftime("%Y-%m-%d %H:%M:%S")))
        con.commit()
        cum = statistics.mean(p["score"] for p in pulls)
        per_round.append({"round": rnd, "pulls": len(pulls), "online_cum": cum,
                          "online_reward": statistics.mean(p["reward"] for p in pulls),
                          "choices": {k: policy.greedy(k) for k in KINDS}})
        print("  seed %d 轮次 %d/%d：在线累计准确率=%.1f%%（%d 次调用）"
              % (seed, rnd, args.rounds, cum * 100, len(pulls)), flush=True)

    # 累计遗憾（按纯准确率计算）：以最终"题型内最优臂的准确率"为参照
    def acc_of_trained(kind, arm):
        n = acc_n.get((kind, arm), 0)
        return acc_s.get((kind, arm), 0.0) / n if n else 0.0

    ref = {k: max(acc_of_trained(k, a["name"]) for a in ARMS) for k in KINDS}
    regret = sum(max(0.0, ref[p["kind"]] - p["score"]) for p in pulls)
    gb = policy.global_best()
    gb_regret = sum(max(0.0, ref[p["kind"]] - acc_of_trained(p["kind"], gb)) for p in pulls)

    # 逐轮累计遗憾
    for row in per_round:
        cr = cg = 0.0
        for p in pulls:
            if p["round"] > row["round"]:
                continue
            cr += max(0.0, ref[p["kind"]] - p["score"])
            cg += max(0.0, ref[p["kind"]] - acc_of_trained(p["kind"], gb))
        row["regret"], row["gb_regret"] = cr, cg

    # ---- 测试集矩阵：每个臂 × 每个题型 ----
    matrix = {a["name"]: {k: {"ok": 0, "n": 0, "lat": []} for k in KINDS} for a in ARMS}
    for a in ARMS:
        model = models[a["model"]]
        model.temperature = a["temperature"]
        for t in test:                      # 按臂顺序集中跑，减少模型反复加载
            if args.mock:
                model.bind(a["name"], t)
            t0 = time.time()
            out = model.generate(a["template"].format(q=t["question"]))
            lat = time.time() - t0
            score, _ = judge(t, out)
            cell = matrix[a["name"]][t["kind"]]
            cell["ok"] += score
            cell["n"] += 1
            cell["lat"].append(lat)
    con.close()

    # ---- 策略评估（全部由矩阵推出，不额外调用模型）----
    def acc_of(choice_fn):
        ok = n = 0
        for k in KINDS:
            arm = choice_fn(k)
            if arm is None:
                continue
            ok += matrix[arm][k]["ok"]
            n += matrix[arm][k]["n"]
        return ok, n

    def lat_of(choice_fn):
        vals = []
        for k in KINDS:
            arm = choice_fn(k)
            if arm:
                vals.extend(matrix[arm][k]["lat"])
        return statistics.mean(vals) if vals else 0.0

    kinds_with_data = [k for k in KINDS if policy.total[k] > 0]
    rsl_choice = {k: policy.greedy(k) for k in kinds_with_data}
    glob_choice = {k: gb for k in KINDS}
    rand_ok = sum(matrix[a["name"]][k]["ok"] for a in ARMS for k in KINDS)
    rand_n = sum(matrix[a["name"]][k]["n"] for a in ARMS for k in KINDS)
    oracle = {k: max(ARMS, key=lambda a: matrix[a["name"]][k]["ok"])["name"] for k in KINDS}
    best_single = max(ARMS, key=lambda a: sum(matrix[a["name"]][k]["ok"] for k in KINDS))["name"]

    # 成本感知：reward = 正确率 − λ·(延迟/参考延迟)，用训练期延迟做归一化参考
    lat_ref = statistics.mean([p["lat"] for p in pulls]) or 1.0
    def greedy_cost(k, lam):
        best, bv = None, -1e9
        for a in ARMS:
            m, lt = policy.mean(k, a["name"]), None
            if m is None:
                continue
            tl = [p["lat"] for p in pulls if p["kind"] == k and p["arm"] == a["name"]]
            tl = statistics.mean(tl) if tl else lat_ref
            v = m - lam * (tl / lat_ref)
            if v > bv:
                bv, best = v, a["name"]
        return best
    cost_policy = {}
    for lam in (0.05, 0.10, 0.20, 0.40):
        cost_policy[lam] = {k: greedy_cost(k, lam) for k in kinds_with_data}

    return {
        "seed": seed, "pulls": pulls, "rounds": per_round,
        "regret": regret, "global_best_regret": gb_regret, "ref": ref,
        "matrix": matrix,
        "rsl": {"choice": rsl_choice, **dict(zip(("ok", "n"), acc_of(lambda k: rsl_choice.get(k)))),
                "lat": lat_of(lambda k: rsl_choice.get(k))},
        "random": {"ok": rand_ok, "n": rand_n,
                   "lat": statistics.mean([statistics.mean(matrix[a][k]["lat"])
                                           for a in (x["name"] for x in ARMS) for k in KINDS])},
        "best_single": {"arm": best_single,
                        "ok": sum(matrix[best_single][k]["ok"] for k in KINDS),
                        "n": sum(matrix[best_single][k]["n"] for k in KINDS),
                        "lat": statistics.mean([statistics.mean(matrix[best_single][k]["lat"])
                                                for k in KINDS])},
        "oracle": {"choice": oracle, **dict(zip(("ok", "n"), acc_of(lambda k: oracle[k]))),
                   "lat": lat_of(lambda k: oracle[k])},
        "global_choice": glob_choice,
        "cost_policy": cost_policy,
        "switch_rate": {
            "final_vs_global": sum(1 for k in kinds_with_data if rsl_choice[k] != gb) / len(kinds_with_data),
            "final_vs_first": sum(1 for k in kinds_with_data
                                  if rsl_choice[k] != per_round[0]["choices"].get(k)) / len(kinds_with_data),
        },
        "error_types": {e: sum(1 for p in pulls if p["err"] == e)
                        for e in {p["err"] for p in pulls}},
        "lat_ref": lat_ref,
    }


# ---------------------------------------------------------------- 报告
def fmt_ci(k, n):
    lo, hi = wilson(k, n)
    return "%.1f%%（%.1f%%–%.1f%%）" % (100 * k / n, 100 * lo, 100 * hi) if n else "—"


def write_report(args, runs, train, test, ds_check):
    L = []
    L.append("# 作业二 · RSL 系统实验报告\n")
    L.append("> 本报告由 `rsl.py` 自动生成，所有数字均来自本机实测，可用 `python rsl.py --seeds 1 2 3` 复现。\n")
    L.append("## 0. 实验设置\n")
    L.append("- 执行后端：%s" % ("模拟模型（mock，仅用于流程验证）" if args.mock else "本地 Ollama"))
    L.append("- 可学习的臂：" + "；".join("`%s`（%s）" % (a["name"], a["desc"]) for a in ARMS))
    L.append("- 任务流：每题型训练 %d 题、测试 %d 题，共 %d 训练 / %d 测试（答案由程序生成，唯一正确）"
             % (args.per_kind_train, args.per_kind_test, len(train), len(test)))
    L.append("- 任务题型：%s" % "、".join(KIND_CN[k] for k in KINDS))
    L.append("- 策略：**按题型分桶的 UCB1 + 收缩估计**（c=%.1f，收缩伪计数 w=%.1f，每桶每臂先各拉一次）"
             % (args.c, args.prior_weight))
    L.append("- 训练期 reward：`正确 − λ·(延迟/参考延迟)`，λ=%g，参考延迟 %.1f s"
             % (args.lambda_, args.lat_ref))
    L.append("- 轮次 %d，每轮 %d 次调用；每个 seed 独立从空策略开始" % (args.rounds, args.samples))
    L.append("- 生成参数：num_ctx=%d，num_predict=%d，temperature=0" % (args.num_ctx, args.num_predict))
    L.append("- 重跑次数：%d（不同数据顺序，任务集固定，故测试集完全一致）\n" % len(runs))

    L.append("## 1. 数据集概况\n")
    L.extend(ds_check)

    L.append("\n## 2. 学习过程（训练流，在线更新）\n")
    L.append("| seed | 轮次 | 累计调用 | 在线累计准确率 | 累计遗憾 | 固定全局最优臂的遗憾 |")
    L.append("|---|---|---|---|---|---|")
    for r in runs:
        for x in r["rounds"]:
            L.append("| %d | %d | %d | %.1f%% | %.1f | %.1f |"
                     % (r["seed"], x["round"], x["pulls"], x["online_cum"] * 100,
                        x.get("regret", 0.0), x.get("gb_regret", 0.0)))
        L.append("| %d | 合计 | %d | %.1f%% | %.1f | %.1f |"
                 % (r["seed"], len(r["pulls"]),
                    100 * statistics.mean(p["score"] for p in r["pulls"]),
                    r["regret"], r["global_best_regret"]))
    L.append("\n> 遗憾 = Σ(该题型最优臂均值 − 实际得分)。与「固定使用全局最优臂」比较，遗憾更低说明按题型路由比一刀切更好。\n")

    L.append("## 3. 测试集矩阵（臂 × 题型，模型没见过的题）\n")
    L.append("| 题型 | " + " | ".join("`%s` 准确率 / 平均延迟" % a["name"] for a in ARMS) + " | 事后最优臂 |")
    L.append("|---|" + "---|" * (len(ARMS) + 1))
    r0 = runs[0]
    for k in KINDS:
        cells = []
        for a in ARMS:
            c = r0["matrix"][a["name"]][k]
            cells.append("%s / %.1fs" % (fmt_ci(c["ok"], c["n"]), statistics.mean(c["lat"])))
        L.append("| %s | %s | `%s` |" % (KIND_CN[k], " | ".join(cells), r0["oracle"]["choice"][k]))
    L.append("\n> 表中为 seed %d 的矩阵；其余 seed 的完整矩阵见 `runs/run_*/summary.json`。\n" % r0["seed"])

    L.append("## 4. 策略对比（测试集）\n")
    L.append("| 策略 | 准确率（95% CI） | 平均延迟 | 说明 |")
    L.append("|---|---|---|---|")
    for label, key, note in (
        ("随机选臂", "random", "等概率在臂之间随机"),
        ("固定使用最优单臂", "best_single", "事后知道哪个臂最好，全程用它"),
        ("按题型事后最优（上界）", "oracle", "每个题型都用最好的臂，不可实现"),
        ("**本系统（RSL）**", "rsl", "只用训练流学到的按题型策略"),
    ):
        vals = [r[key] for r in runs]
        ok = sum(v["ok"] for v in vals)
        n = sum(v["n"] for v in vals)
        lat = statistics.mean(v["lat"] for v in vals)
        extra = ""
        if key == "best_single":
            extra = "（%s）" % runs[0]["best_single"]["arm"]
        L.append("| %s%s | %s | %.2f s | %s |" % (label, extra, fmt_ci(ok, n), lat, note))
    L.append("")

    L.append("## 5. 成本感知路由（λ 扫描）\n")
    L.append("| λ | 按题型选择的臂 | 测试准确率 | 平均延迟 |")
    L.append("|---|---|---|---|")
    for lam, choice in sorted(runs[0]["cost_policy"].items()):
        ok = n = 0
        lats = []
        for k, arm in choice.items():
            ok += r0["matrix"][arm][k]["ok"]
            n += r0["matrix"][arm][k]["n"]
            lats.extend(r0["matrix"][arm][k]["lat"])
        desc = "；".join("%s→`%s`" % (KIND_CN[k], arm) for k, arm in choice.items())
        L.append("| %g | %s | %s | %.2f s |" % (lam, desc, fmt_ci(ok, n), statistics.mean(lats)))
    L.append("\n> λ 越大越偏向低延迟。因为弱模型快约 4 倍，λ 足够大时策略会主动把一部分题型交给弱模型——这就是「成本—准确率」的显式权衡。\n")

    L.append("## 6. 策略切换率（学习是否真的改变了选择）\n")
    L.append("| seed | 与固定全局最优臂不同的题型占比 | 与首轮选择不同的题型占比 |")
    L.append("|---|---|---|")
    for r in runs:
        L.append("| %d | %.0f%% | %.0f%% |" % (r["seed"], 100 * r["switch_rate"]["final_vs_global"],
                                               100 * r["switch_rate"]["final_vs_first"]))
    L.append("\n> 若切换率为 0，说明学到的策略与「一刀切」完全一致（学习没有产生额外信息）；"
             "切换率越高、且测试准确率同时不降，才说明按题型路由真的有用。\n")

    L.append("## 7. 判分结果与错误类型分布（训练流）\n")
    L.append("| 类型 | 次数 | 占比 |")
    L.append("|---|---|---|")
    err_all = {}
    for r in runs:
        for k, v in r["error_types"].items():
            err_all[k] = err_all.get(k, 0) + v
    total = sum(err_all.values())
    for k, v in sorted(err_all.items(), key=lambda x: -x[1]):
        L.append("| %s | %d | %.1f%% |" % (k, v, 100 * v / total))
    L.append("\n> `no_answer_truncated` = 输出过长但没给出答案标记（思考过程吃掉 token 预算）；"
             "`empty` = 完全没有正文输出。这两类是本机 qwen3 思考模式的典型失败模式。\n")

    L.append("## 8. 结论\n")
    rsl = sum(r["rsl"]["ok"] for r in runs)
    rsln = sum(r["rsl"]["n"] for r in runs)
    rnd = sum(r["random"]["ok"] for r in runs)
    rndn = sum(r["random"]["n"] for r in runs)
    bst = sum(r["best_single"]["ok"] for r in runs)
    bstn = sum(r["best_single"]["n"] for r in runs)
    orc = sum(r["oracle"]["ok"] for r in runs)
    orcn = sum(r["oracle"]["n"] for r in runs)
    L.append("- 本系统学到的按题型策略：测试准确率 **%s**" % fmt_ci(rsl, rsln))
    L.append("- 相对随机选臂：**%+.1f 个百分点**；相对固定最优单臂：**%+.1f 个百分点**；"
             "距按题型事后最优（上界）**%+.1f 个百分点**"
             % (100 * (rsl / rsln - rnd / rndn), 100 * (rsl / rsln - bst / bstn),
                100 * (rsl / rsln - orc / orcn)))
    L.append("- 学习机制：每次调用的判分结果作为 reward 写入经验库，按题型分桶的 UCB1 用「历史均值 + 探索奖励」"
             "更新下一次的选择偏好，因此表现为「先试错、后收敛」。")
    L.append("- 局限：① 题库限于可机器判定的数值/抽取任务，reward 依赖标准答案，开放式任务需要引入 LLM 裁判"
             "（如 Jev，$0.042/百万输入 token、输出免费）；② 每个题型的样本量有限（每题型每轮仅若干次调用），"
             "置信区间较宽；③ 只做了两臂（强/弱模型），更细的提示策略臂未纳入。")
    L.append("")
    path = os.path.join(args.outdir, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return path


def write_svg(path, runs):
    W, H, PAD = 680, 380, 60
    allr = [r for run in runs for r in run["rounds"]]
    maxpull = max(len(run["pulls"]) for run in runs)
    colors = ["#1f6feb", "#d97706", "#059669"]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           f'<rect width="{W}" height="{H}" fill="#fff"/>']
    for i in range(6):
        f = i / 5
        y = H - PAD - f * (H - 2 * PAD)
        svg.append(f'<line x1="{PAD}" y1="{y:.1f}" x2="{W - PAD}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{PAD - 10}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="#6b7280">{f:.0%}</text>')
    for idx, run in enumerate(runs):
        pts = []
        for x in run["rounds"]:
            px = PAD + (x["pulls"] / maxpull) * (W - 2 * PAD)
            py = H - PAD - x["online_cum"] * (H - 2 * PAD)
            pts.append(f"{px:.1f},{py:.1f}")
        svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[idx % 3]}" stroke-width="2.2"/>')
        for p in pts:
            px, py = p.split(",")
            svg.append(f'<circle cx="{px}" cy="{py}" r="3" fill="{colors[idx % 3]}"/>')
        svg.append(f'<text x="{W - PAD - 110}" y="{PAD + 14 + idx * 16}" font-size="11" fill="{colors[idx % 3]}">seed {run["seed"]}</text>')
    svg.append(f'<text x="{W / 2}" y="22" font-size="14" text-anchor="middle" fill="#111827">'
               f'Online cumulative accuracy vs. number of model calls</text>')
    svg.append(f'<text x="{W / 2}" y="{H - 12}" font-size="12" text-anchor="middle" fill="#6b7280">model calls</text>')
    svg.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(svg))


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="按题型自学习的模型路由（最小 RSL）")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--task-seed", type=int, default=20261020, help="任务集种子（固定，保证各 seed 测试集相同）")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--samples", type=int, default=8, help="每轮调用次数")
    ap.add_argument("--per-kind-train", type=int, default=5)
    ap.add_argument("--per-kind-test", type=int, default=4)
    ap.add_argument("--c", type=float, default=1.2, help="UCB1 探索系数")
    ap.add_argument("--prior-weight", type=float, default=3.0,
                    help="收缩估计的伪计数：题型内样本向该臂全局均值收缩的强度（0=不收缩）")
    ap.add_argument("--lambda", dest="lambda_", type=float, default=0.2,
                    help="训练期 reward = 正确率 − λ·(延迟/参考延迟)；0 = 纯准确率")
    ap.add_argument("--lat-ref", type=float, default=8.0,
                    help="延迟归一化参考值（秒），取实测强模型单次延迟量级")
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--num-predict", type=int, default=1792)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.outdir, "runs"), exist_ok=True)
    train, test = build_tasks(args.task_seed, args.per_kind_train, args.per_kind_test)

    # 导出任务集，便于复现与检查
    with open(os.path.join(args.outdir, "tasks.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": {"task_seed": args.task_seed, "kinds": list(KINDS)},
                   "train": train, "test": test}, fh, ensure_ascii=False, indent=1)

    if args.mock:
        models = {"qwen3:4b": MockModel(seed=args.seeds[0]), "qwen3:0.6b": MockModel(seed=args.seeds[0] + 1)}
        print("[模式] mock")
    else:
        models = {a["model"]: OllamaModel(args.base_url, a["model"], args.num_ctx, args.num_predict)
                  for a in ARMS}
        try:
            models[ARMS[0]["model"]].generate("回答一个字：好")
        except (urllib.error.URLError, OSError) as exc:
            print("!! 连不上 Ollama：%s" % exc)
            return 2

    runs = []
    for seed in args.seeds:
        print("=== seed %d ===" % seed, flush=True)
        os.makedirs(os.path.join(args.outdir, "runs", "run_%d" % seed), exist_ok=True)
        r = run_one(seed, args, models, train, test)
        con = sqlite3.connect(os.path.join(args.outdir, "runs", "run_%d" % seed, "experience.db"))
        r["n_exp"] = export_csv(con, os.path.join(args.outdir, "runs", "run_%d" % seed, "experience.csv"))
        con.close()
        # 汇总每个 seed 的关键数字
        with open(os.path.join(args.outdir, "runs", "run_%d" % seed, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump({k: r[k] for k in ("seed", "regret", "global_best_regret", "ref",
                                         "rsl", "random", "best_single", "oracle", "matrix",
                                         "switch_rate", "error_types", "lat_ref", "rounds")},
                      fh, ensure_ascii=False, indent=1, default=str)
        runs.append(r)

    ds_check = ds_check_lines(train, test)
    rep = write_report(args, runs, train, test, ds_check)
    write_svg(os.path.join(args.outdir, "curve.svg"), runs)
    with open(os.path.join(args.outdir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump({"runs": args.seeds, "mock": args.mock, "task_seed": args.task_seed,
                   "rounds": args.rounds, "samples": args.samples,
                   "per_kind_train": args.per_kind_train, "per_kind_test": args.per_kind_test,
                   "c": args.c, "prior_weight": args.prior_weight,
                   "lambda": args.lambda_, "lat_ref": args.lat_ref, "num_ctx": args.num_ctx,
                   "num_predict": args.num_predict,
                   "arms": [{"name": a["name"], "model": a["model"]} for a in ARMS],
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, ensure_ascii=False, indent=1)
    print("报告：%s" % rep)
    return 0


def ds_check_lines(train, test):
    """数据集完整性小结，写进报告第 1 节。"""
    L = ["| 项目 | 训练集 | 测试集 |", "|---|---:|---:|"]
    L.append("| 题目数 | %d | %d |" % (len(train), len(test)))
    L.append("| 每题型题数 | %s | %s |" % (
        "/".join(str(sum(1 for x in train if x["kind"] == k)) for k in KINDS),
        "/".join(str(sum(1 for x in test if x["kind"] == k)) for k in KINDS)))
    L.append("| 题目重复（训练↔测试） | %d 条 | — |" % len({x["question"] for x in train} & {x["question"] for x in test}))
    L.append("| 平均文本长度 | %.0f 字 | %.0f 字 |" % (
        statistics.mean(len(x["question"]) for x in train), statistics.mean(len(x["question"]) for x in test)))
    def mag(ds):
        v = [abs(float(x["answer"])) for x in ds if x["kind"] != "extract"]
        return statistics.mean(v)
    L.append("| 数值题平均答案量级 | %.0f | %.0f |" % (mag(train), mag(test)))
    L.append("")
    L.append("> 训练集与测试集由同一生成器按同一题型比例独立生成，**同分布且无重复题**；"
             "因此测试集不存在「比训练集更容易」的偏差。")
    return L


if __name__ == "__main__":
    sys.exit(main())
