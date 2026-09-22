# metask-jev-4b API 调用说明

**端点**：`http://58.211.6.133:10420/v1/systemone`
**协议**：TypeSafe Jev 兼容（官方 harness 的 typesafe adapter 可直连）
**模型**：metask-jev-4b 多语言版（16 locales），单次前向返回所有选项的校准概率
**无需认证**。所有响应 JSON，UTF-8。

---

## 1. 判定请求

```bash
POST /v1/systemone
Content-Type: application/json
```

### 请求体

```jsonc
{
  "state": "待判断的上下文文本（工单/政策/对话/JSON 序列化均可）",
  "questions": {                      // 可一次问多个字段（共享一次 prefill）
    "字段名": {
      "type": "boolean | enum | score",
      "description": "问题（或用 instructions，二选一）",
      "criteria": { "选项值": "选项说明", ... }   // boolean 可省略
    }
  }
}
```

### 三种题型

**boolean**（二选一，true/false 概率）

```json
{"state": "商品7天内无理由退货，已拆封但未使用。",
 "questions": {"eligible": {"type": "boolean",
                            "description": "是否仍可无理由退货？"}}}
```

**enum / choice**（N 选一，criteria 的 key 就是选项值，1–26 个）

```json
{"state": "明天早上九点叫我起床",
 "questions": {"domain": {"type": "enum",
                          "description": "该由哪个助手领域处理？",
                          "criteria": {"alarm": "闹钟", "weather": "天气", "cooking": "菜谱"}}}}
```

**score**（评分档位，criteria 用数组按序给档位说明，档位标签为 "0".."N-1"）

```json
{"state": "这篇总结覆盖了三个论点，第二处数据引用有误。",
 "questions": {"accuracy": {"type": "score",
                            "description": "事实准确性打分（0-4）",
                            "criteria": ["严重失实", "多处错误", "个别错误", "基本准确", "完全准确"]}}}
```

### 响应

```jsonc
{
  "answers": {
    "eligible": {                    // 与请求字段名一一对应
      "type": "boolean",
      "probabilities": {"false": 0.109, "true": 0.891}
    },
    "risk_level": {
      "type": "enum",
      "probabilities": {"low": 0.129, "medium": 0.548, "high": 0.324}
    }
  },
  "model": "metask-jev-4b",
  "usage": {"provider": "self-hosted", "tariff": "none"}
}
```

boolean 额外有 `"noul": P(true)` 顶层便捷字段；choice（请求 type 写 `choice` 时）
额外有 `"choice": "<argmax 选项值>"`。

### 错误

| HTTP | 含义 | 示例 |
|---|---|---|
| 200 | 成功（部分字段失败时带 `partial_errors`） | |
| 422 | state 超过 4096 token，**不截断**，需自行切分 | `{"error": "422 over context limit: ... limit is 4096"}` |
| 500 | 推理/服务错误 | `{"error": {...}}` |
| 400 | JSON 解析失败 | |

题型不合法：`{"error": {"字段名": "supported types are enum and boolean."}}`

---

## 2. 健康检查

```bash
GET /health
```

```json
{"ok": true, "batches": 985, "max_batch": 15,
 "gpu_ms_per_req_p50": 123.0, "qps_window": 28.5}
```

---

## 3. 实测性能（2026-09-22）

| 路径 | 并发 | QPS | p50 | p99 |
|---|---|---|---|---|
| kunshan 本机 | 8 | 63.9 | 123ms | — |
| 公网（Mac→公网） | 1 | 6.9 | 141ms | 224ms |
| 公网 | **8** | **36.4** | **190ms** | 710ms |
| 公网 | 16 | 40.3 | 317ms | 3544ms |

- 单请求 GPU 计算恒定 ~63ms；公网与内网的差 = 网络 RTT
- 多字段请求一次 prefill，字段越多单字段成本越低

---

## 4. 语言

state 可用 16 种语言：en / de / zh / ja / ko / fr / es / it / ru / ar / hi / th / vi / tr / id / nl。
criteria/description 建议与 state 同语言（英文说明+中文 state 也可，模型双语训练）。

---

## 5. 限制

- prompt（state + question 渲染后）≤ **4096 token**，超出 422 不截断
- enum/score 选项 ≤ 26 个
- 概率是校准过的（choice 1.9 / noul 2.375 / score 2.3 温度），可直接当置信度用
- argmax 即预测值；不要自己再 softmax

---

## 6. curl 快速验证

```bash
curl -s -X POST http://58.211.6.133:10420/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"state": "明天早上九点叫我起床",
       "questions": {"domain": {"type": "enum",
                                "description": "该由哪个助手领域处理？",
                                "criteria": {"alarm": "闹钟", "weather": "天气", "cooking": "菜谱"}}}}'
# -> {"answers":{"domain":{"probabilities":{"alarm":0.986,...},"type":"enum"}},...}
```