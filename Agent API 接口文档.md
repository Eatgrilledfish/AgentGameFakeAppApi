````markdown
# Agent API 接口文档

> 说明：选手 Agent 程序运行在本地并启动监听端口；启动后将监听端口配置到大赛平台的 Agent 配置中。

---

## 基础信息

- **Base URL**: `http://localhost:8191`
- **Content-Type**: `application/json`

---

## 对话接口

### POST `/api/v1/chat`

#### 请求参数

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| model_ip | string | 是 | 模型资源接口 IP，端口固定为 `8888` |
| session_id | string | 是 | 会话 ID。多轮用例会使用同一个 `session_id` 多次调用 Agent，因此同一 `session_id` 需要做好上下文管理 |
| message | string | 是 | 用户消息 |

#### 请求示例

```bash
curl -X POST http://localhost:8191/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model_ip":"xxx.xxx.xx.x",
    "session_id": "abc123",
    "message": "查询海淀区的房源"
  }'
````

#### 响应参数

| 字段           | 类型     | 说明       |
| ------------ | ------ | -------- |
| session_id   | string | 会话 ID    |
| response     | string | Agent 回复 |
| status       | string | 处理状态     |
| tool_results | array  | 工具调用结果   |
| timestamp    | int    | 时间戳      |
| duration_ms  | int    | 处理耗时（毫秒） |

#### 响应示例

```json
{
  "session_id": "abc123",
  "response": "为您找到海淀区3套房源...",
  "status": "success",
  "tool_results": [
    {
      "name": "bash",
      "success": true,
      "output": "..."
    }
  ],
  "timestamp": 1704067200,
  "duration_ms": 1500
}
```

---

## Agent 输入输出约定

为确保 Agent 能被统一调度，其他实现需遵循以下输入输出规范。

### 输入格式（POST `/api/v1/chat`）

```json
{
  "session_id": "会话ID",
  "message": "用户消息"
}
```

### 输出格式

```json
{
  "session_id": "会话ID",
  "response": "Agent回复内容",
  "status": "success",
  "tool_results": [...],
  "timestamp": 1704067200,
  "duration_ms": 1500
}
```

---

## response 字段说明

| 场景      | response 内容 | 示例                                                    |
| ------- | ----------- | ----------------------------------------------------- |
| 普通对话    | 自然语言文本      | `"您好，请问有什么可以帮您？"`                                     |
| 房源查询完成后 | JSON 字符串    | `"{\"message\": \"...\", \"houses\": [\"HF_2101\"]}"` |

---

## 房源查询返回格式

当完成房源查询后，`response` 字段**必须是合法的 JSON 字符串**，包含以下字段：

| 字段      | 类型     | 说明       |
| ------- | ------ | -------- |
| message | string | 给用户的回复说明 |
| houses  | array  | 房源 ID 列表 |

### 示例（对象形态）

```json
{
  "message": "为您找到以下符合条件的房源：",
  "houses": ["HF_4", "HF_6", "HF_277"]
}
```

---

## 关键规则

* **普通对话**：直接输出自然语言文本
* **房源查询完成后**：`response` 必须是 **JSON 字符串（需转义）**，包含 `message` 和 `houses` 字段
* **JSON 字符串要求**：必须是合法 JSON，不能包含自然语言前缀

---

## 模型调用接口

* **IP**：由判题器通过评测接口请求参数中的 `model_ip` 下发给选手 Agent 获取
* **PORT**：`8888`

---

## 转发模型请求（V1）

### POST `/v1/chat/completions`

代理模型 API 请求，用于评测过程中 Agent 调用模型。

#### 请求头

| 头部         | 必填 | 说明               |
| ---------- | -: | ---------------- |
| Session-ID |  是 | 评测会话 ID（由评测接口生成） |

---

## 转发模型请求（V2）

### POST `/v2/chat/completions`

* **无需 Session-ID**：不需要在请求头中传递 `Session-ID`
* 适用于不需要评测统计、需要调测模型调用的场景

#### 请求参数

与 OpenAI API 兼容的 Chat Completion 请求格式：

```json
{
  "model": "",  // 模型可以为空
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "tools": [...],
  "stream": false
}
```

#### 响应结果

与 OpenAI API 兼容的 Chat Completion 响应格式：

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "qwen3-30b-a3b-instruct-2507",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！有什么可以帮助你的吗？"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

```
```
