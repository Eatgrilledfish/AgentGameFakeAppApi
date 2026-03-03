* agent运行在本地，并启动监听端口

#### 基础信息

* base url： http://localhost:8191
* content-Type： application/json

#### 对话接口

* POST /api/v1/chat
* 参数： 
  | 字段 | 类型 | 必填 | 说明 |
  | model_ip | string | 是 | 模型资源ip，端口为8888 |
  | session_id| string | 是 | 会话id，多轮用例会使用同一个session_id调用agent，注意上下文管理 |
  | message | string | 是 | 用户消息 |

* 响应参数
  | 字段 | 类型 | 说明 |
  | session_id| string  | 会话id |
  | response | string | agent回复消息 |
  | status | string | 处理状态（如success） |
  | tool_results | array | 工具调用结果 |
  | timestamp | int | 时间戳 |
  | duration_ms | int | 处理耗时（ms）|

#### response字段说明

| 场景 | response| 示例 | 
| 普通对话 | 自然语言文本 | "您好，请问有什么可以帮您？" |
| 房源查询完成后 | JSON字符串 | "{\"message\"" \"...\", \"houses\": [\"HF_2101\"]}" |

#### 房源查询返回格式

| 字段 | 类型 | 说明 |
| message | string | 给用户的回复说明 | 
| houses | array | 房源ID列表 |

#### 特别说明

大模型的ip由model_ip字段下发

#### 转发模型请求v1

* POST /vi/chat/completions
* 请求头 
  | 头部 | 必填 | 说明 | 
  | Session-ID | 是 | 评测会话ID（有评测接口生成） |

#### 转发模型请求v2

* POST /vi/chat/completions
* 无需Session-ID，此接口用于不需要评测统计、需要调用模型调用的场景

#### 请求参数

* 与OpenAI API兼容的从Chat Completion请求格式

```json
{
    "model" : "", // 模型可以为空
    "message": [
        {"role": "user", "content": "你好"}
    ],
    "tools" : [...],
    "stream": false
}
```

#### 响应结果

* 与OpenAI API兼容的从Chat Completion响应格式

```json
{
    "id" : "chatcmpl-xxx", 
    "object": "chat.comletion",
    "created": 1234567890,
    "model": "qwen3",
    "choices": [
        {
            "index": 0,
            "message" {
                "role": "assistant",
                "content": "您好！有什么可以帮助您的吗？",
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
