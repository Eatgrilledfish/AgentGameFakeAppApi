模型调用接口
IP由判题器的model_ip字段才发到选agent的参数中获取
PORT 8888
转发模型请求（V1）
POST /v1/chat/completions
代理模型API请求，用于评测过程中Agent调用模型。
请求头
头部 必填 说明
Sessio-ID 是 评测会话ID(由评测接口生成)
转发模型请求(V2)
POST /v2/chat/completions
无需 Session-ID：不需要在请求头中传递Session-ID
此接口适用于不需要评测统计、需要调测模型调用的场景。
请求参数
与OpenAI API兼容的Chat Completion请求格式：
"model":"",//模型可以为空
"message":[
{"role":"user":"content":"你好"}
],
"tools":[...],
"stream":false
}
响应结果
与OpenAI API兼容的Chat Completion响应格式
"id":"chatcmpl-xxx",
"object":"chat.completion",
"created":1234567890,
"model":"qwen3-30b-a3b-instruct-2507",
"choices":[
{
"index":0,
"message":{
"role":"assistant",
"content":"你好！有什么可以帮助你的吗？"
},
"finish_reason":"stop"
}
],
"usage":{
"prompt_token":10,
"completion_tokens":20,
"total_tokens":30
}
}