# OpenManus 專案 AI 模型分析與 Google Gemini API 整合指南

## 目前專案 AI 模型使用分析

OpenManus 是一個類似 Manus 的開源實現，允許使用者無需邀請碼即可實現各種 AI 代理功能。經過分析，該專案目前主要通過以下方式使用 AI 模型：

### 1. 核心 LLM 類實現 (app/llm.py)

專案中的 `app/llm.py` 文件定義了一個 `LLM` 類，該類封裝了與 AI 模型的所有交互：

- 使用單例模式管理 LLM 實例
- 支持從配置文件加載 API 認證和模型參數
- 提供消息格式化功能，確保符合 OpenAI API 規範
- 實現重試機制處理 API 請求失敗情況
- 支持基本的文本生成 (`ask` 方法)
- 支持工具調用功能 (`ask_tool` 方法)

### 2. 配置管理 (app/config.py)

專案使用 `app/config.py` 來管理 AI 模型的配置：

- 使用 TOML 格式的配置文件
- 支持多個 LLM 模型配置
- 默認使用 OpenAI 的模型 (目前主要是 GPT-4o)
- 支持為特定任務配置不同的模型 (如視覺任務)

### 3. 消息結構 (app/schema.py)

專案定義了標準化的消息結構，用於與 AI 模型交互，包含角色和內容等屬性。

## Google Gemini API 最新技術概覽

Google Gemini API 提供了一系列功能強大的 AI 模型，分為幾個主要版本：

- **Gemini 1.0/1.5 系列**：包括 Pro、Pro Vision 和 Flash 等變體
- **Gemini 2.0 系列**：包括最新的 Flash 和 Pro 模型

最重要的是，Google 已經提供了兩個 Python SDK：

1. **google-generativeai** (舊版 SDK)：最早的官方 SDK，仍支持所有模型
2. **google-genai** (新版 SDK)：從 Gemini 2.0 開始推薦使用的新 SDK，具有更多功能支持

## 整合 Google Gemini API 到 OpenManus

### 1. 擴展 LLM 類支持 Gemini API

需要在 `app/llm.py` 中添加對 Gemini API 的支持：

```python
from typing import Dict, List, Literal, Optional, Union
import google.generativeai as genai  # 舊版 SDK
# 或使用新版 SDK
# from google import genai as google_genai

# ... 現有導入 ...

class LLM:
    # ... 現有代碼 ...

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):  # 只有在尚未初始化時進行初始化
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature

            # 根據提供商決定使用哪個客戶端
            if llm_config.provider == "openai":
                self.client = AsyncOpenAI(
                    api_key=llm_config.api_key, base_url=llm_config.base_url
                )
                self.provider = "openai"
            elif llm_config.provider == "gemini":
                # 配置 Gemini API
                genai.configure(api_key=llm_config.api_key)
                self.provider = "gemini"
                # 若使用新版 SDK，則：
                # self.genai_client = google_genai.Client(api_key=llm_config.api_key)
                # self.provider = "gemini_new"
            else:
                raise ValueError(f"不支持的提供商: {llm_config.provider}")

    # ... 修改 ask 方法 ...
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """發送提示到 LLM 並獲取回應。"""
        try:
            # 格式化系統和用戶消息
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # 根據提供商選擇不同的實現
            if self.provider == "openai":
                # 現有的 OpenAI 實現...
                if not stream:
                    # 非流式請求
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=temperature or self.temperature,
                        stream=False,
                    )
                    if not response.choices or not response.choices[0].message.content:
                        raise ValueError("從 LLM 返回的響應為空或無效")
                    return response.choices[0].message.content

                # 流式請求
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=temperature or self.temperature,
                    stream=True,
                )

                collected_messages = []
                async for chunk in response:
                    chunk_message = chunk.choices[0].delta.content or ""
                    collected_messages.append(chunk_message)
                    print(chunk_message, end="", flush=True)

                print()  # 流式輸出後換行
                full_response = "".join(collected_messages).strip()
                if not full_response:
                    raise ValueError("從流式 LLM 返回的響應為空")
                return full_response

            elif self.provider == "gemini":
                # Gemini API 實現
                # 將消息轉換為 Gemini 格式
                gemini_messages = self._convert_to_gemini_format(messages)

                model = genai.GenerativeModel(self.model)

                if not stream:
                    response = model.generate_content(gemini_messages)
                    return response.text

                # 流式響應
                response = model.generate_content(gemini_messages, stream=True)

                collected_messages = []
                for chunk in response:
                    chunk_text = chunk.text
                    collected_messages.append(chunk_text)
                    print(chunk_text, end="", flush=True)

                print()  # 流式輸出後換行
                full_response = "".join(collected_messages).strip()
                return full_response

            elif self.provider == "gemini_new":
                # 新版 Gemini SDK 實現
                # 將消息轉換為 Gemini 格式
                gemini_content = self._convert_to_new_gemini_format(messages)

                if not stream:
                    response = self.genai_client.models.generate_content(
                        model=self.model,
                        contents=gemini_content,
                        generation_config={
                            "temperature": temperature or self.temperature,
                            "max_output_tokens": self.max_tokens,
                        }
                    )
                    return response.text

                # 流式響應
                response = self.genai_client.models.generate_content(
                    model=self.model,
                    contents=gemini_content,
                    generation_config={
                        "temperature": temperature or self.temperature,
                        "max_output_tokens": self.max_tokens,
                    },
                    stream=True
                )

                collected_messages = []
                for chunk in response:
                    chunk_text = chunk.text
                    collected_messages.append(chunk_text)
                    print(chunk_text, end="", flush=True)

                print()  # 流式輸出後換行
                full_response = "".join(collected_messages).strip()
                return full_response

        except ValueError as ve:
            logger.error(f"驗證錯誤: {ve}")
            raise
        except Exception as e:
            logger.error(f"ask 中出現未預期的錯誤: {e}")
            raise

    # 添加輔助方法來轉換消息格式
    def _convert_to_gemini_format(self, messages):
        """將標準消息轉換為 Gemini API 格式"""
        gemini_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                # Gemini 無直接系統消息，將其作為用戶消息添加
                gemini_messages.append({"role": "user", "parts": [content]})
                gemini_messages.append({"role": "model", "parts": ["我理解了指示。"]})
            elif role == "user":
                gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [content]})

        return gemini_messages

    def _convert_to_new_gemini_format(self, messages):
        """將標準消息轉換為新版 Gemini SDK 的格式"""
        # 新 SDK 的內容格式不同，需要適當轉換
        content_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                # 作為指令添加
                content_parts.append({"text": f"[SYSTEM INSTRUCTION] {content}"})
            elif role == "user":
                content_parts.append({"text": content})
            elif role == "assistant":
                content_parts.append({"text": f"[ASSISTANT] {content}"})

        return content_parts

    # ... 類似地修改 ask_tool 方法 ...
```

### 2. 更新配置定義 (app/config.py)

需要擴展 `LLMSettings` 類支持 Gemini 配置：

```python
class LLMSettings(BaseModel):
    provider: str = Field("openai", description="AI 提供商 (openai, gemini)")
    model: str = Field(..., description="模型名稱")
    base_url: str = Field(..., description="API 基礎 URL")
    api_key: str = Field(..., description="API 金鑰")
    max_tokens: int = Field(4096, description="每次請求的最大令牌數")
    temperature: float = Field(1.0, description="採樣溫度")
```

### 3. 更新範例配置 (config/config.example.toml)

```toml
# 全局 LLM 配置
[llm]
provider = "openai"
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."
max_tokens = 4096
temperature = 0.0

# 可選的特定 LLM 模型配置
[llm.vision]
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."

# Google Gemini 配置
[llm.gemini]
provider = "gemini"
model = "gemini-1.5-flash"
base_url = ""  # 不需要，使用默認
api_key = "..."
max_tokens = 4096
temperature = 0.0

# 新版 Gemini 2.0 配置
[llm.gemini2]
provider = "gemini_new"
model = "gemini-2.0-flash"
base_url = ""  # 不需要，使用默認
api_key = "..."
max_tokens = 4096
temperature = 0.0
```

### 4. 安裝相依性

更新 `requirements.txt` 添加 Gemini API 的依賴項：

```
# 現有依賴...
google-generativeai>=0.5.0  # 舊版 Gemini SDK
google-genai>=1.0.0  # 新版 Gemini SDK (Gemini 2.0+)
```

## Gemini API 模型選項

Google Gemini API 提供多種模型選項，可根據需求選擇適合的模型：

### Gemini 1.5 系列

- `gemini-1.5-pro` - 全能型高性能模型
- `gemini-1.5-flash` - 輕量快速模型
- `gemini-1.5-pro-vision` - 支持視覺輸入的模型

### Gemini 2.0 系列 (最新)

- `gemini-2.0-pro` - 最新的高性能模型
- `gemini-2.0-flash` - 經濟高效的快速回應模型

## 遷移考慮事項

1. **格式轉換**：OpenAI 和 Gemini API 的消息格式不同，需要轉換
2. **系統消息**：Gemini API 不直接支持系統消息角色，需要模擬實現
3. **流式響應**：兩個 API 的流式響應機制不同
4. **工具調用**：Gemini API 的工具調用與 OpenAI 有所不同
5. **錯誤處理**：需要處理 Gemini API 特有的錯誤類型

## 結論

通過上述修改，OpenManus 專案可以無縫支持 Google Gemini API，為用戶提供更多 AI 模型選擇。這種多模型支持策略可以讓專案在不同場景下選擇最適合的模型，並避免對單一 AI 提供商的依賴。

Google Gemini API 的集成也將使 OpenManus 專案能夠利用 Google 最新的 AI 技術，包括更強的多模態能力和更優的性價比選項。

---

**備註**：實際集成時可能需要根據專案的特定需求和架構進行調整。上述代碼僅供參考，可能需要進一步測試和優化。
