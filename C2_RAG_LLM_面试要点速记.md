# C2 课程面试要点速记（RAG / LLM 应用开发）

## 1. 核心概念

- 数据加载与结构化：RAG 前处理第一步是把原始文档（PDF/TXT）解析成可处理结构（文本 + 元数据）。
- 文本分块（Chunking）：把长文拆成可检索、可向量化的语义单元。
- 分块必要性：受 embedding 模型和 LLM 上下文窗口限制，不分块会截断与信息丢失。
- 块大小权衡：不是越大越好。块太大易语义稀释、召回不准；太小会上下文断裂。
- Lost in the Middle：长上下文中间信息更容易被忽略，合理 chunk + overlap 可以缓解。
- 分块策略家族：
  - 固定/规则分块（快、稳定）
  - 递归分块（更好平衡语义和长度）
  - 语义分块（按语义突变切）
  - 结构化分块（按标题层级，保留章节语义）

## 2. 代码层面的关键点

### 2.1 文档解析（Unstructured）

- `partition`：自动识别文件类型并路由。
- `partition_pdf`：可直接配置 PDF 策略与参数，面试中更能体现工程控制能力。
- `strategy` 典型对比：
  - `hi_res`：更偏版面/结构理解。
  - `ocr_only`：更偏 OCR 抽字。
- C2 示例里对解析结果做了：
  - 元素总量统计
  - `element.category` 分类计数（`Counter`）
  - 全量 element 打印检查质量

### 2.2 分块实现（LangChain）

- `CharacterTextSplitter`：
  - 关键参数：`chunk_size`、`chunk_overlap`
  - 实际行为不是绝对固定长度，而是“按分隔符切 + 合并”。
- `RecursiveCharacterTextSplitter`：
  - 关键是 `separators` 顺序（由粗到细，最后 `""` 兜底）。
  - C2 示例中文分隔符：`["\n\n", "\n", "。", "，", " ", ""]`
- `SemanticChunker`：
  - 依赖 embedding，基于语义距离找断点。
  - 常用参数：
    - `breakpoint_threshold_type`
    - `breakpoint_threshold_amount`
    - `buffer_size`（上下文感知）

### 2.3 嵌入配置

- C2 使用：`HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")`
- `encode_kwargs={"normalize_embeddings": True}`：通常用于余弦相似度场景，提升相似度稳定性。

## 3. 面试官可能会提的问题（基础 -> 中等）

- 为什么 RAG 必须做 chunking？
- `chunk_size` 和 `chunk_overlap` 你怎么定？
- `CharacterTextSplitter` 和 `RecursiveCharacterTextSplitter` 的本质差异是什么？
- 中文语料为什么要定制 `separators`？
- `SemanticChunker` 的断点识别逻辑是什么？
- `percentile`、`standard_deviation`、`interquartile`、`gradient` 有什么差别？
- `partition` 和 `partition_pdf` 的选择依据是什么？
- `hi_res` 和 `ocr_only` 什么时候分别使用？
- 你如何评估“分块质量好不好”？
- 为什么 metadata（页码、标题路径、来源）对检索重要？

## 4. 你应该特别熟悉的实现细节

- `TextLoader.load()` 返回 `List[Document]`。
- `split_documents()` 输出仍是 `Document` 列表，核心字段：
  - `page_content`
  - `metadata`
- `overlap` 的作用是缓解边界语义断裂，但过大会增加冗余和噪声。
- 超长段落处理：
  - 规则分块可能保留大块或有限拆分。
  - 递归分块会继续降级分隔符直到满足约束。
- embedding 维度由模型决定，不建议硬背；可运行代码实测。
- 向量检索调用链（面试常追问）：
  - `docs -> chunks -> embeddings -> vectorstore -> similarity_search`

## 5. 30 秒项目复述模板（可直接说）

“在 C2 我主要做了 RAG 的数据前处理闭环：先用 Unstructured 解析 PDF，再用 Character/Recursive/Semantic 三种策略做分块对比。调参重点是 `partition_pdf` 的解析策略、`chunk_size/chunk_overlap/separators` 以及语义断点阈值。我的核心目标不是追求块数量，而是提升块内语义一致性和检索命中质量，最终让生成阶段拿到更干净、更相关的上下文。”

## 6. 一句话加分总结

- 先用规则分块做可解释基线，再按业务文档复杂度引入语义/结构化分块，并用检索效果反向驱动参数迭代。
