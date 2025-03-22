# langchain-rag-agent

## 主要コンポーネント

- LLM/Chat model:様々なLLMとのインテグレーション
- Prompt Templete:プロンプトのテンプレート
- Output parser:LLMの出力を指定した形式に変換
- Chain:各種コンポーネントを使った処理の連鎖
- RAGに関するコンポーネント
  - Document Loader:データソースからドキュメントを読み込む
  - Document transformer:ドキュメントに何らかの変換をかける
  - Embedding model:ドキュメントをベクトル化する
  - Vector store:ベクトル化したドキュメントの保存先
  - Retriever:入力のテキストと関連するドキュメントを検索する

### Document Loader
- UnstructuredLoader
- DirectoryLoader
- GitLoader
- BigQueryLoader
- GoogleDriveLoader

### Document transformer
-  

### Embedding model
- 

## Runnable
- Ruunableの実行
  - invoke
  - stream
  - batch

## RAGの精度向上Tips
- インデクシング
- 検索クエリ
- 検索語
- 複数のRetrieverを使う
- 生成後

## 参考論文
- [In-context Learning](https://arxiv.org/abs/2301.00234)
- [RAG for LLM Survey](https://arxiv.org/abs/2312.10997)
- [RAG Chatbots](https://arxiv.org/abs/2407.07858)

## 参照情報
- [RAG-AIAgent本リポジトリ](https://github.com/GenerativeAgents/agent-book)