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
  - HyDE(Hypothetical Document Embeddings)
- 検索後
  - RAG-Fusion
  - 

- 複数のRetrieverを使う
- 生成後


## AIエージェント
- AIエージェントとは、複雑な目標を自律的に遂行できるAIシステムを指す
- Zero-shot-CoTとPlan-and-Solveプロンプティングの実行比較

1. Zero-shot-CoT
```
Q:生徒が20人いるダンスクラブで、20%はコンテンポラリーダンスに登録し、残りの人数のうち25%はジャズダンスに登録し、それ以外はヒップホップダンスに登録しました。
ヒップホップダンスに登録した生徒は何%ですか？

A: ステップバイステップで考えてください。
``` 
2. Plan-and-Solve
```
Q:生徒が20人いるダンスクラブで、20%はコンテンポラリーダンスに登録し、残りの人数のうち25%はジャズダンスに登録し、それ以外はヒップホップダンスに登録しました。
ヒップホップダンスに登録した生徒は何%ですか？

A: はじめに問題を理解し、問題を解決する計画に分割しましょう。その後、各プランを実行して問題をステップバイステップで解決しましょう。
``` 

- 汎用LLMエージェントのフレームワーク
  - LLMエージェント
    LLMが目標を達成するために必要なフロー制御をするメインコントローラーとして動作し、プランニングやメモリ、結果の評価・観測・内省・自己改善、プロンプト最適化などの主要なモジュールを組み合わせたアーキテクチャを使用して、ユーザーからの単純な指示から、内部的に複雑なタスクを順次実行できるのがLLMアプリケーションの中でも特に「LLMエージェント」と呼ばれています。
  - AutoGPT
  - BabyAGI
  - AutoGen
    - AutoGenは高度に抽象化された会話エージェントとコード実行エージェントを複数組み合わせて汎用的なタスクの実行を実現するAIエージェントツール
    - 実現可能なアプリケーション
      - 数学の問題を解く
      - 検索拡張チャット
      - 意思決定
      - マルチエージェントコーディング
      - 動的グループチャット
      - 対話的チェス
  - crewAI
    - 抽象化されたモジュールを組み合わせて自動化エージェントを実現
      - エージェント
      - タスク
      - ツール
      - プロセス
      - クルー
      - メモリ
      - 計画
      - トレーニング
    - 実行時の人間の入力の介入を設定することが可能
    - AgentOps(https://www.agentops.ai/)
    - Langtrace(https://www.langtrace.ai/)

- マルチエージェント・アプローチ
  - マルチステップなマルチエージェント
    一連の処理の中で、複数のシステムプロンプトを使って、役割やステップごとに別々のAIエージェントで処理を行う、ワークフローの最適化を目的とした処理形態
  - マルチロールなマルチエージェント
    異なるペルソナや役割を持たせた複数のエージェントを、目的に向かって協調動作させる形態
- プロンプトエンジニアリングの効果
  AIエージェントを評価する際、Zero-shotプロンプティングだけでどこまで精度を高められるか試しておくことも重要

## 参考論文
- [In-context Learning](https://arxiv.org/abs/2301.00234)
- [RAG for LLM Survey](https://arxiv.org/abs/2312.10997)
- [RAG Chatbots](https://arxiv.org/abs/2407.07858)
- [HyDE関係](https://arxiv.org/abs/2212.10496)
- [Zero Shot plus CoT](https://arxiv.org/abs/2205.11916)
- [MRKL Systems](https://arxiv.org/abs/2205.00445)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [Plan-and-Solve](https://arxiv.org/abs/2305.04091)
- [Experimental Co-Learning 経験的共同学習](https://arxiv.org/abs/2312.17025)
- [MetaGPT](https://arxiv.org/abs/2308.00352)
- [MAGIS](https://arxiv.org/abs/2403.17927)

## 参照情報
- [RAG-AIAgent本リポジトリ](https://github.com/GenerativeAgents/agent-book)
- [Azure AI Foundry with Langchain](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/langchain)