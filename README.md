# Stable Diffusion + ChatGPTのデモ
簡単なテキスト入力からStable diffusionのプロンプトを作成した後、  
Stability AI APIを使用して画像生成を行う


## 使用方法
- 必要なライブラリのインストール

```bash
poetry install
```

- makeコマンドで実行

Stability AI API Core (V2)
```bash
make run-core
```

Stability AI API V1 (txt2img)
```bash
make run-v1
```

- UI上からプロンプトを入力してボタンを押下 => 画像生成

## ローカル環境でStable Diffusionを実行する場合
自身のPC内でStable Diffusion Web UIをgithubからクローンして  
環境構築してください。
その後、stability_v1.pyで関数に渡すapi_versionの値を"local"にすれば実行可能です。
