## Python環境構築
- uv

```bash
uv init uv-management
```

- Pythonのバージョン指定
```bash
uv python install 3.10 3.11 3.12
```
- venv環境へPythonバージョン指定

```bash
uv venv --python 3.12.0
```

- .tomlファイルに記載されたライブラリと同期
```bash
uv sync
```

- ライブラリの固定
```bash
uv lock
```

- requirements.txtと同期させたい場合
```bash
uv pip sync docs/requirements.txt
```