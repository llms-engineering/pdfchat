# pdfchat
《探秘LLM应用开发应用》示例代码


### 本地环境(macos)安装方法：

* 安装unstructured依赖(windows 需要手工下载)
```
mac:
brew intall libmagic  poppler libreoffice pandoc tesseract
linux debian:
apt-get update && apt-get install -y libmagic-dev poppler-utils tesseract-ocr libgl1
```
* 下载nltk包
```
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
```

* 拉取模型
```
ollama pull nomic-embed-text
ollama pull phi3:mini
```

### 生产环境安装方法

* 打包
``` 
cd pdfchat
docker build -t llms-engineering/app:0.0.1 .
```
* 启动
```
docker compose up -d
```
* 查看app_containerId
```
docker ps

CONTAINER ID   IMAGE                              COMMAND                   CREATED             STATUS                   PORTS                                                                                                          NAMES
641101a31ea5   llms-engineering/app:0.0.1         "streamlit run app.p…"    About an hour ago   Up About an hour         0.0.0.0:8501->8501/tcp, :::8501->8501/tcp                                                                      app
cde551d98021   qdrant/qdrant:v1.11.0              "./entrypoint.sh"         About an hour ago   Up About an hour         0.0.0.0:6333-6334->6333-6334/tcp, :::6333-6334->6333-6334/tcp                                                  qdrant
2720bff94ec5   ollama/ollama:0.3.6                "/bin/ollama serve"       About an hour ago   Up About an hour         0.0.0.0:11434->11434/tcp, :::11434->11434/tcp
```
* 拉取模型
```
docker exec -it {app_containerId} ollama run nomic-embed-text
docker exec -it {app_containerId} ollama run phi3:mini
```
*enjoy it!*
