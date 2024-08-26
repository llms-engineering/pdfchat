# pdfchat
《探秘LLM应用开发应用》示例代码

### 本地开发环境安装方法：

* 安装unstructured依赖(windows 需要手工下载)
```
mac:
brew install libmagic  poppler  tesseract

linux debian:
apt-get update && apt-get install -y libmagic-dev poppler-utils tesseract-ocr libgl1
```
* 安装python依赖（python>3.11+,建议使用conda或者venv）
```
pip install -r requirements.txt
```  
* 下载nltk包
```
python setup_libs.py
```

* 官网下载ollama，然后拉取模型
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
docker exec -it {app_containerId} ollama pull nomic-embed-text
docker exec -it {app_containerId} ollama pull phi3:mini
```
*enjoy it!*
