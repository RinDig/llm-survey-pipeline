# 📚 Deployment Guide for LLM Survey Pipeline

This guide provides step-by-step instructions for deploying the LLM Survey Pipeline to various platforms.

## 🚀 Streamlit Cloud (Recommended)

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at streamlit.io)

### Steps

1. **Fork or Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/llm-survey-pipeline.git
   git push -u origin main
   ```

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub account if not already connected

3. **Configure Deployment**
   - Repository: `YOUR_USERNAME/llm-survey-pipeline`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: Choose a custom URL (e.g., `llm-survey-pipeline`)

4. **Deploy**
   - Click "Deploy"
   - Wait for the build to complete (3-5 minutes)
   - Your app will be live at `https://YOUR_APP_NAME.streamlit.app`

### Advanced Configuration

For production deployments, you may want to:

1. **Set Resource Limits** (in `.streamlit/config.toml`):
   ```toml
   [server]
   maxUploadSize = 200
   maxMessageSize = 200
   ```

2. **Configure Secrets** (if needed):
   - Go to app settings in Streamlit Cloud
   - Add secrets in TOML format
   - These will be available as environment variables

## 🐳 Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build the image
docker build -t llm-survey-pipeline .

# Run the container
docker run -p 8501:8501 llm-survey-pipeline

# With volume for data persistence
docker run -p 8501:8501 -v $(pwd)/data:/app/data llm-survey-pipeline
```

## ☁️ Cloud Platform Deployment

### AWS EC2

1. **Launch EC2 Instance**
   - Choose Ubuntu 22.04 LTS
   - Instance type: t3.medium (minimum)
   - Configure security group to allow port 8501

2. **Setup Instance**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Python
   sudo apt install python3.10 python3-pip python3-venv -y
   
   # Clone repository
   git clone https://github.com/YOUR_USERNAME/llm-survey-pipeline.git
   cd llm-survey-pipeline
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run with nohup
   nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
   ```

3. **Setup Nginx (Optional)**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
       }
   }
   ```

### Google Cloud Run

1. **Create `app.yaml`**
   ```yaml
   runtime: python310
   
   handlers:
   - url: /.*
     script: auto
   
   env_variables:
     STREAMLIT_SERVER_PORT: "8080"
     STREAMLIT_SERVER_ADDRESS: "0.0.0.0"
   ```

2. **Deploy**
   ```bash
   gcloud app deploy
   ```

### Heroku

1. **Create `Procfile`**
   ```
   web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

2. **Create `runtime.txt`**
   ```
   python-3.10.12
   ```

3. **Deploy**
   ```bash
   heroku create llm-survey-pipeline
   git push heroku main
   ```

## 🔧 Environment Variables

For all deployments, ensure these environment variables are NOT set (users provide their own):
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `LLAMA_API_KEY`
- `XAI_API_KEY`
- `DEEPSEEK_API_KEY`

## 📊 Data Persistence

### Local Storage
By default, data is stored in:
- `data/storage/` - JSON survey results
- `data/exports/` - Exported files

### Cloud Storage Options

#### AWS S3
```python
# In backend/storage/json_handler.py
import boto3

def upload_to_s3(file_path, bucket_name):
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket_name, file_path)
```

#### Google Cloud Storage
```python
from google.cloud import storage

def upload_to_gcs(file_path, bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    blob.upload_from_filename(file_path)
```

## 🔒 Security Considerations

1. **HTTPS Only**
   - Always use HTTPS in production
   - Configure SSL certificates

2. **Rate Limiting**
   - Implement rate limiting at the reverse proxy level
   - Use Cloudflare or similar CDN for DDoS protection

3. **Session Security**
   ```python
   # In app.py
   st.set_page_config(
       page_title="LLM Survey Pipeline",
       page_icon="🔬",
       layout="wide",
       initial_sidebar_state="expanded",
       menu_items={
           'Report a bug': "https://github.com/YOUR_USERNAME/llm-survey-pipeline/issues",
           'About': "LLM Survey Pipeline - Research Tool"
       }
   )
   ```

4. **Input Validation**
   - All user inputs are validated
   - API keys are never logged or persisted

## 📈 Monitoring

### Health Check Endpoint
The Streamlit health endpoint is available at:
```
http://your-domain.com/_stcore/health
```

### Logging
Configure logging in your deployment:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Metrics
Consider integrating:
- Google Analytics
- Mixpanel
- Custom metrics to track usage

## 🔄 Continuous Deployment

### GitHub Actions
The repository includes `.github/workflows/deploy.yml` for CI/CD.

To enable automatic deployment to Streamlit Cloud:
1. Go to Streamlit Cloud dashboard
2. Enable "Auto-deploy" for your app
3. Every push to main will trigger a redeploy

## 🆘 Troubleshooting

### Common Issues

1. **"ModuleNotFoundError"**
   - Ensure all dependencies are in `requirements.txt`
   - Check Python version compatibility

2. **"Port already in use"**
   ```bash
   # Find and kill the process
   lsof -i :8501
   kill -9 <PID>
   ```

3. **"Memory issues"**
   - Increase instance size
   - Implement data pagination
   - Clear session state periodically

4. **"API rate limits"**
   - Implement exponential backoff
   - Use queue system for large surveys

### Debug Mode
Run in debug mode locally:
```bash
streamlit run app.py --server.runOnSave true --server.fileWatcherType auto
```

## 📝 Production Checklist

- [ ] HTTPS configured
- [ ] Environment variables secured
- [ ] Data backup strategy implemented
- [ ] Error logging configured
- [ ] Rate limiting enabled
- [ ] Health checks configured
- [ ] Monitoring setup
- [ ] Documentation updated
- [ ] Security review completed
- [ ] Load testing performed

## 📚 Additional Resources

- [Streamlit Deployment Docs](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Cloud Security Guidelines](https://owasp.org/www-project-cloud-security/)

---

For support, please open an issue on [GitHub](https://github.com/YOUR_USERNAME/llm-survey-pipeline/issues).