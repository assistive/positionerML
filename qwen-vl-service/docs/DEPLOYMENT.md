# Qwen 2.5-VL Deployment Guide

## Server Deployment

### Docker Deployment (Recommended)

1. **Build and run with Docker Compose:**
   ```bash
   cd docker
   docker-compose up -d
   ```

2. **Access the service:**
   - API: http://localhost:8000
   - Health: http://localhost:8000/health
   - Metrics: http://localhost:9090 (Prometheus)
   - Dashboard: http://localhost:3000 (Grafana)

### Manual Deployment

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download model:**
   ```bash
   python scripts/download_model.py 7b
   ```

3. **Start service:**
   ```bash
   python scripts/run_service.py --auto-load-model
   ```

## Mobile Deployment

### iOS Deployment

1. **Convert model:**
   ```bash
   python scripts/convert_mobile.py qwen-2.5-vl-3b --platform ios
   ```

2. **Integration steps:**
   - Add the generated `.mlpackage` to your Xcode project
   - Import the CoreML framework
   - Use the provided Swift integration code

### Android Deployment

1. **Convert model:**
   ```bash
   python scripts/convert_mobile.py qwen-2.5-vl-3b --platform android
   ```

2. **Integration steps:**
   - Add the `.tflite` file to your Android assets
   - Add TensorFlow Lite dependencies
   - Use the provided Kotlin integration code

## Production Considerations

### Hardware Requirements

**Server Deployment:**
- GPU: RTX 4090 or A100 (16GB+ VRAM)
- RAM: 32GB+ system memory
- Storage: 100GB+ SSD

**Mobile Deployment:**
- iOS: iPhone 12+ or iPad Pro 2020+
- Android: 6GB+ RAM, preferably flagship device

### Security

1. **Enable API authentication:**
   ```yaml
   authentication:
     api_key_required: true
   ```

2. **Use HTTPS in production**

3. **Configure rate limiting**

### Monitoring

1. **Health checks:** `/health` endpoint
2. **Metrics:** Prometheus metrics at `/metrics`
3. **Logging:** Structured JSON logs
4. **Grafana dashboards:** Pre-configured dashboards available

### Scaling

1. **Horizontal scaling:** Deploy multiple instances behind a load balancer
2. **Model variants:** Use smaller models (3B) for higher throughput
3. **Caching:** Enable Redis caching for repeated requests
