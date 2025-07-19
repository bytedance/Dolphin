# Dolphin Scanner Deployment Guide

This guide covers deploying the Dolphin Scanner service to Kubernetes clusters with GPU support.

## 🚀 Quick Start

### Prerequisites

1. **Kubernetes Cluster** with GPU support
   - NVIDIA GPU nodes with proper drivers
   - GPU device plugin installed
   - Nodes labeled with `accelerator=nvidia-tesla-gpu`

2. **Required Kubernetes Components**
   - NGINX Ingress Controller
   - cert-manager (for TLS certificates)
   - Metrics server (for HPA)
   - Storage class `fast-ssd` available

3. **Command Line Tools**
   - `kubectl` configured for your cluster
   - `docker` for building images

### Deploy to Kubernetes

1. **Build and push the Docker image:**
   ```bash
   docker build -t dolphin/scanner:latest .
   docker push dolphin/scanner:latest
   ```

2. **Deploy using the provided script:**
   ```bash
   cd k8s
   ./deploy.sh
   ```

3. **Check deployment status:**
   ```bash
   ./deploy.sh status
   ```

## 📋 Configuration Options

### Environment Variables

Set these environment variables before deployment:

```bash
export NAMESPACE=production           # Kubernetes namespace
export IMAGE_TAG=v1.0.0              # Docker image tag
export REGISTRY=your-registry.com     # Docker registry
export DRY_RUN=false                  # Set to true for dry run
```

### Cluster Requirements

#### GPU Node Configuration

Ensure your GPU nodes are properly labeled:
```bash
kubectl label nodes <gpu-node-name> accelerator=nvidia-tesla-gpu
```

#### Storage Classes

The deployment requires a `fast-ssd` storage class. Adjust in `pvc.yaml` if needed:
```yaml
storageClassName: your-storage-class
```

## 🔧 Deployment Components

### Core Services

1. **Scanner Deployment** (`deployment.yaml`)
   - 2 replicas with GPU support
   - 4Gi-8Gi memory, 1-2 CPU cores
   - 1 NVIDIA GPU per pod
   - Health checks and security context

2. **Redis Cache** (`redis.yaml`)
   - Single replica for caching
   - 10Gi persistent storage
   - Optimized for LRU eviction

3. **Services** (`service.yaml`)
   - ClusterIP service for internal access
   - Headless service for discovery

### Scaling and Performance

4. **Horizontal Pod Autoscaler** (`hpa.yaml`)
   - Scale 2-10 replicas based on:
     - CPU utilization (70%)
     - Memory utilization (80%)
     - GPU utilization (85%)

5. **Pod Disruption Budget**
   - Ensures at least 1 replica during updates

### Networking and Security

6. **Ingress** (`ingress.yaml`)
   - HTTPS termination with Let's Encrypt
   - Large file upload support (100MB)
   - Custom domain: `dolphin-scanner.example.com`

7. **Network Policy** (`ingress.yaml`)
   - Restricts ingress/egress traffic
   - Allows communication with Redis and ingress

## 🛠 Operations

### Deployment Commands

```bash
# Deploy everything
./deploy.sh deploy

# Check status
./deploy.sh status

# View logs
./deploy.sh logs

# Clean up
./deploy.sh cleanup

# Help
./deploy.sh help
```

### Monitoring

Monitor the deployment with:

```bash
# Pod status
kubectl get pods -l app=dolphin-scanner

# Resource usage
kubectl top pods -l app=dolphin-scanner

# HPA status
kubectl get hpa dolphin-scanner-hpa

# Service endpoints
kubectl get endpoints dolphin-scanner-service
```

### Troubleshooting

#### Common Issues

1. **GPU not available**
   ```bash
   # Check GPU nodes
   kubectl get nodes -l accelerator=nvidia-tesla-gpu
   
   # Check GPU device plugin
   kubectl get pods -n kube-system | grep nvidia
   ```

2. **Image pull errors**
   ```bash
   # Check image exists
   docker pull dolphin/scanner:latest
   
   # Update image pull policy
   kubectl patch deployment dolphin-scanner -p '{"spec":{"template":{"spec":{"containers":[{"name":"scanner","imagePullPolicy":"Always"}]}}}}'
   ```

3. **Storage issues**
   ```bash
   # Check PVC status
   kubectl get pvc
   
   # Check storage class
   kubectl get storageclass
   ```

### Scaling

#### Manual Scaling
```bash
kubectl scale deployment dolphin-scanner --replicas=5
```

#### Update HPA Targets
```bash
kubectl patch hpa dolphin-scanner-hpa -p '{"spec":{"maxReplicas":20}}'
```

## 🔄 Updates and Rollbacks

### Rolling Updates

1. **Update image:**
   ```bash
   kubectl set image deployment/dolphin-scanner scanner=dolphin/scanner:v1.1.0
   ```

2. **Monitor rollout:**
   ```bash
   kubectl rollout status deployment/dolphin-scanner
   ```

### Rollbacks

```bash
# View rollout history
kubectl rollout history deployment/dolphin-scanner

# Rollback to previous version
kubectl rollout undo deployment/dolphin-scanner

# Rollback to specific revision
kubectl rollout undo deployment/dolphin-scanner --to-revision=2
```

## 🧪 Local Development

For local development and testing:

```bash
# Start with Docker Compose
docker-compose up -d

# Build and test locally
docker build -t dolphin/scanner:dev .
docker-compose up --build

# Run tests
python test_runner.py
```

## 📊 Performance Tuning

### GPU Optimization

1. **Batch Size Tuning:**
   - Adjust `DOLPHIN_MAX_BATCH_SIZE` based on GPU memory
   - Monitor GPU utilization with `nvidia-smi`

2. **Memory Management:**
   - Increase pod memory limits for large documents
   - Monitor memory usage patterns

3. **Concurrent Processing:**
   - Scale replicas based on workload
   - Use HPA for automatic scaling

### Storage Optimization

1. **Use fast storage classes for:**
   - Model checkpoints (ReadOnlyMany)
   - Temporary files (EmptyDir with SSD)
   - Results storage (ReadWriteMany)

2. **Cache Strategy:**
   - Redis for frequently accessed data
   - Persistent volumes for long-term storage

## 🔐 Security

### Pod Security

- Non-root user execution
- Read-only root filesystem where possible
- Minimal capabilities
- Security context enforcement

### Network Security

- Network policies restrict traffic
- TLS termination at ingress
- Service-to-service encryption recommended

### Secrets Management

Store sensitive data in Kubernetes secrets:
```bash
kubectl create secret generic scanner-secrets \
  --from-literal=redis-password=your-password \
  --from-literal=api-key=your-api-key
```

## 📞 Support

For deployment issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Review pod logs: `./deploy.sh logs`
3. Verify cluster requirements
4. Check resource limits and requests

---

**Note:** This deployment is optimized for production use with GPU clusters. Adjust resource limits and replica counts based on your specific requirements and cluster capacity. 