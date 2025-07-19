#!/bin/bash

# Dolphin Scanner Kubernetes Deployment Script
# This script deploys the Dolphin Scanner service to a Kubernetes cluster with GPU support

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE=${NAMESPACE:-default}
IMAGE_TAG=${IMAGE_TAG:-latest}
REGISTRY=${REGISTRY:-dolphin}
DRY_RUN=${DRY_RUN:-false}

echo -e "${BLUE}🐬 Dolphin Scanner Kubernetes Deployment${NC}"
echo -e "${BLUE}===========================================${NC}"
echo -e "Namespace: ${YELLOW}${NAMESPACE}${NC}"
echo -e "Image Tag: ${YELLOW}${IMAGE_TAG}${NC}"
echo -e "Registry: ${YELLOW}${REGISTRY}${NC}"
echo -e "Dry Run: ${YELLOW}${DRY_RUN}${NC}"
echo ""

# Functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubectl configuration."
        exit 1
    fi
    
    # Check if GPU nodes are available
    gpu_nodes=$(kubectl get nodes -l accelerator=nvidia-tesla-gpu --no-headers 2>/dev/null | wc -l)
    if [ "$gpu_nodes" -eq 0 ]; then
        log_warning "No GPU nodes found with label 'accelerator=nvidia-tesla-gpu'"
        log_warning "Please ensure your GPU nodes are properly labeled or update the nodeSelector in deployment.yaml"
    else
        log_success "Found $gpu_nodes GPU node(s)"
    fi
    
    log_success "Prerequisites check completed"
}

# Deploy function
deploy_resource() {
    local resource_file=$1
    local resource_name=$(basename "$resource_file" .yaml)
    
    log_info "Deploying $resource_name..."
    
    if [ "$DRY_RUN" = "true" ]; then
        kubectl apply -f "$resource_file" --namespace="$NAMESPACE" --dry-run=client -o yaml
    else
        kubectl apply -f "$resource_file" --namespace="$NAMESPACE"
    fi
    
    log_success "$resource_name deployed"
}

# Main deployment function
deploy() {
    log_info "Starting deployment to namespace: $NAMESPACE"
    
    # Create namespace if it doesn't exist
    if [ "$DRY_RUN" = "false" ]; then
        kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
        log_success "Namespace $NAMESPACE ready"
    fi
    
    # Deploy resources in order
    resources=(
        "configmap.yaml"
        "pvc.yaml"
        "redis.yaml"
        "deployment.yaml"
        "service.yaml"
        "hpa.yaml"
        "ingress.yaml"
    )
    
    for resource in "${resources[@]}"; do
        if [ -f "$resource" ]; then
            deploy_resource "$resource"
        else
            log_warning "Resource file $resource not found, skipping..."
        fi
    done
    
    if [ "$DRY_RUN" = "false" ]; then
        log_info "Waiting for deployment to be ready..."
        kubectl wait --for=condition=available --timeout=300s deployment/dolphin-scanner --namespace="$NAMESPACE"
        kubectl wait --for=condition=available --timeout=300s deployment/redis --namespace="$NAMESPACE"
        
        log_success "All deployments are ready!"
        
        # Show status
        echo ""
        log_info "Deployment Status:"
        kubectl get pods,svc,ingress --namespace="$NAMESPACE" -l app=dolphin-scanner
        kubectl get pods,svc --namespace="$NAMESPACE" -l app=redis
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up Dolphin Scanner deployment..."
    
    resources=(
        "ingress.yaml"
        "hpa.yaml"
        "service.yaml"
        "deployment.yaml"
        "redis.yaml"
        "pvc.yaml"
        "configmap.yaml"
    )
    
    for resource in "${resources[@]}"; do
        if [ -f "$resource" ]; then
            resource_name=$(basename "$resource" .yaml)
            log_info "Removing $resource_name..."
            kubectl delete -f "$resource" --namespace="$NAMESPACE" --ignore-not-found=true
        fi
    done
    
    log_success "Cleanup completed"
}

# Status function
status() {
    log_info "Dolphin Scanner Status in namespace: $NAMESPACE"
    echo ""
    
    echo "Pods:"
    kubectl get pods --namespace="$NAMESPACE" -l app=dolphin-scanner -o wide
    
    echo ""
    echo "Services:"
    kubectl get svc --namespace="$NAMESPACE" -l app=dolphin-scanner
    
    echo ""
    echo "Ingress:"
    kubectl get ingress --namespace="$NAMESPACE" -l app=dolphin-scanner
    
    echo ""
    echo "HPA:"
    kubectl get hpa --namespace="$NAMESPACE" -l app=dolphin-scanner
    
    echo ""
    echo "Redis:"
    kubectl get pods,svc --namespace="$NAMESPACE" -l app=redis
}

# Logs function
logs() {
    local pod_name=$(kubectl get pods --namespace="$NAMESPACE" -l app=dolphin-scanner -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [ -n "$pod_name" ]; then
        log_info "Following logs for pod: $pod_name"
        kubectl logs -f "$pod_name" --namespace="$NAMESPACE"
    else
        log_error "No running pods found for dolphin-scanner"
        exit 1
    fi
}

# Help function
show_help() {
    echo "Dolphin Scanner Kubernetes Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy    Deploy the Dolphin Scanner service (default)"
    echo "  cleanup   Remove the Dolphin Scanner deployment"
    echo "  status    Show deployment status"
    echo "  logs      Follow logs from scanner pods"
    echo "  help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  NAMESPACE     Kubernetes namespace (default: default)"
    echo "  IMAGE_TAG     Docker image tag (default: latest)"
    echo "  REGISTRY      Docker registry (default: dolphin)"
    echo "  DRY_RUN       Set to 'true' for dry run (default: false)"
    echo ""
    echo "Examples:"
    echo "  $0 deploy                    # Deploy with defaults"
    echo "  NAMESPACE=production $0      # Deploy to production namespace"
    echo "  DRY_RUN=true $0             # Dry run deployment"
    echo "  $0 cleanup                  # Remove deployment"
}

# Main script logic
case "${1:-deploy}" in
    deploy)
        check_prerequisites
        deploy
        ;;
    cleanup)
        cleanup
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac 