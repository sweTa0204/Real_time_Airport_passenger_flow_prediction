# Azure Deployment Guide
# Real-Time Airport Passenger Flow Prediction System

## 📋 Overview

This guide explains how to deploy the Airport Passenger Flow Prediction system to **Azure Container Apps** - a fully managed serverless container service.

## 🏗️ Architecture on Azure

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AZURE CLOUD                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Azure Container Apps Environment                │   │
│  │  ┌─────────────────────┐    ┌─────────────────────┐        │   │
│  │  │   Frontend App      │    │   Backend App       │        │   │
│  │  │   (Nginx + React)   │───▶│   (Flask + ML)      │        │   │
│  │  │   - External Ingress│    │   - Internal Ingress│        │   │
│  │  │   - Auto-scaling    │    │   - Auto-scaling    │        │   │
│  │  └─────────────────────┘    └─────────────────────┘        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│  ┌──────────────────────────┴───────────────────────────────────┐  │
│  │                    Azure Container Registry                   │  │
│  │                 (Stores Docker Images)                        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  ┌──────────────────────────┴───────────────────────────────────┐  │
│  │                   Log Analytics Workspace                     │  │
│  │               (Monitoring & Logging)                          │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Deployment Options

### Option 1: Automated Script (Recommended)

```bash
# 1. Login to Azure
az login

# 2. Run the setup script
chmod +x azure/setup-azure.sh
./azure/setup-azure.sh
```

### Option 2: ARM Template

```bash
# 1. Login to Azure
az login

# 2. Create Resource Group
az group create --name airport-flow-rg --location eastus

# 3. Deploy using ARM template
az deployment group create \
    --resource-group airport-flow-rg \
    --template-file azure/arm-template.json \
    --parameters environmentName=airport-flow
```

### Option 3: GitHub Actions CI/CD (Automated)

1. **Add GitHub Secrets:**
   - Go to your repo → Settings → Secrets → Actions
   - Add these secrets:
     ```
     AZURE_CREDENTIALS    → Azure Service Principal JSON
     ACR_USERNAME         → Container Registry username
     ACR_PASSWORD         → Container Registry password
     ```

2. **Create Azure Service Principal:**
   ```bash
   az ad sp create-for-rbac \
       --name "airport-flow-github" \
       --role contributor \
       --scopes /subscriptions/{subscription-id}/resourceGroups/airport-flow-rg \
       --sdk-auth
   ```

3. **Push to main branch** - Pipeline triggers automatically!

## 📦 Azure Resources Created

| Resource | Purpose | SKU/Tier |
|----------|---------|----------|
| Resource Group | Container for all resources | N/A |
| Container Registry | Store Docker images | Basic |
| Container Apps Environment | Managed Kubernetes environment | Consumption |
| Backend Container App | Flask API + ML model | 1 CPU, 2GB RAM |
| Frontend Container App | React dashboard | 0.5 CPU, 1GB RAM |
| Log Analytics Workspace | Logging and monitoring | Pay-per-GB |

## 💰 Estimated Costs

| Component | Estimated Monthly Cost |
|-----------|----------------------|
| Container Registry (Basic) | ~$5 |
| Container Apps (Backend) | ~$30-50 |
| Container Apps (Frontend) | ~$15-25 |
| Log Analytics | ~$5-10 |
| **Total** | **~$55-90/month** |

*Note: Costs vary based on traffic and usage. Container Apps only charge when running.*

## 🔧 Configuration

### Environment Variables

**Backend:**
```
FLASK_ENV=production
```

**Frontend:**
```
BACKEND_URL=https://airport-flow-backend.internal.{region}.azurecontainerapps.io
```

### Scaling Configuration

```yaml
Backend:
  minReplicas: 1
  maxReplicas: 5
  scaleRule: 50 concurrent requests

Frontend:
  minReplicas: 1
  maxReplicas: 10
  scaleRule: 100 concurrent requests
```

## 📊 Monitoring

### View Logs
```bash
# Backend logs
az containerapp logs show \
    --name airport-flow-backend \
    --resource-group airport-flow-rg \
    --follow

# Frontend logs
az containerapp logs show \
    --name airport-flow-frontend \
    --resource-group airport-flow-rg \
    --follow
```

### View Metrics
```bash
# Open Azure Portal
az portal dashboard create --name "Airport Flow Dashboard"
```

## 🔒 Security Features

- ✅ **HTTPS by default** - All traffic encrypted
- ✅ **Internal networking** - Backend not exposed to internet
- ✅ **Managed identity** - Secure access to Azure resources
- ✅ **Container Registry authentication** - Private image storage
- ✅ **Log Analytics** - Audit and security logging

## 🔄 Update Deployment

```bash
# Update backend
az containerapp update \
    --name airport-flow-backend \
    --resource-group airport-flow-rg \
    --image airportflowacr.azurecr.io/airport-flow-backend:v2

# Update frontend
az containerapp update \
    --name airport-flow-frontend \
    --resource-group airport-flow-rg \
    --image airportflowacr.azurecr.io/airport-flow-frontend:v2
```

## 🗑️ Cleanup

```bash
# Delete all resources
az group delete --name airport-flow-rg --yes --no-wait
```

## ❓ Troubleshooting

### Container not starting
```bash
# Check container logs
az containerapp logs show --name airport-flow-backend --resource-group airport-flow-rg

# Check revision status
az containerapp revision list --name airport-flow-backend --resource-group airport-flow-rg
```

### 502 Bad Gateway
- Check if backend container is healthy
- Verify internal DNS resolution
- Check nginx configuration

### Slow ML predictions
- Increase backend CPU/Memory
- Consider Azure Machine Learning for model hosting

## 📚 Additional Resources

- [Azure Container Apps Documentation](https://docs.microsoft.com/azure/container-apps/)
- [Azure Container Registry](https://docs.microsoft.com/azure/container-registry/)
- [GitHub Actions for Azure](https://docs.microsoft.com/azure/developer/github/github-actions)
