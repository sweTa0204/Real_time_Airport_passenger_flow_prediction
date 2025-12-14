#!/bin/bash
# ============================================================
# Azure Infrastructure Setup Script
# Real-Time Airport Passenger Flow Prediction System
# ============================================================
# This script creates all necessary Azure resources for deployment
#
# Prerequisites:
#   - Azure CLI installed (az)
#   - Logged in to Azure (az login)
#   - Sufficient permissions to create resources
#
# Usage: ./setup-azure.sh
# ============================================================

set -e  # Exit on error

# Configuration Variables
RESOURCE_GROUP="airport-flow-rg"
LOCATION="eastus"
ACR_NAME="airportflowacr"
CONTAINER_APP_ENV="airport-flow-env"
LOG_ANALYTICS_WORKSPACE="airport-flow-logs"
BACKEND_APP="airport-flow-backend"
FRONTEND_APP="airport-flow-frontend"

echo "============================================================"
echo "🚀 Setting up Azure Infrastructure for Airport Flow Prediction"
echo "============================================================"

# ------------------------------------------------------------
# Step 1: Create Resource Group
# ------------------------------------------------------------
echo ""
echo "📦 Step 1: Creating Resource Group..."
az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION \
    --output none
echo "✅ Resource Group '$RESOURCE_GROUP' created in '$LOCATION'"

# ------------------------------------------------------------
# Step 2: Create Azure Container Registry (ACR)
# ------------------------------------------------------------
echo ""
echo "📦 Step 2: Creating Azure Container Registry..."
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true \
    --output none
echo "✅ Container Registry '$ACR_NAME' created"

# Get ACR credentials for later use
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)

echo "   Login Server: $ACR_LOGIN_SERVER"

# ------------------------------------------------------------
# Step 3: Create Log Analytics Workspace
# ------------------------------------------------------------
echo ""
echo "📊 Step 3: Creating Log Analytics Workspace..."
az monitor log-analytics workspace create \
    --resource-group $RESOURCE_GROUP \
    --workspace-name $LOG_ANALYTICS_WORKSPACE \
    --output none
echo "✅ Log Analytics Workspace created"

LOG_ANALYTICS_WORKSPACE_ID=$(az monitor log-analytics workspace show \
    --resource-group $RESOURCE_GROUP \
    --workspace-name $LOG_ANALYTICS_WORKSPACE \
    --query customerId -o tsv)

LOG_ANALYTICS_KEY=$(az monitor log-analytics workspace get-shared-keys \
    --resource-group $RESOURCE_GROUP \
    --workspace-name $LOG_ANALYTICS_WORKSPACE \
    --query primarySharedKey -o tsv)

# ------------------------------------------------------------
# Step 4: Create Container Apps Environment
# ------------------------------------------------------------
echo ""
echo "🌐 Step 4: Creating Container Apps Environment..."
az containerapp env create \
    --name $CONTAINER_APP_ENV \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --logs-workspace-id $LOG_ANALYTICS_WORKSPACE_ID \
    --logs-workspace-key $LOG_ANALYTICS_KEY \
    --output none
echo "✅ Container Apps Environment created"

# ------------------------------------------------------------
# Step 5: Build and Push Docker Images
# ------------------------------------------------------------
echo ""
echo "🐳 Step 5: Building and pushing Docker images..."

# Build Backend
echo "   Building backend image..."
az acr build \
    --registry $ACR_NAME \
    --image $BACKEND_APP:latest \
    --file docker/Dockerfile.backend \
    . \
    --output none
echo "   ✅ Backend image pushed to ACR"

# Build Frontend
echo "   Building frontend image..."
az acr build \
    --registry $ACR_NAME \
    --image $FRONTEND_APP:latest \
    --file docker/Dockerfile.frontend \
    . \
    --output none
echo "   ✅ Frontend image pushed to ACR"

# ------------------------------------------------------------
# Step 6: Deploy Backend Container App
# ------------------------------------------------------------
echo ""
echo "🚀 Step 6: Deploying Backend Container App..."
az containerapp create \
    --name $BACKEND_APP \
    --resource-group $RESOURCE_GROUP \
    --environment $CONTAINER_APP_ENV \
    --image $ACR_LOGIN_SERVER/$BACKEND_APP:latest \
    --registry-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --target-port 5000 \
    --ingress internal \
    --min-replicas 1 \
    --max-replicas 5 \
    --cpu 1.0 \
    --memory 2.0Gi \
    --env-vars "FLASK_ENV=production" \
    --output none
echo "✅ Backend deployed"

# Get Backend FQDN for frontend configuration
BACKEND_FQDN=$(az containerapp show \
    --name $BACKEND_APP \
    --resource-group $RESOURCE_GROUP \
    --query properties.configuration.ingress.fqdn -o tsv)

# ------------------------------------------------------------
# Step 7: Deploy Frontend Container App
# ------------------------------------------------------------
echo ""
echo "🚀 Step 7: Deploying Frontend Container App..."
az containerapp create \
    --name $FRONTEND_APP \
    --resource-group $RESOURCE_GROUP \
    --environment $CONTAINER_APP_ENV \
    --image $ACR_LOGIN_SERVER/$FRONTEND_APP:latest \
    --registry-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --target-port 80 \
    --ingress external \
    --min-replicas 1 \
    --max-replicas 10 \
    --cpu 0.5 \
    --memory 1.0Gi \
    --env-vars "BACKEND_URL=https://$BACKEND_FQDN" \
    --output none
echo "✅ Frontend deployed"

# Get Frontend URL
FRONTEND_URL=$(az containerapp show \
    --name $FRONTEND_APP \
    --resource-group $RESOURCE_GROUP \
    --query properties.configuration.ingress.fqdn -o tsv)

# ------------------------------------------------------------
# Step 8: Output Summary
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "🎉 DEPLOYMENT COMPLETE!"
echo "============================================================"
echo ""
echo "📋 Resource Summary:"
echo "   Resource Group:     $RESOURCE_GROUP"
echo "   Location:           $LOCATION"
echo "   Container Registry: $ACR_LOGIN_SERVER"
echo ""
echo "🌐 Application URLs:"
echo "   Frontend:  https://$FRONTEND_URL"
echo "   Backend:   https://$BACKEND_FQDN (internal)"
echo ""
echo "🔐 GitHub Secrets Required:"
echo "   ACR_USERNAME: $ACR_USERNAME"
echo "   ACR_PASSWORD: $ACR_PASSWORD"
echo ""
echo "💡 Next Steps:"
echo "   1. Add GitHub secrets for CI/CD"
echo "   2. Configure custom domain (optional)"
echo "   3. Set up monitoring alerts"
echo ""
echo "============================================================"
