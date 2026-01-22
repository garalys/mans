# Use powershell

# Variables
$AWS_REGION = "eu-north-1"
$AWS_ACCOUNT_ID = "391313099333"
$ECR_REPO_NAME = "rlb"
$IMAGE_TAG = "latest"

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

# Create ECR repository if it doesn't exist
aws ecr create-repository --repository-name $ECR_REPO_NAME --region $AWS_REGION 2>$null

# Build Docker image
docker build -t "${ECR_REPO_NAME}:${IMAGE_TAG}".

docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"

docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"

Write-Host "Image pushed successfully!" -ForegroundColor Green
