$AWS_REGION = "eu-north-1"
$AWS_ACCOUNT_ID = "391313099333"
$ECR_REPO_NAME = "rlb"
$IMAGE_TAG = "latest"

$TASK_FAMILY    = "rlb-dashboard-creation"
$CONTAINER_NAME = "rlb-dashboard-code"

$ECR_URI = "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
$FULL_IMAGE = "${ECR_URI}/${ECR_REPO_NAME}:${IMAGE_TAG}"

# -----------------------------
# Login, build, push
# -----------------------------
Write-Host "Logging in to ECR..." -ForegroundColor Cyan
aws ecr get-login-password --region $AWS_REGION |
    docker login --username AWS --password-stdin $ECR_URI

Write-Host "Building Docker image..." -ForegroundColor Cyan
docker build -t "${ECR_REPO_NAME}:${IMAGE_TAG}" .

Write-Host "Tagging image..." -ForegroundColor Cyan
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" $FULL_IMAGE

Write-Host "Pushing image..." -ForegroundColor Cyan
docker push $FULL_IMAGE

Write-Host "Image pushed successfully!" -ForegroundColor Green

# -----------------------------
# Fetch task definition
# -----------------------------
Write-Host "Fetching current task definition..." -ForegroundColor Cyan

$taskDef = aws ecs describe-task-definition `
    --task-definition $TASK_FAMILY `
    --region $AWS_REGION `
    | ConvertFrom-Json

$containerDefs = $taskDef.taskDefinition.containerDefinitions

# Update image
foreach ($c in $containerDefs) {
    if ($c.name -eq $CONTAINER_NAME) {
        $c.image = $FULL_IMAGE
    }
}

Write-Host "Registering new task definition revision..." -ForegroundColor Cyan

# Build new task definition payload
$newTaskDef = @{
    family                  = $taskDef.taskDefinition.family
    networkMode             = $taskDef.taskDefinition.networkMode
    requiresCompatibilities = $taskDef.taskDefinition.requiresCompatibilities
    cpu                     = $taskDef.taskDefinition.cpu
    memory                  = $taskDef.taskDefinition.memory
    executionRoleArn        = $taskDef.taskDefinition.executionRoleArn
    taskRoleArn             = $taskDef.taskDefinition.taskRoleArn
    containerDefinitions    = $containerDefs
}

$newTaskDefJson = $newTaskDef | ConvertTo-Json -Depth 20

$tempFile = Join-Path $env:TEMP "taskdef.json"

# Write UTF-8 without BOM (PowerShell 5.1 compatible)
[System.IO.File]::WriteAllText($tempFile, $newTaskDefJson, (New-Object System.Text.UTF8Encoding($false)))

$awsOutput = aws ecs register-task-definition `
    --region $AWS_REGION `
    --cli-input-json file://$tempFile 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to register task definition:" -ForegroundColor Red
    Write-Host $awsOutput
    Remove-Item $tempFile -ErrorAction SilentlyContinue
    exit 1
}

$newRevision = $awsOutput | ConvertFrom-Json

Remove-Item $tempFile -ErrorAction SilentlyContinue

Write-Host "New task definition registered:" -ForegroundColor Green
Write-Host $newRevision.taskDefinition.taskDefinitionArn
