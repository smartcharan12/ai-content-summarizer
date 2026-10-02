# AWS Deployment Guide

> 🔑 **Status: not deployed.** Everything below was written and statically validated (hadolint, actionlint +
> shellcheck, Compose-spec schema, JSON syntax), but **none of it has been run against AWS**: it needs your own
> account. Follow the steps in order; each one says what it creates. Commands are for Linux/macOS (bash). On
> Windows, use WSL or Git Bash.

## 1. Architecture (cost-conscious, single instance)

```
                         GitHub (push to main)
                                 │  GitHub Actions: lint → tests → docker build/test
                                 ▼
                    Amazon ECR  ◄── docker push (OIDC role, no stored AWS keys)
                                 │
                                 │  SSM Run Command → deploy/deploy.sh (no SSH, port 22 closed)
                                 ▼
 User ──HTTP:80──►  EC2 t3.large (Amazon Linux 2023, Elastic IP)
                    ┌──────────────────────────────────────────────┐
                    │ nginx (rate limit 10 req/min/IP on /summarize)│
                    │   └─► FastAPI + PEGASUS container (:8000)     │
                    │        └─ HF cache volume (model ~2.3 GB)     │
                    └──────────────────────────────────────────────┘
                                 │ awslogs driver
                                 ▼
                       CloudWatch Logs (/ai-content-summarizer/*)
```

**Why a single EC2 instance?** It is the simplest setup that is still realistic: one VM, Docker, and a reverse proxy.
You can explain every piece, it costs only while running, and you can stop it between demos.

| Option | Good for | Why not (for this project) |
|---|---|---|
| **EC2 + Docker** ✅ | Full control, cheapest always-on single box | You manage the OS (patching), no autoscaling |
| ECS Fargate | Managed containers, autoscaling | Fargate RAM/CPU pricing is high for a 3 GB CPU model; more IAM/network setup |
| ECS on EC2 | Fleet of containers on your instances | Worth it only with several services/instances |
| App Runner | Simplest managed container hosting | Memory limits and per-instance pricing; less to talk about in interviews |
| SageMaker endpoint | Managed model hosting, A/B variants, autoscaling | Most expensive option for a demo; SageMaker-specific packaging |
| Lambda (container image) | Spiky, rare traffic | 2.3 GB model + cold starts of tens of seconds; 10 GB image and memory limits |

**Supporting services**

* **ECR**: private Docker registry. CI pushes images tagged with the git SHA, and EC2 pulls them.
* **S3** (optional): private storage for fine-tuned weights if you don't publish them on the Hugging Face Hub.
* **CloudWatch Logs**: container stdout (JSON lines) from the `awslogs` driver. You can search it with Logs Insights.
* **SSM (Systems Manager)**: runs the deploy script on the instance. There are no SSH keys in GitHub and port 22 stays closed.
* **IAM + OIDC**: GitHub Actions gets short-lived credentials for one role, scoped to your repo's `main` branch.

### Instance sizing

The fp32 model needs about 2.3 GB for its weights, plus PyTorch and the tokenizer: plan for about 3 GB per
container.

| Instance | RAM | Verdict |
|---|---|---|
| t2/t3.micro (free tier) | 1 GB | ❌ cannot load the model |
| t3.medium | 4 GB | ⚠️ too tight with OS + nginx + Docker |
| **t3.large** | 8 GB | ✅ recommended starting point |
| c7i.xlarge | 8 GB, faster CPUs | ➕ if CPU latency is too high |

Latency on these instances is **not measured yet**. Run
`python -m summarizer.pipeline.evaluation_pipeline --model <run_id> --device cpu -n 20` on the instance, or use the
Colab notebook's CPU-latency cell, and report that number.

**Cost.** Check the [EC2 pricing page](https://aws.amazon.com/ec2/pricing/on-demand/) for your region. Stopped
instances don't bill compute, but the EBS volume and an *unattached* Elastic IP do. Create a monthly budget alert
(Billing → Budgets) before launching anything.

---

## 2. Prerequisites

* AWS account, plus [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
  logged in (`aws configure sso` or `aws configure`). Never commit credentials.
* The repo pushed to GitHub, and the fine-tuned model published on the Hugging Face Hub (or in S3, see §6).
* [GitHub CLI](https://cli.github.com/) (`gh auth login`) for setting secrets. The web UI works too.

```bash
export AWS_REGION=ap-south-1                       # pick your region
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export GITHUB_USER=<your-github-username>
export REPO=ai-content-summarizer
cd ai-content-summarizer                           # repo root
```

---

## 3. One-time AWS setup

### 3.1 ECR repository (where CI pushes images)

```bash
aws ecr create-repository --repository-name $REPO --region $AWS_REGION \
  --image-scanning-configuration scanOnPush=true

# Keep only the 10 newest images (old ones cost storage).
aws ecr put-lifecycle-policy --repository-name $REPO --region $AWS_REGION --lifecycle-policy-text '{
  "rules": [{"rulePriority": 1, "description": "keep last 10",
             "selection": {"tagStatus": "any", "countType": "imageCountMoreThan", "countNumber": 10},
             "action": {"type": "expire"}}]}'
```

### 3.2 Instance role (what the EC2 machine may do)

Pull from ECR, be managed by SSM, write logs to CloudWatch.

```bash
aws iam create-role --role-name summarizer-ec2-role \
  --assume-role-policy-document file://deploy/iam/ec2-trust-policy.json
aws iam attach-role-policy --role-name summarizer-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam attach-role-policy --role-name summarizer-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
sed -e "s/<REGION>/$AWS_REGION/g" -e "s/<ACCOUNT_ID>/$ACCOUNT_ID/g" deploy/iam/ec2-cloudwatch-logs.json \
  > /tmp/ec2-logs.json
aws iam put-role-policy --role-name summarizer-ec2-role --policy-name cloudwatch-logs \
  --policy-document file:///tmp/ec2-logs.json

aws iam create-instance-profile --instance-profile-name summarizer-ec2-profile
aws iam add-role-to-instance-profile --instance-profile-name summarizer-ec2-profile \
  --role-name summarizer-ec2-role
```

### 3.3 Security group (HTTP only, no SSH)

```bash
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG_ID=$(aws ec2 create-security-group --group-name summarizer-web --vpc-id $VPC_ID \
  --description "AI Content Summarizer: HTTP only" --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 80 --cidr 0.0.0.0/0
```

### 3.4 Launch the instance

Edit `REPO_URL` and `MODEL_NAME_OR_PATH` at the top of `deploy/ec2-user-data.sh` first. The script installs Docker,
Compose and git, adds 2 GB of swap, clones the repo to `/opt/ai-content-summarizer`, and writes a production `.env`.

```bash
AMI_ID=$(aws ssm get-parameter --region $AWS_REGION \
  --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query Parameter.Value --output text)

INSTANCE_ID=$(aws ec2 run-instances --region $AWS_REGION \
  --image-id $AMI_ID --instance-type t3.large \
  --iam-instance-profile Name=summarizer-ec2-profile \
  --security-group-ids $SG_ID \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=30,VolumeType=gp3}' \
  --metadata-options HttpTokens=required \
  --user-data file://deploy/ec2-user-data.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ai-content-summarizer},{Key=App,Value=ai-content-summarizer}]' \
  --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-status-ok --instance-ids $INSTANCE_ID

# A stable public address that survives stop/start.
ALLOC_ID=$(aws ec2 allocate-address --query AllocationId --output text)
aws ec2 associate-address --instance-id $INSTANCE_ID --allocation-id $ALLOC_ID
PUBLIC_IP=$(aws ec2 describe-addresses --allocation-ids $ALLOC_ID --query 'Addresses[0].PublicIp' --output text)
echo "App URL: http://$PUBLIC_IP"
```

The `App=ai-content-summarizer` tag matters: the CI role may only send deploy commands to instances with that tag.

### 3.5 GitHub → AWS trust (OIDC) and the CI role

```bash
# Once per AWS account. Recent CLI versions don't need --thumbprint-list for GitHub
# (AWS verifies GitHub's certificate itself); if yours insists, see the IAM OIDC docs.
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com --client-id-list sts.amazonaws.com

sed -e "s/<ACCOUNT_ID>/$ACCOUNT_ID/g" -e "s/<GITHUB_USER>/$GITHUB_USER/g" \
  deploy/iam/github-oidc-trust-policy.json > /tmp/gh-trust.json
sed -e "s/<REGION>/$AWS_REGION/g" -e "s/<ACCOUNT_ID>/$ACCOUNT_ID/g" \
  deploy/iam/github-actions-permissions.json > /tmp/gh-perms.json

ROLE_ARN=$(aws iam create-role --role-name summarizer-github-actions \
  --assume-role-policy-document file:///tmp/gh-trust.json --query Role.Arn --output text)
aws iam put-role-policy --role-name summarizer-github-actions --policy-name ci-cd \
  --policy-document file:///tmp/gh-perms.json
```

The trust policy accepts tokens only from **your repo's `main` branch** (and the `production` environment), so a
pull request from a fork can never deploy.

### 3.6 Connect GitHub

```bash
gh secret set AWS_ROLE_ARN --body "$ROLE_ARN"
gh variable set AWS_REGION --body "$AWS_REGION"
gh variable set ECR_REPOSITORY --body "$REPO"
gh variable set APP_URL --body "http://$PUBLIC_IP"
gh variable set AWS_DEPLOY_ENABLED --body "true"
```

Optional: in *Settings → Environments → production*, add yourself as a required reviewer so every deploy waits for
a click.

---

## 4. Deploy

```bash
git push origin main
```

The workflow runs lint → tests → Docker build and container test → push to ECR → SSM deploy. The **first** deploy
downloads the model on the instance, which takes several minutes; `deploy.sh` waits up to 15 minutes for `/health`.

Manual deploy (for example, before CI is set up):

```bash
aws ssm start-session --target $INSTANCE_ID          # browser-less shell, no SSH key needed
sudo -u ec2-user /opt/ai-content-summarizer/deploy/deploy.sh \
  <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/ai-content-summarizer:<git-sha> <REGION>
```

### Verify

```bash
curl http://$PUBLIC_IP/health
curl -X POST http://$PUBLIC_IP/summarize -H "Content-Type: application/json" \
  -d '{"text": "Anna: are we still on for lunch?\nBob: yes, 12:30 at the cafe"}'
open http://$PUBLIC_IP/            # redirects to the web UI (use xdg-open on Linux)
aws logs tail /ai-content-summarizer/api --follow --region $AWS_REGION
```

---

## 5. Operations

| Task | Command |
|---|---|
| Roll back | Re-run `deploy.sh` with an older image tag (every image is tagged with its git SHA) |
| Restart | `docker compose -f docker-compose.yml -f deploy/docker-compose.aws.yml --profile proxy restart` (on the instance) |
| Stop paying for compute | `aws ec2 stop-instances --instance-ids $INSTANCE_ID` (start again with `start-instances`; the Elastic IP stays) |
| Find slow requests | CloudWatch Logs Insights: `fields @timestamp, message \| filter message like /Summarised/ \| sort @timestamp desc` |
| Alarm on failures | CloudWatch alarm on the EC2 `StatusCheckFailed` metric, plus an uptime check on `/health` |

---

## 6. Model weights from S3 instead of the Hugging Face Hub (optional)

```bash
# From the machine that has the trained model:
aws s3 mb s3://<bucket-name> --region $AWS_REGION
aws s3 sync artifacts/models/<run_id> s3://<bucket-name>/models/<run_id>

# Allow the instance to read it: add s3:GetObject + s3:ListBucket on that bucket to summarizer-ec2-role. Then on the instance:
aws s3 sync s3://<bucket-name>/models/<run_id> /opt/ai-content-summarizer/artifacts/models/<run_id>
# and in /opt/ai-content-summarizer/.env:
#   MODEL_SOURCE=path
#   MODEL_NAME_OR_PATH=/app/artifacts/models/<run_id>     (that folder is mounted into the container)
```

## 7. HTTPS (recommended for anything beyond a demo)

* **Cheapest:** point a domain at the Elastic IP and replace nginx with [Caddy](https://caddyserver.com/), which
  gets Let's Encrypt certificates automatically. Open port 443 in the security group.
* **AWS-native:** an Application Load Balancer with an ACM certificate in front of the instance. This also gives health-checked
  routing for the scaling setup below.

## 8. Scaling path

1. **Faster per request:** dynamic int8 quantization or ONNX Runtime export (often several times faster on CPU,
   with a small quality cost; measure ROUGE again), shorter `max_length`, fewer beams (see the experiment table),
   or a distilled model.
2. **Bigger box:** compute-optimised instances (c7i). For heavy traffic, a GPU instance (g5/g6, bf16) handles
   batched requests far faster.
3. **More boxes:** an Auto Scaling group behind an ALB (each instance runs this same Compose stack), or ECS with
   a capacity provider. Scale on CPU or request count. A request queue (SQS) with async workers suits batch jobs.
4. **Avoid repeated work:** cache summaries by input hash (Redis/ElastiCache) for repeated documents.

## 9. Teardown (stop all charges)

```bash
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
aws ec2 wait instance-terminated --instance-ids $INSTANCE_ID
aws ec2 release-address --allocation-id $ALLOC_ID
aws ec2 delete-security-group --group-id $SG_ID
aws ecr delete-repository --repository-name $REPO --force
aws logs delete-log-group --log-group-name /ai-content-summarizer/api
aws logs delete-log-group --log-group-name /ai-content-summarizer/nginx
aws iam remove-role-from-instance-profile --instance-profile-name summarizer-ec2-profile --role-name summarizer-ec2-role
aws iam delete-instance-profile --instance-profile-name summarizer-ec2-profile
aws iam detach-role-policy --role-name summarizer-ec2-role --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam detach-role-policy --role-name summarizer-ec2-role --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam delete-role-policy --role-name summarizer-ec2-role --policy-name cloudwatch-logs
aws iam delete-role --role-name summarizer-ec2-role
aws iam delete-role-policy --role-name summarizer-github-actions --policy-name ci-cd
aws iam delete-role --role-name summarizer-github-actions
gh variable set AWS_DEPLOY_ENABLED --body "false"
```

## 10. Checklist: what is verified vs not

| Item | Status |
|---|---|
| Dockerfile | ✅ hadolint clean. The image's runtime environment (deps + file layout + health check) was reproduced outside Docker and served requests. 🧪 `docker build` not run here (Docker not installed) |
| docker-compose files | ✅ valid against the Compose Specification schema. 🧪 not run |
| CI workflow | ✅ actionlint + shellcheck clean. 🧪 runs on GitHub after you push |
| Deploy/user-data scripts | ✅ shellcheck clean. 🔑 need an EC2 instance |
| IAM policies | ✅ valid JSON, least-privilege by design. 🔑 not applied |
| Latency / cost on AWS | ❓ unknown until deployed and measured |
