#!/usr/bin/env bash
#
# Finishes the Lambda deployment of the Iris Classifier POC.
#
# The image is already built and pushed to ECR; what remains are the AWS
# resources. Safe to re-run: every step checks for what it would create.
#
#   ./deploy/deploy.sh              # deploy
#   ./deploy/deploy.sh --rollback   # remove everything this script created
#
set -euo pipefail

ACCOUNT=599194859776
REGION=us-east-1
NAME=aidlabs-iris-classifier
IMAGE="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$NAME:latest"

# Reused rather than created, at the user's direction. This is the least
# privileged role that already exists in the account: CloudWatch logs, plus
# staging-only secrets and send-only access to the staging queue. The app
# uses none of it. Swap for a dedicated role when this stops being a POC.
ROLE_ARN="arn:aws:iam::$ACCOUNT:role/aidlabs-api-staging-lambda-role"

if [[ "${1:-}" == "--rollback" ]]; then
    echo "Removing $NAME..."
    aws lambda delete-function-url-config --function-name "$NAME" --region "$REGION" 2>/dev/null || true
    aws lambda delete-function --function-name "$NAME" --region "$REGION" 2>/dev/null || true
    aws ecr delete-repository --repository-name "$NAME" --region "$REGION" --force 2>/dev/null || true
    echo "Done. The IAM role was not created by this script and is left alone."
    exit 0
fi

# --- function ---------------------------------------------------------------
if aws lambda get-function --function-name "$NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "==> Function exists; deploying current image"
    aws lambda update-function-code --function-name "$NAME" --image-uri "$IMAGE" \
        --region "$REGION" --query 'LastModified' --output text
else
    echo "==> Creating function"
    # 50 chars of entropy. Must stay identical across instances or session
    # cookies signed by one are rejected by the next.
    SECRET=$(python3 -c "import secrets;print(secrets.token_urlsafe(50))")
    aws lambda create-function --function-name "$NAME" \
        --package-type Image \
        --code "ImageUri=$IMAGE" \
        --role "$ROLE_ARN" \
        --architectures x86_64 \
        --memory-size 2048 \
        --timeout 30 \
        --description "Iris Classifier POC - Django on Lambda via Web Adapter" \
        --environment "Variables={DJANGO_SECRET_KEY=$SECRET,DJANGO_ALLOWED_HOSTS=*}" \
        --region "$REGION" --query 'FunctionArn' --output text
fi

echo "==> Waiting for the function to become active"
aws lambda wait function-active-v2 --function-name "$NAME" --region "$REGION"
aws lambda wait function-updated-v2 --function-name "$NAME" --region "$REGION"

# --- public URL -------------------------------------------------------------
# AuthType NONE means AWS lets everyone through; Django's @login_required is
# what actually gates the app. Deliberate: this is a public demo.
if ! aws lambda get-function-url-config --function-name "$NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "==> Creating Function URL"
    aws lambda create-function-url-config --function-name "$NAME" \
        --auth-type NONE --region "$REGION" >/dev/null
    # A Function URL alone is not reachable: invoking it still needs an
    # explicit resource policy naming the public principal.
    aws lambda add-permission --function-name "$NAME" \
        --statement-id public-function-url \
        --action lambda:InvokeFunctionUrl \
        --principal '*' \
        --function-url-auth-type NONE \
        --region "$REGION" >/dev/null
fi

URL=$(aws lambda get-function-url-config --function-name "$NAME" --region "$REGION" \
      --query 'FunctionUrl' --output text)
HOST=$(echo "$URL" | sed 's|https://||; s|/$||')

# --- host settings ----------------------------------------------------------
# Only knowable now that the URL exists. Narrows ALLOWED_HOSTS from the '*'
# the function was created with, and gives CSRF the origin it must trust for
# the prediction and configuration forms to be accepted.
echo "==> Pinning ALLOWED_HOSTS to $HOST"
CURRENT_SECRET=$(aws lambda get-function-configuration --function-name "$NAME" \
    --region "$REGION" --query 'Environment.Variables.DJANGO_SECRET_KEY' --output text)
aws lambda update-function-configuration --function-name "$NAME" \
    --environment "Variables={DJANGO_SECRET_KEY=$CURRENT_SECRET,DJANGO_ALLOWED_HOSTS=$HOST}" \
    --region "$REGION" >/dev/null
aws lambda wait function-updated-v2 --function-name "$NAME" --region "$REGION"

# --- smoke test -------------------------------------------------------------
echo "==> Smoke test (the first call pays the cold start)"
CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 "$URL")
LOGIN=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 "${URL}accounts/login/")

echo
echo "  URL          $URL"
echo "  /            $CODE (302 = redirects to login, correct)"
echo "  /accounts/   $LOGIN (200 = login page renders)"
echo
if [[ "$CODE" == "302" && "$LOGIN" == "200" ]]; then
    echo "Deployed. Open $URL in a browser."
else
    echo "Unexpected status. Check logs with:"
    echo "  aws logs tail /aws/lambda/$NAME --follow --region $REGION"
    exit 1
fi
