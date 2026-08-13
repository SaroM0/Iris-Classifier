# Deliberately a normal Python image rather than a Lambda base image. The
# Lambda Web Adapter (copied in below) runs as an extension that translates
# each invocation into an ordinary HTTP request against a server listening
# on this container, so Django runs unmodified behind Gunicorn — the same
# way it would on any other host. Nothing in the application knows about
# Lambda.
#
# Built as a container image rather than a .zip because scipy, scikit-learn
# and numpy come to roughly 230 MB together, and the .zip limit is 250 MB
# uncompressed. The container limit is 10 GB.
FROM public.ecr.aws/docker/library/python:3.11-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    AWS_LWA_PORT=8080 \
    DJANGO_SETTINGS_MODULE=mysite.settings_prod

WORKDIR /var/task

COPY deploy/requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt

# The project root (manage.py, mysite/, myapp/, templates/, db.sqlite3).
# db.sqlite3 ships as the read-only seed that settings_prod copies to /tmp.
COPY mysite/ /var/task/

# Baked into the image so no request pays for it. The key here is used only
# by this command; the real one arrives as an environment variable.
RUN DJANGO_SECRET_KEY=build-only python manage.py collectstatic --noinput

# One worker: Lambda sends a single request to an instance at a time, so
# extra workers would only multiply the memory cost of loading scikit-learn.
CMD ["gunicorn", "--bind=0.0.0.0:8080", "--workers=1", "--threads=4", "--timeout=60", "--access-logfile=-", "mysite.wsgi:application"]
