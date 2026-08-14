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

# Everything preparing the seed database runs against the development
# settings on purpose: those point the database at BASE_DIR/db.sqlite3, the
# seed baked into the image, whereas settings_prod points at the /tmp copy
# that this build would then throw away. Set per command rather than with
# ENV, which would leak into the running container and serve production
# with DEBUG on.
RUN DJANGO_SETTINGS_MODULE=mysite.settings python manage.py migrate --noinput

# db.sqlite3 is whatever state the developer's machine left it in, so the
# configuration row could hold any past experiment. Reset it to the
# baseline so every deployment starts from the same known accuracy rather
# than from someone's half-finished tuning session.
RUN DJANGO_SETTINGS_MODULE=mysite.settings python manage.py shell -c "\
from myapp.models import ModelConfiguration; \
from myapp import training; \
c = ModelConfiguration.load(); \
[setattr(c, k, v) for k, v in training.DEFAULTS.items() if k != 'features']; \
[setattr(c, 'use_' + f, f in training.DEFAULTS['features']) for f in training.FEATURE_KEYS]; \
c.save(); \
print('configuration reset to defaults, seed =', c.random_state)"

# The accounts this project ships with have hashed passwords that were
# distributed privately, so nobody can log in to a fresh deployment. This
# adds one account with a known password for the demo.
#
# Deliberately not staff and not superuser: the credentials are public, so
# the account can use the classifier but cannot reach /admin/.
ARG DEMO_USER=demo
ARG DEMO_PASSWORD=iris-demo-2026
RUN DJANGO_SETTINGS_MODULE=mysite.settings python manage.py shell -c "\
from django.contrib.auth.models import User; \
u, _ = User.objects.get_or_create(username='$DEMO_USER'); \
u.set_password('$DEMO_PASSWORD'); \
u.is_staff = False; u.is_superuser = False; u.save(); \
print('demo account ready:', u.username)"

# Baked into the image so no request pays for it. The key here is used only
# by this command; the real one arrives as an environment variable.
RUN DJANGO_SECRET_KEY=build-only python manage.py collectstatic --noinput

# One worker: Lambda sends a single request to an instance at a time, so
# extra workers would only multiply the memory cost of loading scikit-learn.
CMD ["gunicorn", "--bind=0.0.0.0:8080", "--workers=1", "--threads=4", "--timeout=60", "--access-logfile=-", "mysite.wsgi:application"]
