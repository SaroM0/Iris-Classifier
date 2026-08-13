"""Settings for the AWS Lambda deployment.

Everything here follows from three properties of the Lambda execution
environment, and nothing here is a preference:

  1. The filesystem is read-only apart from /tmp, which is per-instance and
     does not survive a cold start.
  2. Any instance may serve any request, so nothing may be kept in one
     instance that a later request needs.
  3. TLS is terminated before the request arrives, so Django sees plain
     HTTP unless told otherwise.

Development settings are imported wholesale and only the affected values
are overridden, so settings.py stays the single description of the app.
"""
import os
import shutil
from pathlib import Path

from .settings import *  # noqa: F401,F403
from .settings import BASE_DIR, DATABASES, MIDDLEWARE

DEBUG = False

# Required. Fails loudly at import rather than silently falling back to a
# guessable key: every instance must sign session cookies identically, or
# users are logged out at random depending on which instance answers.
SECRET_KEY = os.environ['DJANGO_SECRET_KEY']

# The Function URL host does not exist until the function does, so it is
# supplied by environment variable once deployment has created it.
ALLOWED_HOSTS = [
    host for host in os.environ.get('DJANGO_ALLOWED_HOSTS', '*').split(',') if host
]

# Django 3.2 expects bare hostnames here, not origins with a scheme. Every
# page in this app submits a POST, so getting this wrong means the forms
# return 403 even though the pages render.
CSRF_TRUSTED_ORIGINS = [host for host in ALLOWED_HOSTS if host != '*']

# Without this Django believes the connection is insecure and refuses to
# set cookies marked Secure, which turns login into a redirect loop.
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# Off only for local container testing over plain HTTP.
_secure_cookies = os.environ.get('DJANGO_SECURE_COOKIES', '1') == '1'
SESSION_COOKIE_SECURE = _secure_cookies
CSRF_COOKIE_SECURE = _secure_cookies

# Sessions cannot live in the database here: consecutive requests may land
# on different instances, each holding its own private copy of /tmp. Signing
# them into the cookie makes login independent of which instance answers.
SESSION_ENGINE = 'django.contrib.sessions.backends.signed_cookies'

# Seed the writable copy from the image on cold start. Reads (the user
# table) and writes (the configuration row) then both work, but writes last
# only as long as the instance does — a deliberate trade for a POC, and the
# one thing here that a real deployment would replace with Postgres.
_SEED_DB = Path(BASE_DIR) / 'db.sqlite3'
_LIVE_DB = Path('/tmp/db.sqlite3')
if _SEED_DB.exists() and not _LIVE_DB.exists():
    shutil.copy2(_SEED_DB, _LIVE_DB)
DATABASES['default']['NAME'] = str(_LIVE_DB)

# Nothing sits in front of Lambda to serve static files, so the app serves
# its own. Collected at build time into the image.
STATIC_ROOT = str(Path(BASE_DIR) / 'staticfiles')
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
MIDDLEWARE = [
    MIDDLEWARE[0],  # SecurityMiddleware stays first
    'whitenoise.middleware.WhiteNoiseMiddleware',
    *MIDDLEWARE[1:],
]

# Django's default lands on /accounts/profile/, which this project has no
# route for. Only reached by visitors who go straight to the login page
# rather than being sent there by @login_required, but that is a 404 in the
# middle of the demo.
LOGIN_REDIRECT_URL = '/'

# Log to stdout, which is what CloudWatch captures.
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {'console': {'class': 'logging.StreamHandler'}},
    'root': {'handlers': ['console'], 'level': 'INFO'},
}
