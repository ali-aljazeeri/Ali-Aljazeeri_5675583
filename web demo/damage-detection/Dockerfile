# Stage 1: Composer dependencies
FROM composer:2.6 AS composer

WORKDIR /app
COPY composer.* ./

# Install production dependencies only
RUN composer install --no-scripts --no-autoloader --prefer-dist --ignore-platform-reqs

# Copy the rest of the application code
COPY . .

# Generate optimized autoloader
RUN composer dump-autoload --optimize

# Stage 2: Node.js build
FROM node:22-alpine AS node

WORKDIR /app
COPY package*.json ./
RUN npm ci

ARG VITE_PUBLIC_ZERO_SERVER
ENV VITE_PUBLIC_ZERO_SERVER=$VITE_PUBLIC_ZERO_SERVER
ENV NODE_ENV=production

COPY . .
RUN npm run build

# Stage 3: Production image
FROM webdevops/php-nginx:8.4-alpine

# Disable unused PHP modules
ENV PHP_DISMOD=bz2,calendar,ffi,intl,gettext,ldap,mysqli,imap,soap,sockets,sysvmsg,sysvsm,sysvshm,shmop,xsl,zip,gd,apcu,vips,yaml,mongodb,amqp

# Install Node.js
RUN apk add --no-cache nodejs npm

# Configure PHP and Nginx
ENV WEB_DOCUMENT_ROOT=/app/public \
    WEB_DOCUMENT_INDEX=index.php \
    PHP_MEMORY_LIMIT=256M \
    PHP_MAX_EXECUTION_TIME=600 \
    PHP_POST_MAX_SIZE=50M \
    PHP_UPLOAD_MAX_FILESIZE=50M \
    PHP_OPCACHE_MEMORY_CONSUMPTION=128 \
    PHP_OPCACHE_MAX_ACCELERATED_FILES=4000 \
    PHP_OPCACHE_VALIDATE_TIMESTAMPS=0 \
    PHP_OPCACHE_REVALIDATE_FREQ=0 \
    PHP_OPCACHE_INTERNED_STRINGS_BUFFER=8 \
    FPM_PM_MAX_CHILDREN=10 \
    FPM_PM_START_SERVERS=2 \
    FPM_PM_MIN_SPARE_SERVERS=1 \
    FPM_PM_MAX_SPARE_SERVERS=3

# Copy Nginx configuration
COPY docker/nginx/vhost.conf /opt/docker/etc/nginx/vhost.conf

# Set up application
WORKDIR /app

# Copy application files
COPY --chown=application:application . .
COPY --from=composer --chown=application:application /app/vendor ./vendor
COPY --from=node --chown=application:application /app/public/build ./public/build
COPY --from=node --chown=application:application /app/node_modules ./node_modules
#COPY --from=node --chown=application:application /app/bootstrap/ssr ./bootstrap/ssr

# Create storage directory structure
RUN mkdir -p storage/framework/sessions \
    && mkdir -p storage/framework/views \
    && mkdir -p storage/framework/cache \
    && mkdir -p storage/logs

# Optimize Laravel
RUN php artisan cache:clear \
    && php artisan config:clear \
    && php artisan view:clear \
    && php artisan route:cache

RUN chown -R application:application storage \
    && chown -R application:application bootstrap/cache \
    && chmod -R 755 storage bootstrap/cache

USER application

#RUN php artisan cache:clear \
#    && php artisan config:clear \
#    && php artisan view:clear \
#    && php artisan route:cache


EXPOSE 80
