# Build stage — install MkDocs and build the static site
FROM python:3.11-slim AS builder

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir \
    mkdocs-material \
    mkdocs-macros-plugin \
    mkdocs-minify-plugin \
    mkdocs-mermaid2-plugin

RUN mkdocs build

# Serve stage — nginx serving the built site on Railway's $PORT
FROM nginx:alpine

COPY --from=builder /app/site /usr/share/nginx/html
COPY nginx.conf /etc/nginx/templates/default.conf.template

EXPOSE 8080

CMD ["/bin/sh", "-c", "PORT=${PORT:-8080} envsubst '$PORT' < /etc/nginx/templates/default.conf.template > /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"]
