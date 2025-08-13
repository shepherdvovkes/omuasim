FROM postgres:15

# Copy initialization script
COPY init.sql /docker-entrypoint-initdb.d/

# Set environment variables
ENV POSTGRES_DB=omuasim
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=password

# Expose port
EXPOSE 5432

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD pg_isready -U postgres -d omuasim
