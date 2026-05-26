FROM quay.io/astronomer/astro-runtime:12.4.0

ENV ASTRO__PYTHON__DISABLE_ONBUILD=1

USER root

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    curl \
    ant \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -----------------------------
# JAVA ENV
# -----------------------------
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# -----------------------------
# Upgrade Python tooling
# -----------------------------
RUN pip install --upgrade pip setuptools wheel

# -----------------------------
# Install Postgres JDBC driver
# -----------------------------
RUN mkdir -p /opt/spark/jars && \
    curl -L -o /opt/spark/jars/postgresql-42.7.3.jar \
    https://jdbc.postgresql.org/download/postgresql-42.7.3.jar

# -----------------------------
# Install Python dependencies
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER astro