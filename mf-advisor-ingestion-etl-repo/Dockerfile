FROM quay.io/astronomer/astro-runtime:12.4.0

# Disable Astronomer's ONBUILD automatic pip install
ENV ASTRO__PYTHON__DISABLE_ONBUILD=1

USER root

# Upgrade pip, setuptools, wheel (fix Python 3.12 incompatibility)
RUN pip install --upgrade pip setuptools wheel

# Install OpenJDK-17 and download Postgres JDBC driver
RUN apt-get update --fix-missing && \
    apt-get install -y openjdk-17-jdk ant curl && \
    mkdir -p /usr/local/airflow/include/jars && \
    curl -L -o /usr/local/airflow/include/jars/postgresql-42.7.3.jar \
        https://jdbc.postgresql.org/download/postgresql-42.7.3.jar && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install Python deps manually
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
RUN export JAVA_HOME

USER astro
