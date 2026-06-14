"""
Database connection pooling utility for efficient connection management.
"""
import logging
import psycopg2
from psycopg2 import pool
from threading import Lock
from contextlib import contextmanager
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DBConnectionPool:
    """Singleton connection pool for PostgreSQL."""
    
    _instance = None
    _lock = Lock()
    _pool = None
    _connection_config = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DBConnectionPool, cls).__new__(cls)
        return cls._instance

    def initialize(
        self, 
        connection_config: Dict[str, Any],
        min_connections: int = 1,
        max_connections: int = 20
    ) -> None:
        """
        Initialize the connection pool.
        
        Args:
            connection_config: Database connection parameters
            min_connections: Minimum pool size
            max_connections: Maximum pool size
        """
        if self._pool is None:
            with self._lock:
                if self._pool is None:
                    try:
                        self._connection_config = connection_config
                        self._pool = psycopg2.pool.SimpleConnectionPool(
                            min_connections,
                            max_connections,
                            host=connection_config.get("host"),
                            port=connection_config.get("port", 5432),
                            database=connection_config.get("database"),
                            user=connection_config.get("user"),
                            password=connection_config.get("password")
                        )
                        logger.info(
                            f"Connection pool initialized with "
                            f"{min_connections}-{max_connections} connections"
                        )
                    except psycopg2.Error as e:
                        logger.error(f"Failed to initialize connection pool: {e}")
                        raise

    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool as a context manager.
        
        Yields:
            Database connection
            
        Raises:
            RuntimeError: If pool is not initialized
            psycopg2.Error: If connection retrieval fails
        """
        if self._pool is None:
            raise RuntimeError(
                "Connection pool not initialized. "
                "Call initialize() first."
            )
        
        connection = None
        try:
            connection = self._pool.getconn()
            logger.debug("Connection retrieved from pool")
            yield connection
        except psycopg2.Error as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            if connection is not None:
                self._pool.putconn(connection)
                logger.debug("Connection returned to pool")

    def close_all(self) -> None:
        """Close all connections in the pool."""
        if self._pool is not None:
            with self._lock:
                if self._pool is not None:
                    self._pool.closeall()
                    self._pool = None
                    logger.info("Connection pool closed")

    @classmethod
    def reset(cls):
        """Reset the pool (useful for testing)."""
        with cls._lock:
            if cls._instance is not None and cls._instance._pool is not None:
                cls._instance._pool.closeall()
                cls._instance._pool = None
