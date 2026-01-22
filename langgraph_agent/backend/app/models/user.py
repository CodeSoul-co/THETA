"""
User Model and Database Operations
Simple SQLite-based user storage
"""

import sqlite3
import aiosqlite
import hashlib
import base64
from pathlib import Path
from typing import Optional
from datetime import datetime
from passlib.context import CryptContext
from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _prehash_password(password: str) -> str:
    """
    Pre-hash password with SHA-256 before bcrypt.
    This allows passwords of any length while staying within bcrypt's 72 byte limit.
    Returns base64 encoded hash (44 characters, well under 72 bytes).
    """
    password_bytes = password.encode('utf-8')
    sha256_hash = hashlib.sha256(password_bytes).digest()
    return base64.b64encode(sha256_hash).decode('ascii')


class User:
    """User model"""
    def __init__(
        self,
        id: int,
        username: str,
        email: str,
        hashed_password: str,
        full_name: Optional[str] = None,
        created_at: Optional[datetime] = None,
        is_active: bool = True
    ):
        self.id = id
        self.username = username
        self.email = email
        self.hashed_password = hashed_password
        self.full_name = full_name
        self.created_at = created_at or datetime.now()
        self.is_active = is_active

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        # Pre-hash with SHA-256 to handle any length password
        prehashed = _prehash_password(plain_password)
        return pwd_context.verify(prehashed, hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash a password"""
        # Pre-hash with SHA-256 to handle any length password (bcrypt limit is 72 bytes)
        prehashed = _prehash_password(password)
        return pwd_context.hash(prehashed)


class UserDB:
    """User database operations"""
    
    def __init__(self):
        self.db_path = Path(settings.DATA_DIR.parent / "users.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    async def initialize(self):
        """Initialize database and create tables"""
        # Check if database file exists and has tables
        if self._initialized and self.db_path.exists():
            # Verify table exists by checking if we can query it
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    async with db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'") as cursor:
                        row = await cursor.fetchone()
                        if row:
                            return  # Database is valid and initialized
            except Exception:
                # Database might be corrupted, reset flag
                self._initialized = False
        
        # Initialize or re-initialize database
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    full_name TEXT,
                    created_at TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_username ON users(username)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_email ON users(email)
            """)
            await db.commit()
        
        self._initialized = True
        logger.info(f"User database initialized at {self.db_path}")

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None
    ) -> User:
        """Create a new user"""
        await self.initialize()
        
        hashed_password = User.get_password_hash(password)
        created_at = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            try:
                cursor = await db.execute("""
                    INSERT INTO users (username, email, hashed_password, full_name, created_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (username, email, hashed_password, full_name, created_at, 1))
                await db.commit()
                user_id = cursor.lastrowid
                
                return User(
                    id=user_id,
                    username=username,
                    email=email,
                    hashed_password=hashed_password,
                    full_name=full_name,
                    created_at=datetime.fromisoformat(created_at),
                    is_active=True
                )
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    if "username" in str(e):
                        raise ValueError(f"Username '{username}' already exists")
                    elif "email" in str(e):
                        raise ValueError(f"Email '{email}' already exists")
                raise

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM users WHERE username = ? AND is_active = 1",
                (username,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return User(
                        id=row["id"],
                        username=row["username"],
                        email=row["email"],
                        hashed_password=row["hashed_password"],
                        full_name=row["full_name"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        is_active=bool(row["is_active"])
                    )
                return None

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM users WHERE email = ? AND is_active = 1",
                (email,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return User(
                        id=row["id"],
                        username=row["username"],
                        email=row["email"],
                        hashed_password=row["hashed_password"],
                        full_name=row["full_name"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        is_active=bool(row["is_active"])
                    )
                return None

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM users WHERE id = ? AND is_active = 1",
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return User(
                        id=row["id"],
                        username=row["username"],
                        email=row["email"],
                        hashed_password=row["hashed_password"],
                        full_name=row["full_name"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        is_active=bool(row["is_active"])
                    )
                return None

    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user by username/email and password"""
        # Try username first
        user = await self.get_user_by_username(username)
        if not user:
            # Try email
            user = await self.get_user_by_email(username)
        
        if not user:
            return None
        
        if not User.verify_password(password, user.hashed_password):
            return None
        
        return user

    async def update_user(
        self,
        user_id: int,
        email: Optional[str] = None,
        full_name: Optional[str] = None
    ) -> Optional[User]:
        """Update user profile"""
        await self.initialize()
        
        user = await self.get_user_by_id(user_id)
        if not user:
            return None
        
        updates = []
        params = []
        
        if email is not None and email != user.email:
            # Check if email already exists
            existing = await self.get_user_by_email(email)
            if existing and existing.id != user_id:
                raise ValueError(f"Email '{email}' already in use")
            updates.append("email = ?")
            params.append(email)
        
        if full_name is not None:
            updates.append("full_name = ?")
            params.append(full_name)
        
        if not updates:
            return user
        
        params.append(user_id)
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE id = ?",
                params
            )
            await db.commit()
        
        return await self.get_user_by_id(user_id)

    async def change_password(
        self,
        user_id: int,
        current_password: str,
        new_password: str
    ) -> bool:
        """Change user password"""
        await self.initialize()
        
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        
        # Verify current password
        if not User.verify_password(current_password, user.hashed_password):
            return False
        
        # Hash new password and update
        new_hashed = User.get_password_hash(new_password)
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE users SET hashed_password = ? WHERE id = ?",
                (new_hashed, user_id)
            )
            await db.commit()
        
        return True


# Global user database instance
user_db = UserDB()
