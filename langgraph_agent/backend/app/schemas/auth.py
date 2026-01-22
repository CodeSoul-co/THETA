"""
Authentication Schemas
User registration, login, and token management
"""

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional
from datetime import datetime
import re


class UserRegister(BaseModel):
    """User registration request"""
    username: str = Field(..., min_length=3, max_length=50, description="Username (3-50 characters)")
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=6, max_length=100, description="Password (min 6 characters)")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name (optional)")
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        """Validate username format"""
        if not re.match(r'^[a-zA-Z0-9_\u4e00-\u9fa5]+$', v):
            raise ValueError('用户名只能包含字母、数字、下划线和中文')
        return v
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        """Validate password strength"""
        if len(v) < 6:
            raise ValueError('密码长度至少为6个字符')
        return v


class UserLogin(BaseModel):
    """User login request"""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")


class UserResponse(BaseModel):
    """User information response"""
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    created_at: datetime
    is_active: bool = True


class UserUpdate(BaseModel):
    """User profile update request"""
    email: Optional[EmailStr] = Field(None, description="New email address")
    full_name: Optional[str] = Field(None, max_length=100, description="New full name")


class PasswordChange(BaseModel):
    """Password change request"""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=6, max_length=100, description="New password (min 6 characters)")
    
    @field_validator('new_password')
    @classmethod
    def validate_new_password(cls, v):
        """Validate new password strength"""
        if len(v) < 6:
            raise ValueError('新密码长度至少为6个字符')
        return v


class Token(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: Optional[UserResponse] = None  # Include user info in response


class TokenData(BaseModel):
    """Token payload data"""
    username: Optional[str] = None
    user_id: Optional[int] = None
