from datetime import datetime, timedelta
from typing import Optional, List
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# Secret + JWT config (load from env in production)
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Auth mechanism
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 🔐 Fake in-memory user store
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "hashed_password": pwd_context.hash("password123"),
        "disabled": False,
        "role": "admin"
    },
    "responder": {
        "username": "responder",
        "full_name": "On-Site Agent",
        "hashed_password": pwd_context.hash("responderpass"),
        "disabled": False,
        "role": "responder"
    },
    "viewer": {
        "username": "viewer",
        "full_name": "Read-Only User",
        "hashed_password": pwd_context.hash("viewerpass"),
        "disabled": False,
        "role": "viewer"
    }
}

# 🧠 Auth Utilities
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    return db.get(username)

def authenticate_user(username: str, password: str):
    user = get_user(fake_users_db, username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})

    # Inject role from fake DB (use DB in prod)
    user = get_user(fake_users_db, data["sub"])
    if user:
        to_encode.update({"role": user.get("role")})

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# 👤 Used in all protected routes
def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise credentials_exception
        user = get_user(fake_users_db, username)
        if not user:
            raise credentials_exception
        return user
    except JWTError:
        raise credentials_exception

# 🛡️ Role-Based Access Dependency
def require_role(required_roles: List[str]):
    def role_checker(user: dict = Depends(get_current_user)):
        user_role = user.get("role", "")
        if user_role not in required_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied: Role '{user_role}' not allowed.",
            )
        return user
    return role_checker
