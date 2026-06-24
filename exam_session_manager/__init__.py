# exam_session_manager - Handle active exam sessions, timing, and scoring
#
# Facade package: re-exports the exact public API of the original
# ``exam_session_manager.py`` module so all existing imports keep working.
import os
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from supabase import create_client

from .timing import _utcnow, _utcnow_iso, _parse_dt
from .schemas import ANSWER_JUDGE_SCHEMA, VERDICT_CREDIT
from .manager import ExamSessionManager

# Preserve the original module's import-time side effect.
load_dotenv()
