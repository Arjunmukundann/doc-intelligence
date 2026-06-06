from db.database import ConversationMessage, SessionLocal


def add_message(session_id: str, role: str, content: str):
    db = SessionLocal()
    try:
        message = ConversationMessage(
            session_id=session_id,
            role=role,
            content=content
        )
        db.add(message)
        db.commit()
    finally:
        db.close()


def get_history(session_id: str, limit: int = 6) -> list:
    db = SessionLocal()
    try:
        messages = (
            db.query(ConversationMessage)
            .filter(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.created_at.desc())
            .limit(limit)
            .all()
        )
        return list(reversed(messages))
    finally:
        db.close()


def format_history_for_prompt(session_id: str) -> str:
    messages = get_history(session_id)
    if not messages:
        return ""
    lines = []
    for msg in messages:
        prefix = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{prefix}: {msg.content}")
    return "\n".join(lines)   # returns string, nothing else


def clear_session(session_id: str):
    db = SessionLocal()
    try:
        db.query(ConversationMessage)\
          .filter(ConversationMessage.session_id == session_id)\
          .delete()
        db.commit()
    finally:
        db.close()