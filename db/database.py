from sqlalchemy import create_engine,Column,String,Integer,DateTime,Text
from sqlalchemy.orm import declarative_base,sessionmaker
from datetime import datetime
import uuid

engine=create_engine("sqlite:///./documents.db",connect_args={"check_same_thread":False})
SessionLocal=sessionmaker(bind=engine)
Base=declarative_base()


class Document(Base):
    __tablename__="documents"
    id=Column(String,primary_key=True,default=lambda: str(uuid.uuid4())) 
    filename=Column(String,nullable=False)
    file_path=Column(String,nullable=False)
    status=Column(String,default='processing')
    chunk_count=Column(Integer,default=0)
    error=Column(Text,nullable=True)
    created_at=Column(DateTime,default=datetime.utcnow)



class ConversationMessage(Base):
    __tablename__="conversation_message"
    id=Column( Integer,primary_key=True,index=True)
    session_id=Column(String,nullable=False)
    role=Column(String,nullable=False)
    content=Column(Text,nullable=False)
    created_at=Column(DateTime,default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()