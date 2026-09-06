"""Person identity CRUD endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import db

router = APIRouter()


class PersonCreate(BaseModel):
    person_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=255)
    department: str = "default"
    role: str = "user"
    metadata: dict = Field(default_factory=dict)


@router.post("", status_code=201)
def create_person(payload: PersonCreate) -> dict:
    return db.create_person(
        person_id=payload.person_id,
        name=payload.name,
        department=payload.department,
        role=payload.role,
        metadata=payload.metadata,
    )


@router.get("")
def list_persons() -> list[dict]:
    return db.list_persons()


@router.get("/{person_id}")
def get_person(person_id: str) -> dict:
    person = db.get_person(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="person_id not found")
    return person


@router.delete("/{person_id}", status_code=204)
def delete_person(person_id: str) -> None:
    if not db.delete_person(person_id):
        raise HTTPException(status_code=404, detail="person_id not found")
