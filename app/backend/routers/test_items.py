from fastapi import APIRouter, HTTPException, Response, status, Depends, File, UploadFile, Form, Query
from sqlmodel import Session, delete, select

from models import *
from schemas import *
from core.database import get_session

from logic.utility import *



router = APIRouter()


#region Endpoints
@router.get("/{test_id}/items")
def get_test_instance_items(
            test_id: str,
            session: Session = Depends(get_session),
            ):
    """Get all test items for a specific test instance"""
    # Verify test instance exists
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test instance '{test_id}' not found"
        )
    
    # Get items
    items = session.exec(
        select(TestItem).where(TestItem.test_id == test_id)
    ).all()
    
    # Build items list
    items_list = [
        {
            "item_id": item.item_id,
            "label": item.label,
            "question": item.question,
            "is_problem_solving": item.is_problem_solving,
            "expected_answer_rubric_questions": item.expected_answer_rubric_questions,
        }
        for item in items
    ]
    
    return {
        "test_id": test_id,
        "items": items_list
    }

@router.post("/{test_id}/items", response_model=NewTestItemResponse)
def add_test_item(
            test_id: str,
            item: NewTestItemRequest,
            session: Session = Depends(get_session)
            ):
    """Add new test item (item_id auto-generated)"""
    # Verify test instance exists
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance '{test_id}' not found"
                )
    
    # Determine next item_id (max existing + 1)
    max_item = session.exec(
                    select(TestItem)
                    .order_by(TestItem.item_id.desc())  # type: ignore[union-attr]
                    ).first()

    next_item_id = (max_item.item_id + 1) if max_item else 1

    # Create new test item
    question = item.question if item.question is not None else "Untitled Question"
    is_problem_solving = item.is_problem_solving if item.is_problem_solving is not None else False

    new_item = TestItem(
        item_id=next_item_id,
        test_id=test_id,
        label=item.label or f"Item {next_item_id}",
        question=question,
        is_problem_solving=is_problem_solving,
        expected_answer_rubric_questions=item.expected_answer_rubric_questions or ""
    )
    session.add(new_item)
    session.commit()
    session.refresh(new_item)
    
    # Return full item representation
    return NewTestItemResponse(
            items=[
                FullTestItemResponse(
                    item_id=new_item.item_id,
                    label=new_item.label,
                    question=new_item.question,
                    is_problem_solving=new_item.is_problem_solving,
                    expected_answer_rubric_questions=new_item.expected_answer_rubric_questions
                )
            ])


@router.patch("/{test_id}/items/{item_id}", response_model=FullTestItemResponse)
def edit_test_item(
            test_id: str,
            item_id: int,
            update: UpdateTestItemRequest,
            session: Session = Depends(get_session)
            ):
    """Edit test item details"""
    # Verify test instance exists
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance '{test_id}' not found"
                )
    
    # Find item
    item = session.exec(
            select(TestItem).where(
                TestItem.item_id == item_id,
                TestItem.test_id == test_id
            )
            ).first()
    
    if not item:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test item with ID {item_id} not found in test instance '{test_id}'"
                )
    
    # Update fields
    update_data = update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(item, field, value)
    
    session.add(item)
    session.commit()
    session.refresh(item)
    
    return FullTestItemResponse(
            item_id=item.item_id,
            label=item.label,
            question=item.question,
            is_problem_solving=item.is_problem_solving,
            expected_answer_rubric_questions=item.expected_answer_rubric_questions
            )


@router.delete("/{test_id}/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_test_item(
            test_id: str,
            item_id: int,
            session: Session = Depends(get_session)
            ):
    """Delete test item"""
    # Verify test instance exists
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance '{test_id}' not found"
                )
    
    # Find and delete item
    item = session.exec(
            select(TestItem).where(
                TestItem.item_id == item_id,
                TestItem.test_id == test_id
                )
            ).first()
    
    if not item:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test item with ID {item_id} not found in test instance '{test_id}'"
                )
    
    # Delete related student answers first
    answers = session.exec(
            select(StudentAnswer)
            .join(TestPaperInstance)
            .where(
                TestPaperInstance.test_id == test_id,
                StudentAnswer.item_id == item_id
            )
            ).all()
    
    for answer in answers:
        session.delete(answer)
    
    session.delete(item)
    session.commit()
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)
#endregion
