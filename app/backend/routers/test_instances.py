from fastapi import APIRouter, HTTPException, Response, status, Depends
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session, delete, select
from typing import List

import uuid
import io

from models import *
from schemas import *
from database import get_session

from services.utility import populate_spreadsheet_logic



router = APIRouter()


#region Test Instance Endpoints
@router.get("", response_model=List[TestInstanceResponse])
def get_test_instances(session: Session = Depends(get_session)):
    """Get all test instances with their items"""
    test_instances = session.exec(select(TestInstance)).all()
    
    if not test_instances:
        return []
    
    responses = []
    for instance in test_instances:
        items = session.exec(
                select(TestItem).where(TestItem.test_id == instance.test_id)
                ).all()
        
        summary_items = [
                TestItemSummary(
                    item_id=item.item_id,
                    label=item.label,
                    question=item.question,
                    is_problem_solving=item.is_problem_solving,
                    expected_answer_rubric_questions=item.expected_answer_rubric_questions
                ) for item in items
                ]
        
        responses.append(
                TestInstanceResponse(
                    name=instance.name,
                    section_id=instance.section_id,
                    date=instance.date,
                    test_id=instance.test_id,
                    is_done_rendering=instance.is_done_rendering,
                    items=summary_items
                )
                )
    
    return responses


@router.get("/{test_id}", response_model=TestInstanceResponse)
def get_test_instance(test_id: str, session: Session = Depends(get_session)):
    """Get specific test instance by ID with items"""
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance with ID '{test_id}' not found"
                )
    
    items = session.exec(
            select(TestItem).where(TestItem.test_id == test_id)
            ).all()
    
    summary_items = [
            TestItemSummary(
                item_id=item.item_id,
                label=item.label,
                question=item.question,
                is_problem_solving=item.is_problem_solving,
                expected_answer_rubric_questions=item.expected_answer_rubric_questions
            ) for item in items
            ]
    
    return TestInstanceResponse(
            name=instance.name,
            section_id=instance.section_id,
            date=instance.date,
            test_id=instance.test_id,
            is_done_rendering=instance.is_done_rendering,
            items=summary_items
            )


@router.post("", response_model=TestInstanceResponse, status_code=status.HTTP_201_CREATED)
def add_test_instance(
            request: NewTestInstanceRequest,
            session: Session = Depends(get_session)
            ):
    """Add new test instance"""
    # Verify section exists
    section = session.get(Section, request.section_id)
    if not section:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Section with ID {request.section_id} not found"
                )
    
    # Generate test_id (section-based naming)
    test_id = f"{section.section}_{request.name}_{uuid.uuid4().hex[:6]}"
    
    # Create new instance
    new_instance = TestInstance(
            test_id=test_id,
            name=request.name,
            section_id=request.section_id,
            date=request.date,
            is_done_rendering=False
            )
    session.add(new_instance)
    session.commit()
    session.refresh(new_instance)
    
    # Return with empty items array (no items created yet)
    return TestInstanceResponse(
            name=new_instance.name,
            section_id=new_instance.section_id,
            date=new_instance.date,
            test_id=new_instance.test_id,
            is_done_rendering=new_instance.is_done_rendering,
            items=[]
            )


@router.patch("/{test_id}", response_model=TestInstanceResponse)
def edit_test_instance(
            test_id: str,
            update: UpdateTestInstanceRequest,
            session: Session = Depends(get_session)
            ):
    """Edit test instance details including date and items"""
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance with ID '{test_id}' not found"
                )
    
    # Update date if provided
    if update.date is not None:
        instance.date = update.date
    
    # Update items if provided (REPLACE all items)
    if update.items is not None:
        # Delete existing items
        existing_items = session.exec(
            select(TestItem).where(TestItem.test_id == test_id)
        ).all()
        for item in existing_items:
            session.delete(item)
        
        # Create new items
        for idx, item_data in enumerate(update.items, start=1):
            question = item_data.question if item_data.question is not None else "Untitled Question"
            is_problem_solving = item_data.is_problem_solving if item_data.is_problem_solving is not None else False

            new_item = TestItem(
                item_id=idx,
                test_id=test_id,
                label=item_data.label or f"Item {idx}",
                question=question,
                is_problem_solving=is_problem_solving,
                expected_answer_rubric_questions=item_data.expected_answer_rubric_questions or ""
            )
            session.add(new_item)
    
    session.add(instance)
    session.commit()
    session.refresh(instance)
    
    # Get updated items
    items = session.exec(
            select(TestItem).where(TestItem.test_id == test_id)
            ).all()
    
    summary_items = [
            TestItemSummary(
                item_id=item.item_id,
                label=item.label,
                question=item.question,
                is_problem_solving=item.is_problem_solving,
                expected_answer_rubric_questions=item.expected_answer_rubric_questions
            ) for item in items
            ]
    
    return TestInstanceResponse(
            name=instance.name,
            section_id=instance.section_id,
            date=instance.date,
            test_id=instance.test_id,
            is_done_rendering=instance.is_done_rendering,
            items=summary_items
            )


@router.delete("/{test_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_test_instance(test_id: str, session: Session = Depends(get_session)):
    """Delete test instance and all related data"""
    instance = session.get(TestInstance, test_id)
    if not instance:
        raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test instance with ID '{test_id}' not found"
                )
    
    # Delete related test items
    items = session.exec(
            select(TestItem).where(TestItem.test_id == test_id)
            ).all()
    for item in items:
        session.delete(item)
    
    # Delete related test papers
    papers = session.exec(
            select(TestPaperInstance).where(TestPaperInstance.test_id == test_id)
            ).all()
    for paper in papers:
        # Delete related student answers first
        answers = session.exec(
                select(StudentAnswer).where(StudentAnswer.paper_id == paper.paper_id)
                ).all()
        for answer in answers:
            session.delete(answer)
        session.delete(paper)
    
    # Delete the instance itself
    session.delete(instance)
    session.commit()
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{test_id}/export")
def export_test_results(test_id: str, session: Session = Depends(get_session)):
    """Export test results as Excel spreadsheet"""
    workbook = populate_spreadsheet_logic(test_id, session)
    
    # Create DataFrame and Excel file
    output = io.BytesIO()
    
    workbook.save(output)
    
    output.seek(0)
    
    # Return as downloadable file
    headers = {
            "Content-Disposition": f"attachment; filename={test_id}.xlsx"
            }
    
    return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers
            )
#endregion
