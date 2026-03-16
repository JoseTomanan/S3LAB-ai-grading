export interface GetSpecificEvaluationResponse {
	item_id: number;
	answer_id: number,
	label: string;
	question: string;
	expected_answer_rubric_questions: string;
	ai_evaluation: string;
	scores: string;
}

export interface StudentStoresResponse {
	test_id: string,
	student_no: string;
	name: string;
	evaluations: GetSpecificEvaluationResponse[];
}

export interface GetEvaluationsResponse {
	student_no: string;
	name: string;
	is_done_rendering: boolean;
	total_score: string;
}

export interface CommitBoxesResponseItem {
  index: number;
  image_directory: string;
  item_number: string;
}