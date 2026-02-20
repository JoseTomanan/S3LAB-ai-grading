export interface EvaluationsResponse {
	item_id: number;
	answer_id: number,
	label: string;
	question: string;
	expected_answer_rubric_questions: string;
	ai_evaluation: string;
}

export interface StudentStoresResponse {
	test_id: string,
	student_no: string;
	name: string;
	evaluations: EvaluationsResponse[];
}